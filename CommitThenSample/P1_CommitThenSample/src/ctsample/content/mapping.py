"""Alert → unit mapping and the subset-invariance audit (DESIGN §5.4).

* Replay runs: the eve ``vlan`` pair of an alert gives the replay id rid
  (``replay.unit_from_vlans``); units/<ds>/<source>/part-*.parquet maps rid → unit_gid.
* Original-pcap runs (observability gap, SHOULD): ``pcap_cnt`` → unit_gid via pktmap.
* Subset audit: on the units of the 10 % subset (common to both runs), per sid,
  A = units alerting in the full replay, B = units alerting in the subset replay;
  keep the sid iff |A ∩ B| / |A ∪ B| ≥ 0.95 and |A ∪ B| ≥ 5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ctsample.content.replay import VLAN_N
from ctsample.utils.io import RESULTS

AGREE_MIN = 0.95
UNITS_MIN = 5


def rid_table(ds: str, source: str) -> np.ndarray:
    """gid_by_rid[rid] = unit_gid for the payload units of one source."""
    d = RESULTS / "units" / ds / source
    parts = sorted(d.glob("part-*.parquet"))
    rids, gids = [], []
    for p in parts:
        t = pq.read_table(p, columns=["rid", "unit_gid"])
        r = t.column("rid").to_numpy()
        g = t.column("unit_gid").to_numpy()
        m = r >= 0
        rids.append(r[m])
        gids.append(g[m])
    if not rids:
        return np.zeros(0, np.int64)
    r = np.concatenate(rids)
    g = np.concatenate(gids)
    out = np.full(int(r.max()) + 1 if len(r) else 0, -1, np.int64)
    out[r] = g
    return out


def map_vlans(outer: np.ndarray, inner: np.ndarray, gid_by_rid: np.ndarray) -> np.ndarray:
    outer = np.asarray(outer, np.int64)
    inner = np.asarray(inner, np.int64)
    ok = (outer >= 1) & (outer <= VLAN_N) & (inner >= 1) & (inner <= VLAN_N)
    rid = np.where(ok, (outer - 1) * VLAN_N + (inner - 1), -1)
    ok &= rid < len(gid_by_rid)
    out = np.full(len(rid), -1, np.int64)
    out[ok] = gid_by_rid[rid[ok]]
    return out


def map_pcap_cnt(pcap_cnt: np.ndarray, pktmap_path: str | Path) -> np.ndarray:
    z = np.load(pktmap_path)
    g = z["unit_gid"]
    c = np.asarray(pcap_cnt, np.int64) - 1
    ok = (c >= 0) & (c < len(g))
    out = np.full(len(c), -1, np.int64)
    out[ok] = g[c[ok]]
    return out


def subset_audit(full: dict[int, set], sub: dict[int, set], common: set,
                 agree_min: float = AGREE_MIN, units_min: int = UNITS_MIN) -> dict:
    """full/sub: sid -> set of unit_gids alerting; common: unit_gids present in both runs."""
    res = {}
    for sid in sorted(set(full) | set(sub)):
        a = full.get(sid, set()) & common
        b = sub.get(sid, set())
        u = a | b
        i = a & b
        agree = len(i) / len(u) if u else None
        res[sid] = {"n_full_all": len(full.get(sid, ())), "n_full_common": len(a), "n_sub": len(b),
                    "n_union": len(u), "n_both": len(i), "agreement": agree,
                    "keep": bool(u) and agree >= agree_min and len(u) >= units_min}
    return res


def observability_gap(ds: str, kept: set[int], sources: list[str]) -> dict:
    """gap_k = 1 − |units with a full-pcap kept-sid alert of family k reproduced by the per-unit replay| /
    |units with a full-pcap kept-sid alert of family k| (DESIGN §5.4, SHOULD). Original-pcap alerts are
    attributed to units through pcap_cnt → pktmap; alerts without pcap_cnt (stream inspection at flow
    end, pseudo packets) cannot be attributed to a segment and are counted separately."""
    import polars as pl
    orig, rep, n_unattr, n_orig = [], [], 0, 0
    for s in sources:
        po = RESULTS / "alerts" / ds / f"{s}.orig.parquet"
        pr = RESULTS / "alerts" / ds / f"{s}.parquet"
        if not (po.exists() and pr.exists()):
            continue
        o = pl.read_parquet(po, columns=["unit_gid", "sid", "family", "pcap_cnt"]).filter(pl.col("sid").is_in(sorted(kept)))
        n_orig += o.height
        n_unattr += int((o["unit_gid"] < 0).sum())
        orig.append(o.filter(pl.col("unit_gid") >= 0))
        rep.append(pl.read_parquet(pr, columns=["unit_gid", "sid", "family"]).filter(
            pl.col("sid").is_in(sorted(kept)) & (pl.col("unit_gid") >= 0)))
    if not orig:
        return {"dataset": ds, "available": False}
    o = pl.concat(orig).with_columns(pl.col("family").fill_null("none"))
    r = pl.concat(rep).with_columns(pl.col("family").fill_null("none"))
    out = {"dataset": ds, "available": True, "n_sources": len(orig), "n_orig_alerts_kept_sids": n_orig,
           "n_orig_alerts_unattributed_no_pcap_cnt": n_unattr, "families": {}}
    fams = sorted(set(o["family"].unique().to_list()) | set(r["family"].unique().to_list())) + ["ANY"]
    for f in fams:
        so = set((o if f == "ANY" else o.filter(pl.col("family") == f))["unit_gid"].to_list())
        sr = set((r if f == "ANY" else r.filter(pl.col("family") == f))["unit_gid"].to_list())
        out["families"][f] = {"units_orig": len(so), "units_replay": len(sr), "units_both": len(so & sr),
                              "gap": (1 - len(so & sr) / len(so)) if so else None,
                              "units_replay_only": len(sr - so)}
    return out
