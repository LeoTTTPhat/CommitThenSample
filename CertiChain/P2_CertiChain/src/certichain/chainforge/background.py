"""ChainForge background library (DESIGN §5.4.1): (dataset, devices and label-free
port-profile roles, span, pool) + the D3 benign look-alike catalogue.

* Backgrounds are the benign captures prepared by P2-A (TON: normal_Bro/normal_IoT_1 → train,
  normal_IoT_2 → caltest; Gotham: the 01-14 benign capture split in time). Windows B are drawn
  at random offsets inside a pool's span; their devices are the internal unicast hosts seen in
  the window.
* Template-compatible devices of a window (Alg. 7 step 3): internal unicast hosts that
  originate ≥ 1 record in B and are contacted by fewer than F_infra distinct internal hosts in B
  (a would-be INFRA host, e.g. a DNS server or broker, is never in scope and could not be a
  victim whose activity is visible). Port profiles (label-free) are reported per device.
* D3 look-alikes: the pool background is tiled into 60-min tiles; each tile is sessionised
  label-free with P2-B's Alg. 1 steps 1–5 (``certichain.units.sessionize``, no scope) and
  every unit whose type is attack-like (not a benign-protocol type) is catalogued with its
  record pointers. A D3 decoy re-hosts one such unit from a tile disjoint from B onto a campaign
  victim.

Files (results/interim/<ds>/library/):
    background.json            per pool: source, span, n_records, devices + port-profile roles
    lookalikes_<pool>.parquet  la_id, pool, tile, type, src, svc_class, dir_class, ts, te,
                               n_records, rec_ids (original background rec_ids)
"""
from __future__ import annotations

import ipaddress
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from certichain.io.paths import atomic_write_json, atomic_write_parquet, interim, log

F_INFRA = 3
TILE_S = 3600.0
LOOKALIKE_MAX_RECORDS = 5000
LOOKALIKE_MAX_DUR = 1800.0
BENIGN_PROTOCOL_TYPES = frozenset({"MQTT_PUBSUB", "COAP", "RTSP_STREAM", "DNS_ACTIVITY", "TLS_SESSION",
                                   "BENIGN_GENERIC_TCP", "BENIGN_GENERIC_UDP"})


def _internal_unicast(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if a.version != 4:
        return False
    private = any(a in ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
    return private and not a.is_multicast and not str(a).endswith(".255") and not str(a).endswith(".0")


def ip_class_frame(ips) -> pl.DataFrame:
    ips = sorted({i for i in ips if i is not None})
    return pl.DataFrame({"ip": ips, "iu": [_internal_unicast(i) for i in ips]}, schema={"ip": pl.Utf8, "iu": pl.Boolean})


def background_path(ds: str, pool: str) -> Path:
    return interim(ds, "backgrounds", f"{pool}.parquet")


# --------------------------------------------------------------------------- device tables
def device_table(rec: pl.DataFrame) -> pl.DataFrame:
    """Per internal unicast device: activity and label-free port-profile roles."""
    if rec.height == 0:
        return pl.DataFrame(schema={"ip": pl.Utf8, "n_out": pl.UInt32, "n_in": pl.UInt32, "n_int_contactors": pl.UInt32,
                                    "n_peers": pl.UInt32, "active": pl.Boolean})
    cls = ip_class_frame(set(rec["src"].to_list()) | set(rec["dst"].to_list()))
    r = (rec.select("src", "dst", "dport", "proto")
         .join(cls.rename({"ip": "src", "iu": "src_iu"}), on="src", how="left")
         .join(cls.rename({"ip": "dst", "iu": "dst_iu"}), on="dst", how="left"))
    out_ = r.filter(pl.col("src_iu")).group_by(pl.col("src").alias("ip")).agg(
        pl.len().alias("n_out"), pl.col("dst").n_unique().alias("n_dst"),
        pl.col("dport").drop_nulls().value_counts(sort=True).head(5).alias("_top_used"))
    in_ = r.filter(pl.col("dst_iu")).group_by(pl.col("dst").alias("ip")).agg(
        pl.len().alias("n_in"), pl.col("src").filter(pl.col("src_iu") & (pl.col("src") != pl.col("dst")))
        .n_unique().alias("n_int_contactors"),
        pl.col("src").n_unique().alias("n_src"),
        pl.col("dport").drop_nulls().value_counts(sort=True).head(5).alias("_top_served"))
    dev = out_.join(in_, on="ip", how="full", coalesce=True)
    dev = dev.with_columns(pl.col("n_out").fill_null(0), pl.col("n_in").fill_null(0),
                           pl.col("n_int_contactors").fill_null(0), pl.col("n_dst").fill_null(0),
                           pl.col("n_src").fill_null(0))
    return dev.with_columns((pl.col("n_out") > 0).alias("active"),
                            (pl.col("n_dst") + pl.col("n_src")).alias("n_peers")).sort("ip")


def port_profile_roles(rec: pl.DataFrame) -> dict[str, list[str]]:
    """Label-free descriptive roles per device from ports (for Table III / notes)."""
    cls = ip_class_frame(set(rec["src"].to_list()) | set(rec["dst"].to_list()))
    r = (rec.select("src", "dst", "dport", "proto")
         .join(cls.rename({"ip": "src", "iu": "src_iu"}), on="src", how="left")
         .join(cls.rename({"ip": "dst", "iu": "dst_iu"}), on="dst", how="left"))
    roles: dict[str, set[str]] = {}

    def add(ips, role):
        for i in ips:
            roles.setdefault(i, set()).add(role)

    srv = r.filter(pl.col("dst_iu") & pl.col("src_iu"))
    add(srv.filter((pl.col("dport") == 53))["dst"].unique().to_list(), "dns_server")
    add(srv.filter(pl.col("dport").is_in([1883, 8883]))["dst"].unique().to_list(), "mqtt_broker")
    add(srv.filter(pl.col("dport").is_in([80, 8080, 443, 1880]))["dst"].unique().to_list(), "web_server")
    cli = r.filter(pl.col("src_iu"))
    add(cli.filter(pl.col("dport") == 5353)["src"].unique().to_list(), "mdns")
    add(cli.filter(pl.col("dport").is_in([137, 138, 139]))["src"].unique().to_list(), "netbios")
    add(cli.filter(pl.col("dst").str.ends_with(".255"))["src"].unique().to_list(), "broadcaster")
    add(cli.filter(~pl.col("dst_iu").fill_null(False) & pl.col("dport").is_in([80, 443]))["src"].unique().to_list(),
        "cloud_client")
    add(cli.filter(pl.col("dport") == 123)["src"].unique().to_list(), "ntp_client")
    add(cli.filter(pl.col("dport") == 53)["src"].unique().to_list(), "dns_client")
    add(cli.filter(pl.col("src") == pl.col("dst"))["src"].unique().to_list(), "self_loop")
    peers = (cli.filter(pl.col("dst_iu")).group_by("src").agg(pl.col("dst").n_unique().alias("n")))
    add(peers.filter(pl.col("n") >= 10)["src"].to_list(), "hub")
    return {k: sorted(v) for k, v in sorted(roles.items())}


# --------------------------------------------------------------------------- look-alikes
def build_lookalikes(rec: pl.DataFrame, pool: str, tile_s: float = TILE_S) -> pl.DataFrame:
    """Attack-like benign units per 60-min tile (label-free Alg. 1 steps 1–5, no scope)."""
    from certichain.units import sessionize as S

    schema = {"la_id": pl.Utf8, "pool": pl.Utf8, "tile": pl.Int32, "tile_t0": pl.Float64, "tile_t1": pl.Float64,
              "type": pl.Utf8, "src": pl.Utf8, "svc_class": pl.Utf8, "dir_class": pl.Utf8, "ts": pl.Float64,
              "te": pl.Float64, "ts_last": pl.Float64, "n_records": pl.Int32, "n_dst": pl.Int32,
              "rec_ids": pl.List(pl.Int64)}
    if rec.height == 0:
        return pl.DataFrame(schema=schema)
    t0 = float(rec["ts"].min())
    n_tiles = int(np.ceil((float(rec["ts"].max()) - t0 + 1) / tile_s))
    rows = []
    for k in range(n_tiles):
        a, b = t0 + k * tile_s, t0 + (k + 1) * tile_s
        sub = rec.filter((pl.col("ts") >= a) & (pl.col("ts") < b))
        if sub.height == 0:
            continue
        prep, _ = S.prepare_records(sub)
        res = S.sessionize(S.default_annotations(prep))
        u = res.units
        if u.height == 0:
            continue
        u = u.filter(~pl.col("type").is_in(list(BENIGN_PROTOCOL_TYPES))
                     & (pl.col("n_records") <= LOOKALIKE_MAX_RECORDS)
                     & ((pl.col("te") - pl.col("ts")) <= LOOKALIKE_MAX_DUR))
        if u.height == 0:
            continue
        u = u.filter(pl.col("src").map_elements(_internal_unicast, return_dtype=pl.Boolean))
        tsmap = dict(zip(sub["rec_id"].to_list(), sub["ts"].to_list()))
        for d in u.sort(["ts", "src", "svc_class", "type"]).iter_rows(named=True):
            ids = sorted(int(x) for x in d["rec_ids"])
            rows.append({"pool": pool, "tile": k, "tile_t0": a, "tile_t1": b, "type": d["type"], "src": d["src"],
                         "svc_class": d["svc_class"], "dir_class": d["dir_class"], "ts": float(d["ts"]),
                         "te": float(d["te"]), "ts_last": float(max(tsmap[i] for i in ids)),
                         "n_records": int(d["n_records"]), "n_dst": int(d["n_dst"]), "rec_ids": ids})
    out = pl.DataFrame(rows, schema={k: v for k, v in schema.items() if k != "la_id"}) if rows else \
        pl.DataFrame(schema={k: v for k, v in schema.items() if k != "la_id"})
    out = out.with_columns(pl.format("{}-la{}", pl.lit(pool), pl.int_range(pl.len()).cast(pl.Utf8).str.zfill(4))
                           .alias("la_id"))
    return out.select(list(schema))


def build_background_library(ds: str, pools: dict[str, dict], force: bool = False, tile_s: float = TILE_S) -> dict:
    """pools: {pool: {"source": str, "path": Path}} → background.json + lookalikes_<pool>.parquet."""
    out_dir = interim(ds, "library")
    js = out_dir / "background.json"
    if js.exists() and not force:
        log(f"{ds} background library exists ({js}); --force to rebuild")
        return json.loads(js.read_text())
    t_start = time.time()
    info = {"dataset": ds, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pools": {},
            "config": {"F_infra": F_INFRA, "tile_s": tile_s, "lookalike_max_records": LOOKALIKE_MAX_RECORDS,
                       "lookalike_max_dur_s": LOOKALIKE_MAX_DUR}}
    for pool, spec in pools.items():
        rec = pl.read_parquet(spec["path"])
        dev = device_table(rec)
        roles = port_profile_roles(rec)
        la = build_lookalikes(rec, pool, tile_s=tile_s)
        atomic_write_parquet(la, out_dir / f"lookalikes_{pool}.parquet")
        info["pools"][pool] = {
            "source": spec.get("source"), "path": str(spec["path"]), "n_records": rec.height,
            "t0": float(rec["ts"].min()), "t1": float(rec["ts"].max()),
            "span_h": round((float(rec["ts"].max()) - float(rec["ts"].min())) / 3600.0, 3),
            "n_devices": dev.height, "n_active_devices": int(dev["active"].sum()),
            "devices": [{**{k: v for k, v in d.items() if not k.startswith("_")}, "roles": roles.get(d["ip"], [])}
                        for d in dev.iter_rows(named=True)],
            "n_lookalikes": la.height,
            "lookalike_types": la.group_by("type").len().sort("type").to_dicts() if la.height else [],
        }
        log(f"{ds}/{pool}: {rec.height} background records, {dev.height} devices, {la.height} look-alike units")
    info["runtime_s"] = round(time.time() - t_start, 1)
    atomic_write_json(js, info)
    return info


# --------------------------------------------------------------------------- runtime access
@dataclass
class BackgroundLibrary:
    ds: str
    info: dict
    _rec: dict = field(default_factory=dict)
    _la: dict = field(default_factory=dict)

    @classmethod
    def load(cls, ds: str) -> "BackgroundLibrary":
        return cls(ds=ds, info=json.loads((interim(ds, "library") / "background.json").read_text()))

    def span(self, pool: str) -> tuple[float, float]:
        p = self.info["pools"][pool]
        return float(p["t0"]), float(p["t1"])

    def records(self, pool: str) -> pl.DataFrame:
        if pool not in self._rec:
            self._rec[pool] = pl.read_parquet(self.info["pools"][pool]["path"]).sort(["ts", "rec_id"])
        return self._rec[pool]

    def window(self, pool: str, t0: float, t1: float) -> pl.DataFrame:
        r = self.records(pool)
        lo = int(np.searchsorted(r["ts"].to_numpy(), t0, side="left"))
        hi = int(np.searchsorted(r["ts"].to_numpy(), t1, side="left"))
        return r.slice(lo, hi - lo)

    def by_rec_ids(self, pool: str, rec_ids) -> pl.DataFrame:
        return self.records(pool).filter(pl.col("rec_id").is_in(pl.Series(list(rec_ids), dtype=pl.Int64).implode()))

    def lookalikes(self, pool: str) -> pl.DataFrame:
        if pool not in self._la:
            p = interim(self.ds, "library", f"lookalikes_{pool}.parquet")
            self._la[pool] = pl.read_parquet(p) if p.exists() else pl.DataFrame()
        return self._la[pool]


def compatible_devices(win: pl.DataFrame, F_infra: int = F_INFRA) -> list[str]:
    dev = device_table(win)
    if dev.height == 0:
        return []
    ok = dev.filter(pl.col("active") & (pl.col("n_int_contactors") < F_infra))
    return sorted(ok["ip"].to_list())


def internal_neighbours(win: pl.DataFrame, x: str) -> list[str]:
    """Internal unicast hosts that exchanged ≥ 1 record with x in the window (N1 of x)."""
    a = win.filter(pl.col("src") == x)["dst"].to_list()
    b = win.filter(pl.col("dst") == x)["src"].to_list()
    return sorted({h for h in a + b if h != x and _internal_unicast(h)})


__all__ = ["F_INFRA", "TILE_S", "BENIGN_PROTOCOL_TYPES", "device_table", "port_profile_roles", "build_lookalikes",
           "build_background_library", "BackgroundLibrary", "compatible_devices", "internal_neighbours",
           "background_path", "ip_class_frame"]
