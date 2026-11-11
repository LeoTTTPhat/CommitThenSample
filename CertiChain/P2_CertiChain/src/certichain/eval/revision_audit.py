"""P2-X revision: IoT-23 claim audit (review issue 2; R2-M2, R3-M3). Owner: P2-X. POST-PREREG, EXPLORATORY.

Lists every claim certified on IoT-23 by CRC-BH (and the other P-LOSO rules on request) with its truth, the true
links in C and which method finds them, and the split of rho_T into the definitional hidden R0 transition
(EXEC(hidden) -> first malicious seed episode, one per malicious window; gt/iot23_episodes.py) and other hidden
transitions. GT-side (labels, transitions, gt_units are read through the driver's guarded readers; the PREREG
exists, so the guard is open for evaluation).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import polars as pl

from ..certify.losses import LossBank
from . import experiments as X
from . import revision as RV
from .splits import random_splits

R0_SRC = "EXEC(hidden)"


def _window_tables(inc: str) -> dict:
    from ..scoring import driver as D
    ds = "iot23"
    cands = D.load_cands(ds, inc)
    units = D.load_units(ds, inc)
    lab = D.load_labels(ds, inc)
    gtu = D.load_gt_units(ds, inc)
    tr = D.load_transitions(ds, inc)
    return {"cands": cands, "units": units, "lab": lab, "gtu": gtu, "tr": tr}


def _unit_info(units: pl.DataFrame, gtu: pl.DataFrame) -> dict[int, dict]:
    ep = dict(zip(gtu["unit_id"].to_list(), gtu["episode"].to_list())) if gtu.height else {}
    mx = dict(zip(gtu["unit_id"].to_list(), gtu["mixed"].to_list())) if gtu.height else {}
    out = {}
    for r in units.select("unit_id", "ts", "te", "src", "dsts", "type", "svc_class", "dir_class", "n_records",
                          "dports").iter_rows(named=True):
        uid = int(r["unit_id"])
        e = ep.get(uid)
        out[uid] = {"type": r["type"], "src": r["src"], "dst0": (r["dsts"] or [None])[0], "n_dst": len(r["dsts"] or []),
                    "svc": r["svc_class"], "dir": r["dir_class"], "ts": float(r["ts"]), "te": float(r["te"]),
                    "n_rec": int(r["n_records"]), "episode": e, "mixed": bool(mx.get(uid, False)),
                    "malicious": e is not None or bool(mx.get(uid, False)),
                    "dport0": (r["dports"] or [None])[0]}
    return out


def truth_class(y: int, iu: dict, iv: dict) -> str:
    """Readable truth of a claim from its label and the GT kinds of its endpoints."""
    if y == 1:
        return "TRUE link (Y=1)"
    if y == -1:
        if iu["episode"] is not None and iu["episode"] == iv["episode"]:
            return "AMB: same episode"
        if iu["mixed"] or iv["mixed"]:
            return "AMB: mixed unit"
        return "AMB: malicious pair not in T* (transitive / reverse / unexplained)"
    mu, mv = iu["malicious"], iv["malicious"]
    if not mu and not mv:
        return "FALSE: benign -> benign"
    if mu and mv:
        return "FALSE: malicious -> malicious (not a transition)"
    return "FALSE: benign <-> malicious"


def _tables(inc: str, cache: dict) -> dict:
    if inc not in cache:
        t = _window_tables(inc)
        t["uinfo"] = _unit_info(t["units"], t["gtu"])
        t["ylab"] = dict(zip(t["lab"]["cand_id"].to_list(), t["lab"]["y"].to_list()))
        trs = t["tr"]
        t["trans_of"] = {int(c): f"{a} -> {b}" for a, b, c, kd in zip(trs["src_ep"].to_list(), trs["dst_ep"].to_list(),
                                                                      trs["cand_id"].to_list(), trs["kind"].to_list())
                         if c is not None and kd == "obs"} if trs.height else {}
        t["pos_of"] = {int(c): i for i, c in enumerate(t["cands"]["cand_id"].to_list())}
        cache[inc] = t
    return cache[inc]


def _claim_row(t: dict, iid: str, cid: int, extra: dict) -> dict:
    pos = t["pos_of"][int(cid)]
    u, v = int(t["cands"]["u"].to_numpy()[pos]), int(t["cands"]["v"].to_numpy()[pos])
    iu, iv = t["uinfo"][u], t["uinfo"][v]
    y = int(t["ylab"][int(cid)])
    return {**extra, "scenario": X.scenario_of(iid), "window": X.inc_of(iid), "cand_id": int(cid), "y": y,
            "truth": truth_class(y, iu, iv), "transition": t["trans_of"].get(int(cid)), "u": u, "v": v,
            "u_type": iu["type"], "v_type": iv["type"], "u_src": iu["src"], "u_dst": iu["dst0"], "v_src": iv["src"],
            "v_dst": iv["dst0"], "u_svc": iu["svc"], "v_svc": iv["svc"], "u_dport": iu["dport0"], "v_dport": iv["dport0"],
            "u_episode": iu["episode"], "v_episode": iv["episode"], "u_nrec": iu["n_rec"], "v_nrec": iv["n_rec"],
            "lag_s": float(iv["ts"] - iu["ts"])}


def claims_frame(bank: LossBank, rows: np.ndarray, R: np.ndarray, *, protocol: str, method: str, sid: str,
                 alpha: float, lam_idx: Sequence | None = None, cache: dict | None = None) -> pl.DataFrame:
    """One row per certified claim of the windows ``rows`` with selection sizes ``R`` (top-R in score order)."""
    cache = {} if cache is None else cache
    out = []
    for k, (row, r) in enumerate(zip(np.asarray(rows).tolist(), np.asarray(R).tolist())):
        if r <= 0:
            continue
        iid = bank.index["incident_id"][row]
        t = _tables(X.inc_of(iid), cache)
        sl = bank.slice(row)
        pos = bank.csr["cand"][sl][:r]
        sc = bank.csr["score"][sl][:r]
        pv = bank.csr["p"][sl][:r]
        yb = bank.csr["y"][sl][:r]
        cid = t["cands"]["cand_id"].to_numpy()[pos]
        for j in range(r):
            d = _claim_row(t, iid, int(cid[j]), {"protocol": protocol, "method": method, "scorer_id": sid,
                                                 "alpha": alpha, "lam_idx": None if lam_idx is None else lam_idx[k],
                                                 "rank": j + 1, "R_window": int(r), "score": float(sc[j]),
                                                 "ref_p": float(pv[j])})
            if d["y"] != int(yb[j]):
                raise RuntimeError(f"{iid}: bank label of cand {cid[j]} differs from labels_pairs")
            out.append(d)
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


def fixed_claims_frame(bank: LossBank, claims: pl.DataFrame, name: str, cache: dict | None = None) -> pl.DataFrame:
    """Truth rows for a fixed baseline claim set (claims.parquet: incident, cand_id, score)."""
    cache = {} if cache is None else cache
    ids = {X.inc_of(i): i for i in bank.index["incident_id"].to_list()}
    out = []
    for inc, cids in claims.group_by("incident").agg(pl.col("cand_id")).iter_rows():
        iid = ids.get(inc)
        if iid is None:
            continue
        t = _tables(inc, cache)
        for c in cids:
            out.append(_claim_row(t, iid, int(c), {"protocol": "fixed", "method": name}))
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


def true_links(bank: LossBank, cache: dict | None = None) -> pl.DataFrame:
    """Every Y = 1 candidate of the IoT-23 bank (the true links in C) with rank / reference p / the smallest BH grid
    index that certifies it."""
    cache = {} if cache is None else cache
    rows = []
    Rbh = bank.t("R", "bh")
    grid = bank.grids["bh"]
    for row in range(bank.n):
        sl = bank.slice(row)
        yb = bank.csr["y"][sl]
        idx = np.flatnonzero(yb == 1)
        if idx.size == 0:
            continue
        iid = bank.index["incident_id"][row]
        inc = X.inc_of(iid)
        t = _tables(inc, cache)
        for j in idx.tolist():
            pos = int(bank.csr["cand"][sl][j])
            cid = int(t["cands"]["cand_id"].to_numpy()[pos])
            u, v = int(t["cands"]["u"].to_numpy()[pos]), int(t["cands"]["v"].to_numpy()[pos])
            rank = j + 1
            ok = np.flatnonzero(Rbh[row] >= rank)
            jmin = int(ok[0]) if ok.size else None
            iu, iv = t["uinfo"][u], t["uinfo"][v]
            rows.append({"scenario": X.scenario_of(iid), "window": inc, "row": row, "cand_id": cid, "rank": rank,
                         "N": int(sl.stop - sl.start), "score": float(bank.csr["score"][sl][j]),
                         "ref_p": float(bank.csr["p"][sl][j]), "min_bh_idx": jmin,
                         "min_bh_lambda": float(grid[jmin]) if jmin is not None else None,
                         "transition": t["trans_of"].get(cid), "u_type": iu["type"], "v_type": iv["type"],
                         "u_src": iu["src"], "u_dst": iu["dst0"], "v_dst": iv["dst0"], "lag_s": float(iv["ts"] - iu["ts"])})
    return pl.DataFrame(rows, infer_schema_length=None)


def baseline_claim_sets(ds: str = "iot23") -> dict[str, pl.DataFrame]:
    out = {}
    for b in ("B1", "B2", "B4"):
        if X.has_baseline(ds, b):
            out[b] = pl.read_parquet(X.baseline_dir(ds, b) / "claims.parquet")
    return out


def transition_split(bank: LossBank) -> dict:
    """rho_T split into the definitional R0 hidden entry transition and other hidden transitions; observable
    transitions in / not in C (pooled over the 166 windows)."""
    from ..scoring import driver as D
    n_T = n_obs = n_hid = n_r0 = n_obs_inC = 0
    per = []
    for iid in bank.index["incident_id"].to_list():
        inc = X.inc_of(iid)
        tr = D.load_transitions("iot23", inc)
        if tr.height == 0:
            per.append({"window": inc, "n_T": 0, "n_r0": 0, "n_obs": 0})
            continue
        hid = tr.filter(pl.col("kind") != "obs")
        r0 = hid.filter(pl.col("src_ep") == R0_SRC)
        obs = tr.filter(pl.col("kind") == "obs")
        n_T += tr.height
        n_hid += hid.height
        n_r0 += r0.height
        n_obs += obs.height
        n_obs_inC += int(obs["in_C"].fill_null(False).sum())
        per.append({"window": inc, "n_T": tr.height, "n_r0": r0.height, "n_obs": obs.height})
    return {"n_T": n_T, "n_T_hid": n_hid, "n_T_hid_R0": n_r0, "n_T_hid_other": n_hid - n_r0, "n_T_obs": n_obs,
            "n_T_obs_in_C": n_obs_inC, "rho_T": n_hid / n_T if n_T else math.nan,
            "rho_T_R0_share": n_r0 / n_T if n_T else math.nan,
            "rho_T_without_R0": (n_hid - n_r0) / (n_T - n_r0) if n_T - n_r0 > 0 else math.nan,
            "ceiling_with_R0": n_obs_inC / n_T if n_T else math.nan,
            "ceiling_without_R0": n_obs_inC / (n_T - n_r0) if n_T - n_r0 > 0 else math.nan,
            "r_CT": n_obs_inC / n_obs if n_obs else math.nan,
            "n_windows_with_R0": int(sum(1 for p in per if p["n_r0"] > 0)),
            "n_windows_with_obs": int(sum(1 for p in per if p["n_obs"] > 0))}


def pwin_frequency(bank: LossBank, method: str, alpha: float, *, K: int = 83, n_test: int = 83,
                   n_splits: int = 200) -> tuple[pl.DataFrame, dict]:
    """P-WIN (window-level random splits): for every (window, rank) the number of splits certifying it among the
    splits in which the window is a test window. Returns per-claim frequencies and totals over splits."""
    sp = random_splits(bank.n, K, range(n_splits), n_test)
    sel = X.select(bank, method, sp, alpha)
    test_rows, tmask, R = sel.test_rows, sel.tmask, sel.R
    n_as_test = np.zeros(bank.n)
    np.add.at(n_as_test, test_rows[tmask], 1)
    maxR = int(R.max()) if R.size else 0
    hits = np.zeros((bank.n, max(maxR, 1)))
    for s in range(sp.S):
        for r, k in zip(test_rows[s][tmask[s]], R[s][tmask[s]]):
            if k > 0:
                hits[r, :k] += 1
    rows = []
    for r in np.flatnonzero(hits.sum(axis=1) > 0):
        sl = bank.slice(int(r))
        for j in np.flatnonzero(hits[r] > 0):
            rows.append({"row": int(r), "window": X.inc_of(bank.index["incident_id"][int(r)]), "rank": int(j) + 1,
                         "y": int(bank.csr["y"][sl][j]), "n_certified": int(hits[r, j]), "n_as_test": int(n_as_test[r]),
                         "freq": float(hits[r, j] / max(n_as_test[r], 1))})
    c = bank.counts(test_rows, np.where(tmask, R, 0))
    tot = {"sum_R_over_splits": float(np.where(tmask, c["R"], 0).sum()),
           "sum_TP_over_splits": float(np.where(tmask, c["TP"], 0).sum()),
           "mean_claims_per_split": float(np.where(tmask, c["R"], 0).sum(axis=1).mean()),
           "mean_true_claims_per_split": float(np.where(tmask, c["TP"], 0).sum(axis=1).mean()),
           "n_distinct_claims": len(rows), "n_distinct_true_claims": int(sum(1 for x in rows if x["y"] == 1))}
    return (pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()), tot
