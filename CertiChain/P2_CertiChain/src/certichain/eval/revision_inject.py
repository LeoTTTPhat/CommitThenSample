"""P2-X revision (coordinator addition C; review R2-M3 / M4): STRONG test-only injection. Owner: P2-X.
POST-PREREG, EXPLORATORY.

E4's test-only injection (llm/corrupt.inject_top) raised q nulls only to just above the incident's own top score
(max + 1e-3), so in incidents whose top score is below the calibrated threshold nothing changes. Here q in {1, 3, 10}
null (Y = 0) candidates per test incident are raised ABOVE THE MAXIMUM SCORE OF THE WHOLE CALIBRATION POOL (and of the
scorer's null pool S0): top = max(max S0, max over all pool scores) + 1, so the injected nulls receive the smallest
reference p-value 1/(|S0| + 1) and pass every finite score threshold. The nulls chosen are exactly E4's (the same
keyed-uniform draw of corrupt.inject_top, seed 0, keys (incident, cand_id)), so weak and strong injection differ only
in the injected score. Banks: results/exp/X2/banks/<ds>/<sid>__X2-inject-above-<q>/bank (TEST ONLY: calibration rows
come from the clean bank, test rows from the injected one; row-aligned).
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from ..io.paths import atomic_write_json
from . import revision_relabel as RL
from . import revision_tasks as RT


def inject_above(s: np.ndarray, y: np.ndarray, q: int, top_value: float, *, seed: int = 0,
                 keys: Sequence[Any] | None = None, margin: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """One incident: force the same q nulls as corrupt.inject_top to ``top_value`` (+ a tiny decreasing offset to
    keep their order). Returns (new scores, injected mask)."""
    from ..llm import corrupt
    s = np.asarray(s, np.float64).copy()
    y = np.asarray(y)
    u = corrupt._uniform(len(s), seed, keys, "inj")
    nulls = np.flatnonzero(y == 0)
    pick = nulls[np.argsort(u[nulls], kind="stable")][: int(q)]
    s[pick] = float(top_value) + margin * np.arange(len(pick))[::-1] / max(1, len(pick))
    inj = np.zeros(len(s), bool)
    inj[pick] = True
    return s, inj


def bank_tag(q: int) -> str:
    return f"X2-inject-above-{q}"


def build_inject_banks(ds: str, sid: str, qs=(1, 3, 10), *, n_jobs: int = 8) -> dict:
    from ..certify.losses import build_loss_tables
    from ..scoring import driver as D
    t0 = time.perf_counter()
    incs = D.list_incidents(ds, "caltest")
    sc = {inc: RL.read_scores(ds, sid, inc) for inc in incs}
    s0 = RL.null_pool(sid)
    top = float(max(np.max(s0), max(float(np.max(v)) for v in sc.values() if v.size))) + 1.0
    ys = {inc: D.load_labels(ds, inc)["y"].to_numpy() for inc in incs}
    cids = {inc: D.load_cands(ds, inc)["cand_id"].to_list() for inc in incs}
    out = {"ds": ds, "sid": sid, "top_value": top, "s0_max": float(np.max(s0)), "n_s0": int(s0.size), "banks": {}}
    for q in qs:
        I, n_inj = [], 0
        for inc in incs:
            s2, inj = inject_above(sc[inc], ys[inc], q, top, seed=0, keys=[(inc, int(c)) for c in cids[inc]])
            n_inj += int(inj.sum())
            I.append(D.build_incident(ds, inc, s2))
        bp = RL.bank_path(ds, sid, bank_tag(q))
        bank = build_loss_tables(I, s0, bp, n_jobs=n_jobs, overwrite=True,
                                 meta={"dataset": ds, "scorer_id": f"{sid}__{bank_tag(q)}", "pool": "caltest",
                                       "injection": f"q={q} nulls per incident at {top:.4f} (> every calibration-pool "
                                                    "and S0 score); TEST ONLY"})
        out["banks"][str(q)] = {"path": str(bp), "n_injected": n_inj, "input_hash": bank.meta.get("input_hash")}
    out["seconds"] = time.perf_counter() - t0
    atomic_write_json(RT.x2_root() / "banks" / ds / f"inject_above_{sid}.json", out)
    return out
