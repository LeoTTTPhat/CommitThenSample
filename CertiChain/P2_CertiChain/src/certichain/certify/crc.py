"""Conformal risk control on per-incident loss tables (DESIGN §3.5 Alg. 4 step 3; Thm 1).

    lambda-hat = max{ lambda in Lambda : (sum_k L~_k(lambda) + B) / (K + 1) <= alpha }

with L~ the monotonised (running max over the grid) per-incident loss in [0, B]. If only
lambda_0 qualifies -- or not even lambda_0, when K + 1 < B / alpha -- the rule abstains and
returns index 0 (S(lambda_0) = {}). The same function serves the FDX indicator loss
1{L~ > gamma}, the composite chain loss (Alg. 5) and the chain-RT 0/1 loss.

``crc_threshold`` takes one calibration table; ``crc_threshold_from_sums`` is the vectorised
form used by eval/runner.py (many splits at once from cal-row sums).
"""
from __future__ import annotations

import json
from typing import Any

import blake3
import numpy as np

TOL = 1e-12


def _check_monotone(L: np.ndarray) -> None:
    if L.shape[-1] > 1 and np.any(np.diff(L, axis=-1) < -1e-12):
        raise ValueError("loss table is not monotone in the grid index; pass the running max "
                         "(CRC Cor. 1) or check_monotone=False for the un-monotonised ablation")


def crc_threshold(L: np.ndarray, alpha: float, B: float = 1.0, check_monotone: bool = True) -> int:
    """Grid index of lambda-hat for the calibration table ``L`` [K, |Lambda|] (0 = abstain)."""
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("L must be [K, G]")
    if check_monotone:
        _check_monotone(L)
    K = L.shape[0]
    return int(crc_threshold_from_sums(L.sum(axis=0)[None, :], np.array([K]), alpha, B)[0])


def crc_threshold_from_sums(sums: np.ndarray, K, alpha: float, B: float = 1.0) -> np.ndarray:
    """Vectorised CRC: ``sums`` [S, G] = sum of calibration losses per split, ``K`` [S] (or int).
    Returns the largest qualifying grid index per split (0 if none beyond lambda_0)."""
    sums = np.atleast_2d(np.asarray(sums, dtype=np.float64))
    K = np.broadcast_to(np.asarray(K, dtype=np.float64), (sums.shape[0],))
    risk = (sums + B) / (K[:, None] + 1.0)
    ok = risk <= alpha + TOL
    G = sums.shape[1]
    last = G - 1 - np.argmax(ok[:, ::-1], axis=1)
    return np.where(ok.any(axis=1), last, 0).astype(np.int64)


def table_hash(L_cal: np.ndarray) -> str:
    """BLAKE3 of the calibration table (dtype, shape and C-order bytes)."""
    a = np.ascontiguousarray(np.asarray(L_cal, dtype=np.float64))
    h = blake3.blake3()
    h.update(json.dumps({"dtype": str(a.dtype), "shape": list(a.shape)}).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def certificate(L_cal: np.ndarray, alpha: float, lam_idx: int, lam_grid=None, *, B: float = 1.0,
                scorer_id: str | None = None, prompt_hash: str | None = None,
                rule: str = "BH-ref", loss: str = "FDP-strict-monotonised") -> dict[str, Any]:
    """Alg. 4 step 4 certificate: (alpha, lambda-hat, K, calibration-table BLAKE3, scorer, prompt)."""
    L_cal = np.asarray(L_cal, dtype=np.float64)
    K = int(L_cal.shape[0])
    lam = None if lam_grid is None else float(np.asarray(lam_grid)[lam_idx])
    return {
        "alpha": float(alpha),
        "lambda_hat_index": int(lam_idx),
        "lambda_hat": lam,
        "abstain": bool(lam_idx == 0),
        "K": K,
        "B": float(B),
        "calibration_risk_at_lambda_hat": float((L_cal[:, lam_idx].sum() + B) / (K + 1)),
        "calibration_table_blake3": table_hash(L_cal),
        "rule": rule,
        "loss": loss,
        "scorer_id": scorer_id,
        "prompt_hash": prompt_hash,
    }


def crc_certify(L_cal: np.ndarray, alpha: float, lam_grid=None, B: float = 1.0, **kw) -> tuple[int, dict]:
    """crc_threshold + certificate in one call."""
    idx = crc_threshold(L_cal, alpha, B)
    return idx, certificate(L_cal, alpha, idx, lam_grid, B=B, **kw)
