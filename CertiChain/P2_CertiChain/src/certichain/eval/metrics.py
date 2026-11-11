"""Metrics (DESIGN §6.2), vectorised over splits: inputs are [S, M] arrays over the test incidents
of S splits (with a [S, M] validity mask for padded LOSO splits); outputs are [S] arrays.

Conventions (stated in every table):
* per-incident FDP_strict = V / max(R, 1) (AMB counted false); FDP_lenient = V0 / max(R - A, 1);
* realised FDR = mean over test incidents of FDP (0 for no claims) -- the estimand CRC-BH/FDX/LTT
  guarantee; pooled FDR = sum V / max(sum R, 1) over the split's test incidents -- the estimand
  Lee-Ren and pooled BH-iid guarantee;
* recall-type ratios are reported both as a mean over incidents with a positive denominator
  ("_mean") and pooled over the split ("_pooled");
* analyst load = sum R / sum TP (inf when nothing true is certified but claims exist).
"""
from __future__ import annotations

import numpy as np


def _m(a, mask):
    return np.where(mask, a, 0.0)


def masked_mean(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, _m(a, mask).sum(axis=1) / np.maximum(n, 1), np.nan)


def ratio_pooled(num: np.ndarray, den: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = _m(num, mask).sum(axis=1)
    d = _m(den, mask).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)


def ratio_mean(num: np.ndarray, den: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ok = mask & np.isfinite(den) & (den > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ok, num / np.where(ok, den, 1.0), 0.0)
    return masked_mean(r, ok)


# --------------------------------------------------------------------------- per incident
def fdp_strict(V, R):
    return np.asarray(V, float) / np.maximum(np.asarray(R, float), 1.0)


def fdp_lenient(V, A, R):
    V, A, R = (np.asarray(x, float) for x in (V, A, R))
    return (V - A) / np.maximum(R - A, 1.0)


# --------------------------------------------------------------------------- split level
def realised_fdr(fdp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return masked_mean(fdp, mask)


def pooled_fdr(V, R, mask) -> np.ndarray:
    v = _m(np.asarray(V, float), mask).sum(1)
    r = _m(np.asarray(R, float), mask).sum(1)
    return v / np.maximum(r, 1.0)


def exceedance(fdp: np.ndarray, gamma: float, mask: np.ndarray) -> np.ndarray:
    """P(FDP > gamma) over test incidents (FDX; with gamma = 2 alpha the tail metric)."""
    return masked_mean((fdp > gamma + 1e-12).astype(float), mask)


def per_stratum(values: np.ndarray, strata: np.ndarray, mask: np.ndarray) -> dict:
    """Mean of ``values`` within each stratum value (per split)."""
    out = {}
    for v in np.unique(strata[mask]) if mask.any() else []:
        out[v] = masked_mean(values, mask & (strata == v))
    return out


def wrongful_implication(R, benign, mask) -> np.ndarray:
    """P(any claim | benign-only incident)."""
    return masked_mean((np.asarray(R) > 0).astype(float), mask & benign.astype(bool))


def analyst_load(R, TP, mask) -> np.ndarray:
    r = _m(np.asarray(R, float), mask).sum(1)
    t = _m(np.asarray(TP, float), mask).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, r / np.where(t > 0, t, 1.0), np.where(r > 0, np.inf, np.nan))


def split_metrics(c: dict[str, np.ndarray], meta: dict[str, np.ndarray], mask: np.ndarray, alpha: float,
                  gammas=(0.2, 0.3)) -> dict[str, np.ndarray]:
    """All edge-level metrics for S splits. ``c``: counts R, V, A, TP, TR, BR, DC, FD [S, M];
    ``meta``: n_true, n_trans, n_trans_obs, n_trans_hid, n_elink, n_bridges, n_noncamp_dev,
    benign_only [S, M]."""
    R, V, A, TP = (c[k].astype(float) for k in ("R", "V", "A", "TP"))
    fs = fdp_strict(V, R)
    fl = fdp_lenient(V, A, R)
    out = {
        "fdr": realised_fdr(fs, mask),
        "fdr_lenient": realised_fdr(fl, mask),
        "pooled_fdr": pooled_fdr(V, R, mask),
        "pooled_fdr_lenient": pooled_fdr(V - A, R - A, mask),
        "p_fdp_gt_2a": exceedance(fs, 2 * alpha, mask),
        "n_claims": masked_mean(R, mask),
        "frac_empty": masked_mean((R == 0).astype(float), mask),
        "recall_mean": ratio_mean(TP, meta["n_true"], mask),
        "recall_pooled": ratio_pooled(TP, meta["n_true"], mask),
        "str_mean": ratio_mean(c["TR"], meta["n_trans"], mask),
        "str_pooled": ratio_pooled(c["TR"], meta["n_trans"], mask),
        "ostr_mean": ratio_mean(c["TR"], meta["n_trans_obs"], mask),
        "cer_link_mean": ratio_mean(TP, meta["n_elink"] + meta["n_trans_hid"], mask),
        "cer_link_pooled": ratio_pooled(TP, meta["n_elink"] + meta["n_trans_hid"], mask),
        "bridge_recall": ratio_pooled(c["BR"], meta["n_bridges"], mask),
        "ucr": ratio_pooled(c["DC"], R, mask),
        "analyst_load": analyst_load(R, TP, mask),
        "wrongful_benign": wrongful_implication(R, meta["benign_only"], mask),
        "dev_false_impl": ratio_pooled(c["FD"], meta["n_noncamp_dev"], mask),
    }
    for g in gammas:
        out[f"fdx_{int(round(g * 100)):03d}"] = exceedance(fs, g, mask)
    return out


def chain_metrics(ch: dict[str, np.ndarray], n_gt_paths: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """``ch``: chN, chBad, chBad0, pfN, pfBad, ccr [S, M] at the chosen grid index."""
    chN = ch["chN"].astype(float)
    bad = ch["chBad"].astype(float)
    bad0 = ch["chBad0"].astype(float)
    return {
        "maxchain_fdr": masked_mean(bad / np.maximum(chN, 1.0), mask),
        "maxchain_fdr_lenient": masked_mean(bad0 / np.maximum(chN - (bad - bad0), 1.0), mask),
        "prefix_fdr": masked_mean(ch["pfBad"] / np.maximum(ch["pfN"].astype(float), 1.0), mask),
        "chain_fwer": masked_mean((bad > 0).astype(float), mask),
        "n_chains": masked_mean(chN, mask),
        "ccr_mean": ratio_mean(ch["ccr"].astype(float), n_gt_paths, mask),
        "ccr_pooled": ratio_pooled(ch["ccr"].astype(float), n_gt_paths, mask),
        "maxchain_fdr_pooled": ratio_pooled(bad, chN, mask),
    }


def certification_gap(fdp_table_test: np.ndarray, rec_table_test: np.ndarray, col: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """Recall at the oracle grid index (largest index whose realised TEST FDR <= the method's realised
    FDR) minus the method's recall. Tables [S, M, G] over test incidents; recall entries that are NaN
    (no denominator) are ignored. Descriptive only (uses test labels)."""
    S, _, G = fdp_table_test.shape
    fdr_c = np.stack([masked_mean(fdp_table_test[:, :, g], mask) for g in range(G)], 1)
    rec_c = np.stack([masked_mean(np.nan_to_num(rec_table_test[:, :, g]),
                                  mask & np.isfinite(rec_table_test[:, :, g])) for g in range(G)], 1)
    tgt = fdr_c[np.arange(S), col]
    ok = fdr_c <= tgt[:, None] + 1e-12
    star = G - 1 - np.argmax(ok[:, ::-1], axis=1)
    return rec_c[np.arange(S), star] - rec_c[np.arange(S), col]


def claims_to_reach(bank, rows, frac: float = 0.8, denom: str = "n_trans_obs") -> np.ndarray:
    """Analyst load of a RANKED baseline (§6.2): per incident, the number of top-score candidates to
    inspect until the distinct transition hits reach ``frac`` x ``denom`` (default 80% of |T*_obs|).
    NaN when the incident has no denominator or the target is unreachable within C."""
    rows = np.asarray(rows, dtype=np.int64)
    den = bank.col(denom)[rows]
    out = np.full(rows.size, np.nan)
    off2 = bank.csr["off2"]
    for i, r in enumerate(rows):
        if not np.isfinite(den[i]) or den[i] <= 0:
            continue
        cum = bank.csr["c_trans"][off2[r]: off2[r + 1]]
        k = int(np.searchsorted(cum, frac * den[i] - 1e-9, side="left"))
        if k < cum.size:
            out[i] = k
    return out


def summarize(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, float)
    f = v[np.isfinite(v)]
    if f.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"), "n": 0}
    return {"mean": float(f.mean()), "sd": float(f.std(ddof=1)) if f.size > 1 else 0.0,
            "se": float(f.std(ddof=1) / np.sqrt(f.size)) if f.size > 1 else 0.0, "n": int(f.size)}
