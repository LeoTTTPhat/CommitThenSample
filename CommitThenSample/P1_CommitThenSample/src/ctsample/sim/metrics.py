"""Statistical reporting helpers (DESIGN §6.6).

* Wilson 95% intervals for coverage / miscoverage rates (+-1.4 pp at R = 1,000).
* Coverage tables from the MC ``cov`` frame (one row per rep x estimand x variant).
* Family-wise coverage of a Q-statement report (any miss across the report's estimands).
* Geometric means; cluster bootstrap over scenarios (2,000 resamples, estimands nested).
* Width summaries: median and IQR across reps.
"""
from __future__ import annotations

import numpy as np

__all__ = ["wilson", "coverage_table", "familywise", "geometric_mean", "cluster_bootstrap", "median_iqr",
           "width_summary", "MISS_FLAGS"]

MISS_FLAGS = ("lcs_tu", "lcs_ft", "ucs_tu", "ucs_ft", "wald_lo_tu", "wald_lo_ft", "wald_hi_tu", "wald_hi_ft")


def wilson(k, n, z: float = 1.959963984540054):
    """(p, lo, hi) Wilson score interval for k successes out of n."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def coverage_table(cov, flags=MISS_FLAGS, by=("estimand", "variant")):
    """Miscoverage rates with Wilson intervals per group (pandas DataFrame in, DataFrame out).

    UCS flags are counted as P(issued and wrong): a withheld UCS is a non-miss (statement policy)."""
    import pandas as pd
    rows = []
    for key, g in cov.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        n = len(g)
        row["n"] = n
        for fl in flags:
            if fl not in g:
                continue
            k = int(np.asarray(g[fl], dtype=bool).sum())
            p, lo, hi = wilson(k, n)
            row[fl] = float(p)
            row[fl + "_lo"] = float(lo)
            row[fl + "_hi"] = float(hi)
        if "ucs_issued" in g:
            row["ucs_issued_rate"] = float(np.asarray(g["ucs_issued"], dtype=bool).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def familywise(cov, flag: str = "lcs_tu", by=("variant",), estimands=None):
    """Family-wise miscoverage: per rep, any miss across the report's estimands (Bonferroni target alpha*Q)."""
    import pandas as pd
    c = cov if estimands is None else cov[cov["estimand"].isin(list(estimands))]
    rows = []
    for key, g in c.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        anym = g.groupby("rep")[flag].any()
        p, lo, hi = wilson(int(anym.sum()), len(anym))
        rows.append(dict(zip(by, key)) | {"flag": flag, "n_reps": len(anym), "n_estimands": g["estimand"].nunique(),
                                          "fw_miss": float(p), "fw_lo": float(lo), "fw_hi": float(hi)})
    return pd.DataFrame(rows)


def geometric_mean(x, axis=None):
    x = np.asarray(x, dtype=float)
    return np.exp(np.mean(np.log(x), axis=axis))


def cluster_bootstrap(values_by_scenario, B: int = 2000, seed: int = 0, nested: bool = True, per_scenario: bool = False,
                      level: float = 0.95):
    """Cluster bootstrap of the geometric mean of positive ratios.

    values_by_scenario: dict scenario -> 1-D array of per-estimand ratios (> 0). Scenarios are resampled
    with replacement; with nested=True the estimands within each resampled scenario are resampled too.
    Statistic: pooled geometric mean over all (scenario, estimand) values, or (per_scenario=True) the
    geometric mean of per-scenario geometric means. Returns dict(point, lo, hi, B)."""
    rng = np.random.default_rng(seed)
    logs = [np.log(np.asarray(v, dtype=float)) for v in values_by_scenario.values()]
    logs = [v[np.isfinite(v)] for v in logs]
    logs = [v for v in logs if len(v)]
    S = len(logs)
    if S == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "B": B}

    def stat(groups):
        if per_scenario:
            return float(np.exp(np.mean([g.mean() for g in groups])))
        allv = np.concatenate(groups)
        return float(np.exp(allv.mean()))

    point = stat(logs)
    sums = np.array([v.sum() for v in logs])
    cnts = np.array([len(v) for v in logs])
    means = sums / cnts
    boots = np.empty(B)
    for b in range(B):
        sc = rng.integers(0, S, S)
        if nested:
            ss, cc, mm = 0.0, 0, []
            for s in sc:
                v = logs[s]
                pick = v[rng.integers(0, len(v), len(v))]
                if per_scenario:
                    mm.append(pick.mean())
                else:
                    ss += pick.sum()
                    cc += len(pick)
            boots[b] = np.exp(np.mean(mm)) if per_scenario else np.exp(ss / cc)
        else:
            boots[b] = np.exp(means[sc].mean()) if per_scenario else np.exp(sums[sc].sum() / cnts[sc].sum())
    a = (1 - level) / 2
    return {"point": point, "lo": float(np.quantile(boots, a)), "hi": float(np.quantile(boots, 1 - a)), "B": B}


def median_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "n": 0}
    q = np.quantile(x, [0.25, 0.5, 0.75])
    return {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]), "n": int(len(x))}


def width_summary(width, by=("estimand", "variant"), at: str = "last"):
    """Median/IQR of relative LCB gap (Q - L+)/Q, statistical gain (L* - S)/Q, share L* > S, UCB slack
    U+ - Q, and Wald relative half-width, at the last report epoch per (rep, estimand, variant)."""
    import pandas as pd
    w = width
    if at == "last":
        idx = w.groupby(["rep", "estimand", "variant"])["epoch"].idxmax()
        w = w.loc[idx]
    rows = []
    for key, g in w.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        Q = g["Q"].to_numpy(float)
        ok = Q > 0
        gap = (Q - g["Lplus"].to_numpy(float))[ok] / Q[ok]
        gain = (g["Lstar"].to_numpy(float) - g["S"].to_numpy(float))[ok] / Q[ok]
        slack = (g["Uplus"].to_numpy(float) - Q)
        whalf = ((g["wald_hi"] - g["wald_lo"]).to_numpy(float) / 2)[ok] / Q[ok]
        row = dict(zip(by, key)) | {"n": len(g)}
        for nm, arr in (("rel_lcb_gap", gap), ("stat_gain", gain), ("ucb_slack", slack), ("wald_rel_halfwidth", whalf)):
            mi = median_iqr(arr)
            row[nm + "_median"] = mi["median"]
            row[nm + "_q25"] = mi["q25"]
            row[nm + "_q75"] = mi["q75"]
        row["share_Lstar_gt_S"] = float(np.mean(g["Lstar"].to_numpy(float) > g["S"].to_numpy(float)))
        row["ucs_issued_rate"] = float(np.mean(g["ucs_issued"].to_numpy(bool)))
        rows.append(row)
    return pd.DataFrame(rows)
