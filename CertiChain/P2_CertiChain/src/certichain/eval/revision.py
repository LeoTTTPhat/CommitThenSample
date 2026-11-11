"""P2-X revision analyses (results/exp/X2, results/summary/X2_*.json). Owner: P2-X (Mon 19 Oct 2026).

EVERYTHING HERE IS POST-PREREG AND EXPLORATORY (documented post-PREREG analyses requested by the mock review,
paper/review_mock.md). Nothing in this module changes a pre-registered number: it reads the frozen loss banks
(results/losses), the frozen fixed-claim baselines (results/baselines), derived interim files (generator logs,
transitions, units) and writes new outputs only.

Contents
--------
* ``conditional_metrics``   the operational error rates of a selection per split: abstention (share of test
                             incidents with no claim), the guaranteed mean per-incident FDP, the CONDITIONAL FDP given
                             >= 1 claim, pooled FDR, claims / false claims per incident, FDX exceedance (also given
                             >= 1 claim), P(>= 1 false claim) -- next to every standard metric of eval/metrics.
* ``x2_eval`` / ``x2_bootstrap``   the same on P2-G's selections (``experiments.select``) or a fixed-claim baseline,
                             with the incident-pool cluster bootstrap on P2-C's RNG stream, or an episode-cluster
                             ("two-way", multiple-membership) resampler (``episode_pool_sampler``).
* ``loso_eval`` / ``loso_bootstrap``   IoT-23 P-LOSO with every rule family (incl. LTT and fixed baselines) at the
                             window level, and a scenario-cluster bootstrap with LOSO recomputation (bounded percentile
                             CIs; each held-out copy is calibrated on the copies of the OTHER scenarios only).
* ``mondrian_select``       Mondrian CRC: one CRC threshold per label-free stratum (e.g. N tercile).
* ``chain_exceedance``      per-incident maximal-chain FDP exceedance of the chain rules.
* ``two_sample_auc_cv`` / ``permutation_auc`` / ``group_boot_auc``   calibrated versions of the E10 two-sample AUC.
* ``mde``                    minimum detectable effect of a one-sided z-type test.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
from scipy.stats import norm
from scipy.stats import t as _tdist

from ..baselines.bh_iid import NullIndex, bh_iid_R
from ..baselines.by_ref import by_ref_R
from ..baselines.naive import bh_ref_plain_R
from ..certify.crc import crc_threshold_from_sums
from ..certify.losses import LossBank
from ..certify.ltt import ltt_threshold_from_sums
from ..scoring.calibrate import b7_R, b8_R
from . import experiments as X
from . import runner as RN
from .metrics import chain_metrics, masked_mean, split_metrics
from .splits import Splits, random_splits

STATUS = "exploratory"
STATUS_NOTE = "post-PREREG revision analysis (P2-X, X2); not pre-registered"
X2_STREAM = 0x2A2A0B             # episode-cluster resampler stream (incident stage reuses P2-C's stream offset)
FIXED = ("B1", "B2", "B4", "B5", "oracle")

# metrics carried through X2 bootstraps (means over the splits of a replicate; ratio metrics as noted)
X2_METRICS = (
    "frac_empty", "fdr", "cond_fdp", "cond_fdp_ratio", "pooled_fdr", "fdr_lenient", "cond_fdp_lenient",
    "pooled_fdr_lenient", "n_claims", "n_false", "n_true_claims", "n_claims_cond", "n_false_cond",
    "fdx_020", "fdx_030", "p_fdp_gt_2a", "cond_fdx_020", "cond_fdx_030", "cond_p_fdp_gt_2a", "p_any_false",
    "cond_p_any_false", "str_mean", "str_pooled", "ostr_mean", "cer_link_mean", "cer_link_pooled", "recall_pooled",
    "wrongful_benign", "dev_false_impl", "ucr", "abstain_split",
)
CHAIN_X2 = ("maxchain_fdr", "maxchain_fdr_pooled", "maxchain_fdr_lenient", "cond_maxchain_fdp", "chain_exceed_a",
            "cond_chain_exceed_a", "chain_fwer", "n_chains", "frac_no_chain", "ccr_mean", "ccr_pooled", "prefix_fdr")

DEFS = {
    "frac_empty": "abstention rate = share of test incidents with no certified claim",
    "abstain_split": "share of splits whose lambda-hat is lambda_0 (no claims in any test incident)",
    "fdr": "mean per-incident strict FDP over test incidents, 0 for no claims (the estimand CRC-BH guarantees)",
    "cond_fdp": "CONDITIONAL FDP: mean strict FDP over test incidents with >= 1 claim (per split; mean over splits "
                "with >= 1 claimed incident)",
    "cond_fdp_ratio": "conditional FDP as a ratio over all splits: sum of FDPs / number of claimed test incidents "
                      "(= mean FDR / (1 - mean abstention) when the test size is constant)",
    "pooled_fdr": "pooled FDR = sum V / max(sum R, 1) over a split's test incidents (share of issued claims that are "
                  "false), mean over splits",
    "fdr_lenient": "mean per-incident lenient FDP (AMB excluded)",
    "cond_fdp_lenient": "conditional lenient FDP given >= 1 claim",
    "pooled_fdr_lenient": "pooled lenient FDR (AMB excluded)",
    "n_claims": "mean claims per test incident", "n_false": "mean false (Y != 1) claims per test incident",
    "n_true_claims": "mean true claims per test incident",
    "n_claims_cond": "mean claims per claimed test incident", "n_false_cond": "mean false claims per claimed incident",
    "fdx_020": "FDX exceedance P(FDP > 0.2) over test incidents", "fdx_030": "P(FDP > 0.3)",
    "p_fdp_gt_2a": "P(FDP > 2 alpha)", "cond_fdx_020": "P(FDP > 0.2 | >= 1 claim)",
    "cond_fdx_030": "P(FDP > 0.3 | >= 1 claim)", "cond_p_fdp_gt_2a": "P(FDP > 2 alpha | >= 1 claim)",
    "p_any_false": "P(>= 1 false claim) per test incident", "cond_p_any_false": "P(>= 1 false claim | >= 1 claim)",
    "str_mean": "STR = certified distinct transitions / |T*|, mean over incidents with |T*| > 0",
    "str_pooled": "STR pooled over a split", "ostr_mean": "observable STR (/|T*_obs|), mean over incidents",
    "cer_link_mean": "CER_link, mean over incidents", "cer_link_pooled": "CER_link pooled",
    "recall_pooled": "certified true links / true links in C (pooled)",
    "wrongful_benign": "P(any claim | benign-only incident)",
    "dev_false_impl": "non-campaign devices endpoint of >= 1 false claim / non-campaign devices (pooled)",
    "ucr": "share of certified links with a decoy endpoint (pooled)",
    "maxchain_fdr": "maximal-chain FDR (strict), mean over test incidents",
    "maxchain_fdr_pooled": "maximal-chain FDR pooled (bad chains / chains)",
    "maxchain_fdr_lenient": "maximal-chain FDR (lenient)",
    "cond_maxchain_fdp": "mean maximal-chain FDP over test incidents with >= 1 reported chain",
    "chain_exceed_a": "share of test incidents whose maximal-chain FDP exceeds alpha",
    "cond_chain_exceed_a": "share of incidents with >= 1 chain whose maximal-chain FDP exceeds alpha",
    "chain_fwer": "P(some reported maximal chain is wrong)", "n_chains": "maximal chains per test incident",
    "frac_no_chain": "share of test incidents with no reported chain", "ccr_mean": "complete-chain recall, mean",
    "ccr_pooled": "complete-chain recall, pooled", "prefix_fdr": "prefix-family FDR",
}


# =========================================================================== conditional metrics
def conditional_metrics(counts: dict[str, np.ndarray], meta: dict[str, np.ndarray], tmask: np.ndarray, alpha: float,
                        gammas=(0.2, 0.3)) -> dict[str, np.ndarray]:
    """Standard split metrics (eval/metrics.split_metrics) + the conditional / count metrics of the revision.

    ``counts``: R, V, A, TP, TR, BR, DC, FD [S, M] at the chosen selection; ``meta`` as in split_metrics; ``tmask``
    [S, M] the valid test incidents. Every output is an [S] array (NaN when undefined, e.g. no claimed incident)."""
    out = split_metrics(counts, meta, tmask, alpha, gammas)
    R = np.where(tmask, counts["R"], 0).astype(float)
    V = np.where(tmask, counts["V"], 0).astype(float)
    A = np.where(tmask, counts["A"], 0).astype(float)
    TP = np.where(tmask, counts["TP"], 0).astype(float)
    fs = V / np.maximum(R, 1.0)
    fl = (V - A) / np.maximum(R - A, 1.0)
    claimed = tmask & (R > 0)
    out["n_test"] = tmask.sum(axis=1).astype(float)
    out["n_claimed"] = claimed.sum(axis=1).astype(float)
    out["sum_fdp"] = np.where(tmask, fs, 0.0).sum(axis=1)
    out["cond_fdp"] = masked_mean(fs, claimed)
    out["cond_fdp_lenient"] = masked_mean(fl, claimed)
    out["n_false"] = masked_mean(V, tmask)
    out["n_true_claims"] = masked_mean(TP, tmask)
    out["n_claims_cond"] = masked_mean(R, claimed)
    out["n_false_cond"] = masked_mean(V, claimed)
    out["p_any_false"] = masked_mean((V > 0).astype(float), tmask)
    out["cond_p_any_false"] = masked_mean((V > 0).astype(float), claimed)
    for g in tuple(gammas):
        out[f"cond_fdx_{int(round(g * 100)):03d}"] = masked_mean((fs > g + 1e-12).astype(float), claimed)
    out["cond_p_fdp_gt_2a"] = masked_mean((fs > 2 * alpha + 1e-12).astype(float), claimed)
    out["sum_R"] = R.sum(axis=1)
    out["sum_V"] = V.sum(axis=1)
    out["sum_TP"] = TP.sum(axis=1)
    out["sum_TR"] = np.where(tmask, counts["TR"], 0).astype(float).sum(axis=1)
    return out


def chain_exceedance(ch: dict[str, np.ndarray], tmask: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Per-incident maximal-chain FDP exceedance of a chain selection. ``ch``: chN, chBad [S, M] at the chosen
    grid index. Strict chain FDP = chBad / max(chN, 1) (0 for no chain)."""
    chN = np.where(tmask, ch["chN"], 0).astype(float)
    bad = np.where(tmask, ch["chBad"], 0).astype(float)
    cfdp = bad / np.maximum(chN, 1.0)
    has = tmask & (chN > 0)
    return {"cond_maxchain_fdp": masked_mean(cfdp, has),
            "chain_exceed_a": masked_mean((cfdp > alpha + 1e-12).astype(float), tmask),
            "cond_chain_exceed_a": masked_mean((cfdp > alpha + 1e-12).astype(float), has),
            "frac_no_chain": masked_mean((chN == 0).astype(float), tmask)}


def _meta(bank: LossBank, test_rows: np.ndarray) -> dict[str, np.ndarray]:
    meta = {k: bank.col(k)[test_rows] for k in RN.META_COLS}
    meta["benign_only"] = bank.col("benign_only")[test_rows]
    return meta


def fixed_selection_counts(bank: LossBank, fixed: dict[str, np.ndarray], test_rows: np.ndarray,
                           tmask: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Counts [S, M] of a split-independent claim set (experiments.baseline_counts / oracle_counts) on the test rows.
    Incidents without baseline output are masked out."""
    ok = np.isfinite(fixed["R"])[test_rows]
    tm = tmask & ok
    c = {k: np.where(tm, np.nan_to_num(fixed[k][test_rows]), 0).astype(np.int64)
         for k in ("R", "V", "A", "TP", "TR", "BR", "DC")}
    fd = fixed.get("FD")
    c["FD"] = (np.where(tm, np.nan_to_num(fd[test_rows]), 0).astype(np.int64) if fd is not None
               else np.zeros(test_rows.shape, np.int64))
    return c, tm


def x2_eval(bank: LossBank, method: str, splits: Splits, alpha: float, *, pool_rows=None, aux=None,
            fixed: dict[str, np.ndarray] | None = None, cache: dict | None = None, test_bank: LossBank | None = None,
            mask_col: np.ndarray | None = None, chains: bool | None = None,
            selection: "X.Selection | None" = None) -> dict[str, np.ndarray]:
    """Per-split conditional metrics [S] of ``method`` (any experiments.select method, or a fixed-claim baseline when
    ``fixed`` is given). ``mask_col`` [bank.n] restricts the metrics to test incidents of a stratum (calibration
    unchanged). ``selection``: a precomputed Selection (e.g. Mondrian)."""
    pool_rows = np.arange(bank.n) if pool_rows is None else np.asarray(pool_rows, dtype=np.int64)
    if fixed is not None:
        test_rows, tmask = X._rows(pool_rows, splits.test)
        if mask_col is not None:
            tmask = tmask & mask_col[test_rows]
        counts, tm = fixed_selection_counts(bank, fixed, test_rows, tmask)
        meta = _meta(bank, test_rows)
        if fixed.get("FD") is None:
            meta["n_noncamp_dev"] = np.full(test_rows.shape, np.nan)
        out = conditional_metrics(counts, meta, tm, alpha)
        if fixed.get("FD") is None:
            out["dev_false_impl"] = np.full(splits.S, np.nan)
        out["abstain_split"] = np.full(splits.S, np.nan)
        return out
    sel = selection if selection is not None else X.select(bank, method, splits, alpha, pool_rows=pool_rows,
                                                           aux=aux, cache=cache, test_bank=test_bank)
    cb = bank if test_bank is None else test_bank
    tmask = sel.tmask if mask_col is None else (sel.tmask & mask_col[sel.test_rows])
    R = np.where(tmask, sel.R, 0)
    counts = cb.counts(sel.test_rows, R)
    out = conditional_metrics(counts, _meta(cb, sel.test_rows), tmask, alpha)
    out["abstain_split"] = (sel.col == 0).astype(float) if sel.col is not None else np.full(splits.S, np.nan)
    if sel.col is not None:
        out["lam_idx"] = sel.col.astype(float)
    want_chain = (chains if chains is not None else (method.startswith("chain") or method in ("crc_bh", "crc_st")))
    if want_chain and cb.has_chains and sel.family is not None and sel.col is not None:
        from ..certify.losses import CHAIN_KEYS
        ch = {k: cb.t(k, sel.family)[sel.test_rows, sel.col[:, None]] for k in CHAIN_KEYS}
        out.update(chain_metrics(ch, cb.col("n_gt_paths")[sel.test_rows], tmask))
        out.update(chain_exceedance(ch, tmask, alpha))
    return out


def summarize_splits(res: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Mean / sd / split quantiles over splits of every [S] metric; ratio-type extras (cond_fdp_ratio, analyst load
    as a ratio of sums)."""
    out: dict[str, dict[str, float]] = {}
    S = None
    for k, v in res.items():
        v = np.asarray(v, float)
        if v.ndim != 1:
            continue
        S = v.size if S is None else S
        f = v[np.isfinite(v)]
        if f.size == 0:
            out[k] = {"mean": None, "sd": None, "q05": None, "q95": None, "n": 0}
            continue
        out[k] = {"mean": float(f.mean()), "sd": float(f.std(ddof=1)) if f.size > 1 else 0.0,
                  "q05": float(np.quantile(f, 0.05)), "q95": float(np.quantile(f, 0.95)), "n": int(f.size)}
    out.update(ratio_extras(res))
    return out


def ratio_extras(res: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out = {}
    if "sum_fdp" in res and "n_claimed" in res:
        nc = float(np.nansum(res["n_claimed"]))
        out["cond_fdp_ratio"] = {"mean": float(np.nansum(res["sum_fdp"]) / nc) if nc > 0 else None}
    if "sum_R" in res and "sum_TP" in res:
        r, t = float(np.nansum(res["sum_R"])), float(np.nansum(res["sum_TP"]))
        out["load_ratio"] = {"mean": (r / t) if t > 0 else (math.inf if r > 0 else None)}
    return out


def replicate_vector(res: dict[str, np.ndarray], metrics: Sequence[str]) -> np.ndarray:
    """One bootstrap replicate: mean over the replicate's splits of each metric (ratio metrics as ratios)."""
    ex = ratio_extras(res)
    v = np.full(len(metrics), np.nan)
    for i, m in enumerate(metrics):
        if m in ex:
            x = ex[m]["mean"]
            v[i] = np.nan if x is None else float(x)
        elif m in res:
            a = np.asarray(res[m], float)
            a = a[np.isfinite(a)]
            v[i] = float(a.mean()) if a.size else np.nan
    return v


# =========================================================================== resamplers
def incident_pool_sampler(n: int, B: int, seed: int = 0) -> Iterator[np.ndarray]:
    """P2-C / P2-G's incident-pool bootstrap stream (identical resampled pools)."""
    yield from X.boot_pools(n, B, seed)


def episode_pool_sampler(clusters: Sequence[Sequence[int]], n_clusters: int, B: int, seed: int = 0,
                         ) -> Iterator[np.ndarray]:
    """Two-way (incident x source-episode) multiple-membership bootstrap.

    Per replicate: (1) resample the n_clusters source episodes with replacement (multinomial counts c_e, ~Poisson(1));
    (2) incident weight w_i = mean_{e in E_i} c_e (incidents without any source episode keep w_i = 1);
    (3) resample n incidents with replacement with probabilities proportional to w_i. With one episode per incident
    this is the product-weight (pigeonhole) bootstrap of a crossed design (Owen 2007); with all c_e = 1 it is the
    incident-pool bootstrap. Conservative for the variance (the residual term is counted twice)."""
    n = len(clusters)
    rng = np.random.default_rng((int(seed), X2_STREAM))
    sizes = np.array([len(c) for c in clusters], dtype=np.int64)
    flat = np.concatenate([np.asarray(c, dtype=np.int64) for c in clusters]) if sizes.sum() else np.zeros(0, np.int64)
    owner = np.repeat(np.arange(n), sizes)
    made = 0
    while made < B:
        cnt = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters)).astype(float) if n_clusters else None
        w = np.ones(n)
        if flat.size:
            s = np.bincount(owner, weights=cnt[flat], minlength=n)
            has = sizes > 0
            w[has] = s[has] / sizes[has]
        if w.sum() <= 0:
            continue
        rows = rng.choice(n, size=n, replace=True, p=w / w.sum())
        made += 1
        yield rows.astype(np.int64)


def x2_bootstrap(bank: LossBank, method: str, alpha: float, *, K: int, n_test: int | None, B: int = 1000,
                 n_splits: int = 50, seed: int = 0, aux=None, fixed=None, metrics: Sequence[str] = X2_METRICS,
                 sampler: Iterable[np.ndarray] | None = None, mask_col: np.ndarray | None = None,
                 selector: Callable | None = None, progress: Callable[[int], None] | None = None,
                 test_bank: LossBank | None = None) -> np.ndarray:
    """Replicates [B, len(metrics)] of the conditional metrics: per replicate a resampled pool (default: P2-C's
    incident-pool stream) and ``n_splits`` random splits of it. ``selector(bank, splits, pool_rows, cache)`` returns a
    Selection (e.g. Mondrian) instead of experiments.select."""
    seeds = np.arange(n_splits)
    pools = sampler if sampler is not None else incident_pool_sampler(bank.n, B, seed)
    reps = np.full((B, len(metrics)), np.nan)
    cache: dict = {}
    for b, rows in enumerate(pools):
        if b >= B:
            break
        sp = random_splits(rows.size, K, seeds, n_test)
        sel = selector(bank, sp, rows, cache) if selector is not None else None
        res = x2_eval(bank, method, sp, alpha, pool_rows=rows, aux=aux, fixed=fixed, cache=cache,
                      mask_col=mask_col, selection=sel, test_bank=test_bank)
        reps[b] = replicate_vector(res, metrics)
        if progress is not None and (b + 1) % 100 == 0:
            progress(b + 1)
    return reps


def ci_dict(reps: np.ndarray, point: float | None = None, level: float = 0.95) -> dict[str, Any]:
    """Percentile CI + SE of replicates; one-sided 95% upper bounds (normal and percentile)."""
    f = np.asarray(reps, float)
    f = f[np.isfinite(f)]
    if f.size < 2:
        return {"se": None, "lo": None, "hi": None, "ucb95_normal": None, "ucb95_pct": None, "n_rep": int(f.size)}
    a = (1 - level) / 2
    se = float(f.std(ddof=1))
    d = {"se": se, "lo": float(np.quantile(f, a)), "hi": float(np.quantile(f, 1 - a)),
         "ucb95_pct": float(np.quantile(f, 0.95)), "n_rep": int(f.size)}
    d["ucb95_normal"] = (float(point) + float(norm.ppf(0.95)) * se) if point is not None and np.isfinite(point) else None
    return d


# =========================================================================== Mondrian CRC
def label_free_strata(bank: LossBank, kind: str = "N_tercile", pool_rows=None) -> np.ndarray:
    """Stratum id per bank row from label-free incident descriptors. N_tercile: cut points = pool terciles of N
    (the same strata as Table S7 / runner._strata_arrays); a fixed, label-free function of the pool."""
    pool_rows = np.arange(bank.n) if pool_rows is None else np.asarray(pool_rows, dtype=np.int64)
    if kind == "N_tercile":
        N = bank.col("N")
        q = np.quantile(N[pool_rows], [1 / 3, 2 / 3])
        return np.searchsorted(q, N, side="right").astype(np.int64)
    raise ValueError(kind)


def mondrian_select(bank: LossBank, method: str, splits: Splits, alpha: float, strata: np.ndarray, *,
                    pool_rows=None, delta: float = 0.1) -> "X.Selection":
    """Mondrian (stratum-conditional) calibration: per split and stratum k, lambda-hat_k = CRC (or LTT / FDX) on the
    calibration incidents of stratum k only, applied to the test incidents of stratum k. Valid conditional on the
    stratum when the stratum map is a fixed (label-free) function of the incident (Cor. S1)."""
    if method not in RN.COLUMN_METHODS or RN.COLUMN_METHODS[method].get("chain"):
        raise ValueError("Mondrian needs an edge loss-table method")
    spec = RN.COLUMN_METHODS[method]
    fam = spec["family"]
    pool_rows = np.arange(bank.n) if pool_rows is None else np.asarray(pool_rows, dtype=np.int64)
    cal_rows, cal_mask = X._rows(pool_rows, splits.cal)
    test_rows, tmask = X._rows(pool_rows, splits.test)
    Rtab = bank.t("R", fam)
    R = np.zeros(test_rows.shape, dtype=np.int64)
    colmat = np.zeros((splits.S, int(strata.max()) + 1), dtype=np.int64)
    for k in np.unique(strata):
        cm = cal_mask & (strata[cal_rows] == k)
        col = RN.choose_columns(bank, spec, cal_rows, cm, alpha, delta)
        colmat[:, int(k)] = col
        tm = tmask & (strata[test_rows] == k)
        R = np.where(tm, Rtab[test_rows, col[:, None]].astype(np.int64), R)
    sel = X.Selection(cal_rows, cal_mask, test_rows, tmask, R, None, fam)
    sel.stratum_cols = colmat           # type: ignore[attr-defined]
    return sel


# =========================================================================== P-LOSO (all rule families)
def _scenario_tables(bank: LossBank, groups: np.ndarray, scen: Sequence[str], table: str, fam: str) -> np.ndarray:
    T = bank.t(table, fam)
    return np.stack([T[np.flatnonzero(groups == s)].mean(axis=0) for s in scen])


def loso_window_R(bank: LossBank, method: str, alpha: float, *, groups: np.ndarray, copies: Sequence[str] | None = None,
                  delta: float = 0.1, aux=None, cache: dict | None = None) -> tuple[list[np.ndarray], list[np.ndarray], list]:
    """P-LOSO selection sizes. ``copies``: the (possibly resampled, with duplicates) list of scenario copies; the
    held-out copy j is calibrated on all copies whose scenario differs from copies[j] (K_j = their number).
    Column methods use the scenario loss = mean window loss (DESIGN §5.1.1); LTT uses the same scenario means of
    the raw FDP; bh_iid / bh_iid_pooled / b8 calibrate on the windows of the calibration copies; bh_ref / by_ref / b7
    need no calibration. Returns per copy: (window rows, R per window, lambda-hat index or None)."""
    cache = {} if cache is None else cache
    g = np.asarray(groups)
    scen_all = sorted(set(g.tolist()))
    copies = list(scen_all) if copies is None else list(copies)
    rows_of = {s: np.flatnonzero(g == s) for s in scen_all}
    out_rows, out_R, out_col = [], [], []
    if method in RN.COLUMN_METHODS:
        spec = RN.COLUMN_METHODS[method]
        fam, tab, calib = spec["family"], spec["table"], spec["calib"]
        key = ("loso_L", method)
        if key not in cache:
            cache[key] = dict(zip(scen_all, _scenario_tables(bank, g, scen_all, tab, fam)))
        Ls = cache[key]
        Rtab = bank.t("R", fam)
        for j, s in enumerate(copies):
            others = [c for c in copies if c != s]
            if not others:
                col = 0
            else:
                sums = np.sum([Ls[c] for c in others], axis=0)[None, :]
                Kc = np.array([len(others)])
                if calib == "crc":
                    col = int(crc_threshold_from_sums(sums, Kc, alpha)[0])
                elif calib == "ltt":
                    col = int(ltt_threshold_from_sums(sums, Kc, alpha, delta)[0])
                else:
                    raise ValueError(f"{method}: calibration {calib!r} not supported under P-LOSO")
            r = rows_of[s]
            out_rows.append(r)
            out_R.append(Rtab[r, col].astype(np.int64))
            out_col.append(col)
        return out_rows, out_R, out_col
    if method in ("bh_ref", "by_ref", "b7"):
        k2 = (method, alpha)
        if k2 not in cache:
            cache[k2] = (bh_ref_plain_R(bank, alpha) if method == "bh_ref" else by_ref_R(bank, alpha)
                         if method == "by_ref" else b7_R(bank, aux["b7"]))
        for s in copies:
            r = rows_of[s]
            out_rows.append(r)
            out_R.append(np.asarray(cache[k2][r], dtype=np.int64))
            out_col.append(None)
        return out_rows, out_R, out_col
    if method in ("bh_iid", "bh_iid_pooled", "b8"):
        if method.startswith("bh_iid") and "null_index" not in cache:
            cache["null_index"] = NullIndex(bank)
        for s in copies:
            cal = np.concatenate([rows_of[c] for c in copies if c != s]) if any(c != s for c in copies) \
                else np.zeros(0, np.int64)
            r = rows_of[s]
            if method == "b8":
                R = b8_R(bank, cal, r, alpha)
            else:
                R = bh_iid_R(bank, cal, r, alpha, pooled=(method == "bh_iid_pooled"), null_index=cache["null_index"])
            out_rows.append(r)
            out_R.append(np.asarray(R, dtype=np.int64))
            out_col.append(None)
        return out_rows, out_R, out_col
    raise ValueError(f"{method} not supported under P-LOSO")


def loso_metrics(bank: LossBank, rows_list: list[np.ndarray], R_list: list[np.ndarray], alpha: float, *,
                 fixed: dict[str, np.ndarray] | None = None, gammas=(0.2, 0.3)) -> dict[str, float]:
    """Window- and scenario-level metrics of a P-LOSO selection (one entry per held-out copy)."""
    win = {k: [] for k in ("R", "V", "A", "TP", "TR", "nT", "nTo", "nTh", "benign")}
    scen_fdp, scen_any, scen_cfdp = [], [], []
    nt, no, nh = bank.col("n_trans"), bank.col("n_trans_obs"), bank.col("n_trans_hid")
    bo = bank.col("benign_only")
    for rows, R in zip(rows_list, R_list):
        if fixed is not None:
            c = {k: np.nan_to_num(fixed[k][rows]).astype(np.int64) for k in ("R", "V", "A", "TP", "TR")}
        else:
            c = bank.counts(rows, R)
        Rf, V = c["R"].astype(float), c["V"].astype(float)
        fdp = V / np.maximum(Rf, 1.0)
        scen_fdp.append(float(fdp.mean()))
        anyc = bool((Rf > 0).any())
        scen_any.append(anyc)
        if anyc:
            scen_cfdp.append(float(fdp.mean()))
        win["R"].append(Rf)
        win["V"].append(V)
        win["A"].append(c["A"].astype(float))
        win["TP"].append(c["TP"].astype(float))
        win["TR"].append(c["TR"].astype(float))
        win["nT"].append(nt[rows])
        win["nTo"].append(no[rows])
        win["nTh"].append(nh[rows])
        win["benign"].append(bo[rows].astype(bool))
    W = {k: np.concatenate(v) for k, v in win.items()}
    R, V, A, TP = W["R"], W["V"], W["A"], W["TP"]
    fs = V / np.maximum(R, 1.0)
    fl = (V - A) / np.maximum(R - A, 1.0)
    cl = R > 0
    nwin = R.size

    def m(x, mask=None):
        x = x if mask is None else x[mask]
        return float(x.mean()) if x.size else math.nan
    out = {
        "fdr_scen": float(np.mean(scen_fdp)),                      # the pre-registered P-LOSO estimand
        "frac_scen_with_claims": float(np.mean(scen_any)),
        "cond_fdp_scen": float(np.mean(scen_cfdp)) if scen_cfdp else math.nan,
        "n_windows": float(nwin), "frac_empty": m((R == 0).astype(float)), "fdr": m(fs), "cond_fdp": m(fs, cl),
        "fdr_lenient": m(fl), "cond_fdp_lenient": m(fl, cl),
        "pooled_fdr": float(V.sum() / max(R.sum(), 1.0)), "pooled_fdr_lenient": float((V - A).sum() / max((R - A).sum(), 1.0)),
        "n_claims": m(R), "n_false": m(V), "n_true_claims": m(TP), "n_claims_cond": m(R, cl), "n_false_cond": m(V, cl),
        "p_any_false": m((V > 0).astype(float)), "cond_p_any_false": m((V > 0).astype(float), cl),
        "p_fdp_gt_2a": m((fs > 2 * alpha + 1e-12).astype(float)),
        "cond_p_fdp_gt_2a": m((fs > 2 * alpha + 1e-12).astype(float), cl),
        "sum_R": float(R.sum()), "sum_V": float(V.sum()), "sum_TP": float(TP.sum()), "sum_TR": float(W["TR"].sum()),
        "str_pooled": float(W["TR"].sum() / W["nT"].sum()) if W["nT"].sum() > 0 else math.nan,
        "ostr_pooled": float(W["TR"].sum() / W["nTo"].sum()) if W["nTo"].sum() > 0 else math.nan,
        "wrongful_benign": m(cl.astype(float), W["benign"]) if W["benign"].any() else math.nan,
        "n_windows_claimed": float(cl.sum()),
    }
    for g in gammas:
        key = f"{int(round(g * 100)):03d}"
        out[f"fdx_{key}"] = m((fs > g + 1e-12).astype(float))
        out[f"cond_fdx_{key}"] = m((fs > g + 1e-12).astype(float), cl)
    return out


LOSO_METRICS = ("fdr_scen", "frac_scen_with_claims", "cond_fdp_scen", "frac_empty", "fdr", "cond_fdp", "pooled_fdr",
                "fdr_lenient", "cond_fdp_lenient", "pooled_fdr_lenient", "n_claims", "n_false", "n_true_claims",
                "n_claims_cond", "n_false_cond", "fdx_020", "fdx_030", "p_fdp_gt_2a", "cond_fdx_020", "cond_fdx_030",
                "cond_p_fdp_gt_2a", "p_any_false", "cond_p_any_false", "str_pooled", "ostr_pooled", "wrongful_benign",
                "sum_R", "sum_V", "sum_TP", "sum_TR")


def loso_eval(bank: LossBank, method: str, alpha: float, *, groups: np.ndarray, copies=None, fixed=None,
              delta: float = 0.1, aux=None, cache=None) -> tuple[dict[str, float], list]:
    g = np.asarray(groups)
    if fixed is not None:
        scen_all = sorted(set(g.tolist()))
        copies = list(scen_all) if copies is None else list(copies)
        rows_list = [np.flatnonzero(g == s) for s in copies]
        R_list = [np.nan_to_num(fixed["R"][r]).astype(np.int64) for r in rows_list]
        cols = [None] * len(copies)
    else:
        rows_list, R_list, cols = loso_window_R(bank, method, alpha, groups=g, copies=copies, delta=delta, aux=aux,
                                                cache=cache)
    return loso_metrics(bank, rows_list, R_list, alpha, fixed=fixed), cols


def loso_bootstrap(bank: LossBank, method: str, alpha: float, *, groups: np.ndarray, B: int = 1000, seed: int = 0,
                   fixed=None, metrics: Sequence[str] = LOSO_METRICS, delta: float = 0.1, aux=None) -> np.ndarray:
    """Scenario-cluster bootstrap of P-LOSO: resample the scenarios with replacement and recompute LOSO (each held-out
    copy calibrated on the copies of the other scenarios). Percentile CIs are bounded by construction."""
    g = np.asarray(groups)
    scen = sorted(set(g.tolist()))
    rng = np.random.default_rng((int(seed), 0x105050))
    reps = np.full((B, len(metrics)), np.nan)
    cache: dict = {}
    for b in range(B):
        copies = [scen[i] for i in rng.integers(0, len(scen), len(scen))]
        res, _ = loso_eval(bank, method, alpha, groups=g, copies=copies, fixed=fixed, delta=delta, aux=aux, cache=cache)
        reps[b] = [res.get(m, np.nan) for m in metrics]
    return reps


def loso_jackknife(bank: LossBank, method: str, alpha: float, *, groups: np.ndarray, metrics: Sequence[str],
                   fixed=None, delta: float = 0.1, aux=None, level: float = 0.95) -> dict[str, dict[str, float]]:
    """Delete-one-scenario jackknife of P-LOSO metrics (as P2-G's jackknife_loso, any metric, any method; one LOSO
    recomputation per deletion serves every metric). CI = estimate +- t_{n-1} SE_jack; ucb95 = one-sided 95%."""
    g = np.asarray(groups)
    scen = sorted(set(g.tolist()))
    full, _ = loso_eval(bank, method, alpha, groups=g, fixed=fixed, delta=delta, aux=aux)
    n = len(scen)
    th = np.full((n, len(metrics)), np.nan)
    cache: dict = {}
    for d in range(n):
        sub, _ = loso_eval(bank, method, alpha, groups=g, copies=[s for s in scen if s != scen[d]], fixed=fixed,
                           delta=delta, aux=aux, cache=cache)
        th[d] = [sub.get(m, np.nan) for m in metrics]
    out = {}
    for j, m in enumerate(metrics):
        est = full.get(m, math.nan)
        t = th[:, j][np.isfinite(th[:, j])]
        if t.size < 2 or not np.isfinite(est):
            out[m] = {"estimate": est, "se_jack": math.nan, "lo": math.nan, "hi": math.nan, "ucb95": math.nan, "n": n}
            continue
        m_ = t.size
        se = float(math.sqrt((m_ - 1) / m_ * np.sum((t - t.mean()) ** 2)))
        q = float(_tdist.ppf(1 - (1 - level) / 2, m_ - 1))
        q1 = float(_tdist.ppf(0.95, m_ - 1))
        out[m] = {"estimate": float(est), "se_jack": se, "lo": est - q * se, "hi": est + q * se, "ucb95": est + q1 * se,
                  "n": n, "n_finite": int(m_)}
    return out


# =========================================================================== statistics helpers
def mde(se: float, level: float, power: float = 0.8) -> float:
    """Minimum detectable effect of a one-sided z test at ``level`` with ``power``: (z_{1-level} + z_power) SE."""
    return float((norm.ppf(1 - level) + norm.ppf(power)) * se)


def power_at(effect: float, se: float, level: float) -> float:
    """Power of the one-sided z test at ``level`` against a true effect ``effect``."""
    return float(norm.sf(norm.ppf(1 - level) - effect / se)) if se > 0 else math.nan


def auc_scores(pos: np.ndarray, neg: np.ndarray) -> float:
    from .gates import auc
    return float(auc(np.asarray(pos, float), np.asarray(neg, float)))


def two_sample_auc_cv(Xa: np.ndarray, Xb: np.ndarray, *, folds: int = 5, seed: int = 0,
                      groups: np.ndarray | None = None, return_oof: bool = False):
    """E10's cross-validated logistic two-sample AUC (bank = 0 vs deployment = 1) on phi. With ``groups`` (incident
    ids of a resampled pool) duplicates are kept in the same fold (StratifiedGroupKFold)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X_ = np.vstack([Xa, Xb])
    X_ = np.where(np.isfinite(X_), X_, 0.0)
    y = np.r_[np.zeros(len(Xa), int), np.ones(len(Xb), int)]
    oof = np.zeros(y.size)
    if groups is None:
        nf = int(min(folds, min(len(Xa), len(Xb))))
        split = StratifiedKFold(nf, shuffle=True, random_state=seed).split(X_, y)
    else:
        nb = len(np.unique(np.asarray(groups)[y == 1]))
        nf = int(min(folds, nb))
        split = StratifiedGroupKFold(nf, shuffle=True, random_state=seed).split(X_, y, groups)
    for tr, te in split:
        if y[tr].min() == y[tr].max():
            oof[te] = 0.5
            continue
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        mdl.fit(X_[tr], y[tr])
        oof[te] = mdl.predict_proba(X_[te])[:, 1]
    a = auc_scores(oof[y == 1], oof[y == 0])
    return (a, oof, y) if return_oof else a


def permutation_auc(Xa: np.ndarray, Xb: np.ndarray, *, P: int = 1000, seed: int = 0, folds: int = 5,
                    cv_seeds: Sequence[int] = (0,)) -> dict[str, Any]:
    """Permutation test of the two-sample AUC: group labels permuted (sizes kept), the same CV statistic recomputed
    (mean over ``cv_seeds``). p = (1 + #{AUC* >= AUC_obs}) / (1 + P)."""
    X_ = np.vstack([Xa, Xb])
    na = len(Xa)
    stat = lambda A, B_: float(np.mean([two_sample_auc_cv(A, B_, folds=folds, seed=s) for s in cv_seeds]))
    obs = stat(Xa, Xb)
    rng = np.random.default_rng((int(seed), 0xA0C))
    null = np.empty(P)
    for p in range(P):
        perm = rng.permutation(X_.shape[0])
        null[p] = stat(X_[perm[:na]], X_[perm[na:]])
    pv = float((1 + np.sum(null >= obs - 1e-12)) / (1 + P))
    return {"auc_obs": obs, "p_perm": pv, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "null_q99": float(np.quantile(null, 0.99)), "P": P, "cv_seeds": list(cv_seeds), "null": null}


def group_boot_auc(Xa: np.ndarray, Xb: np.ndarray, *, B: int = 500, seed: int = 0, folds: int = 5) -> np.ndarray:
    """Stratified incident bootstrap of the CV two-sample AUC (both groups resampled, duplicates kept in one fold)."""
    rng = np.random.default_rng((int(seed), 0xB0A))
    na, nb = len(Xa), len(Xb)
    out = np.empty(B)
    for b in range(B):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        grp = np.r_[ia, na + ib]
        out[b] = two_sample_auc_cv(Xa[ia], Xb[ib], folds=folds, seed=b, groups=grp)
    return out


def exchangeable_null_auc(Xpool: np.ndarray, n_bank: int, n_deploy: int, *, R: int = 500, seed: int = 0,
                          folds: int = 5) -> np.ndarray:
    """Reference distribution of the two-sample AUC under exchangeability: R random (bank, deployment) draws of
    sizes (n_bank, n_deploy) from one pool."""
    rng = np.random.default_rng((int(seed), 0xE0C))
    out = np.empty(R)
    for r in range(R):
        perm = rng.permutation(Xpool.shape[0])
        out[r] = two_sample_auc_cv(Xpool[perm[:n_bank]], Xpool[perm[n_bank:n_bank + n_deploy]], folds=folds, seed=r)
    return out
