"""P2-G experiment layer: the pre-registered analyses of DESIGN §6 on cached loss banks (owner: P2-G).

Reads only derived files (results/losses, results/baselines, results/models, results/interim) and writes

    results/exp/E<n>/<ds>/<method>__<scorer_id>/<cfg_hash>/
        split_<sss>.json      one record per split (every §6.2 metric of eval/runner.evaluate + sums)
        per_split.parquet     the same rows as one table
        summary.parquet       mean / sd / se / split quantiles of every metric (+ ratio-of-sums analyst load)
        cell.json             config, bank input hashes, PREREG BLAKE3, timing
        bootstrap*.json/.npy  incident-pool cluster bootstrap (B x 50 splits) where run
    results/summary/E<n>_numbers.json   every number the paper quotes: value, CI, definition, source,
                                        and a status flag {primary, secondary, exploratory}

Conventions (identical to P2-C's runner; DESIGN §6.2): strict FDP = AMB counted false (the calibrated loss);
lenient = AMB excluded; realised FDR = mean over a split's test incidents (0 for no claims), pooled FDR =
sum V / max(sum R, 1) over the split; STR = certified distinct transitions / |T*|; CER_link = certified true
links / (|E*_link| + |T*_hid|). The ratio-of-sums analyst load (sum R / sum TP over all splits) is added
because the per-split load is infinite whenever a split certifies nothing true.

Statistics (PREREG 'statistics', 'endpoints'): incident-pool cluster bootstrap with P2-C's resampling stream
(eval/bootstrap.cluster_bootstrap: default_rng((seed, 0xB00757)), B = 1000, 50 split seeds 0-49 per
replicate), percentile CIs; validity z = (mean over 200 splits - alpha) / SE_boot, one-sided, Holm at 0.05;
IoT-23 P-LOSO: delete-one-scenario jackknife; Gotham: Clopper-Pearson per incident (conditional).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import blake3
import numpy as np
import polars as pl
from scipy.stats import beta as _beta
from scipy.stats import norm
from scipy.stats import t as _tdist

from ..baselines.bh_iid import NullIndex, bh_iid_R
from ..baselines.by_ref import by_ref_R
from ..baselines.lee_ren import lee_ren_R
from ..baselines.naive import bh_ref_plain_R
from ..certify.crc import crc_threshold_from_sums
from ..certify.losses import CHAIN_KEYS, LossBank
from ..io.paths import P2_ROOT, atomic_write_json, results_root
from ..scoring.calibrate import IsotonicB7, b7_R, b8_R
from . import runner as RN
from .bootstrap import holm, validity_z
from .metrics import certification_gap, chain_metrics, masked_mean, split_metrics
from .prereg import verify_prereg
from .splits import Splits, random_splits

CODE_VERSION = "p2g-1"
PREREG_NAME = "PREREG_2026-10-21.json"
PREREG_BLAKE3 = "955c768e80f31f4040548e0e944c327605396eaedc80430597abfda5b8f610a6"
CO_PRIMARY = (0.1, 0.2)
ALPHAS = (0.05, 0.1, 0.2)
FINE_GRID = tuple(round(0.01 * i, 2) for i in range(1, 51))
N_SPLITS = 200
BOOT_B = 1000
BOOT_SPLITS = 50
BOOT_SEED = 0
BOOT_STREAM = 0xB00757          # eval/bootstrap.cluster_bootstrap

# dataset -> scorer, calibration size K and test incidents per split (PREREG K_per_dataset / test_per_split).
# IoT-23 P-WIN: window-level 50/50 random splits of the 166 windows (DESIGN §5.1.1).
DATASETS: dict[str, dict[str, Any]] = {
    "cf_ton": {"scorer": "S_TON", "K": 150, "n_test": 150},
    "cf_ton_h": {"scorer": "S_TON_H", "K": 100, "n_test": 100},
    "cf_gotham": {"scorer": "S_GOT", "K": 100, "n_test": 100},
    "iot23": {"scorer": "S_CF", "K": 83, "n_test": 83},
    "gotham_real": {"scorer": "S_TON"},
    "cf_gotham_t": {"scorer": "S_TON", "K": 150},
}

LABELS = {
    "crc_bh": "CRC-BH", "crc_bh_raw": "CRC-BH (un-monotonised)", "crc_fdx020": "CRC-FDX ($\\gamma$=0.2)",
    "crc_fdx030": "CRC-FDX ($\\gamma$=0.3)", "ltt": "LTT ($\\delta$=0.1)", "crc_st": "CRC (score threshold)",
    "crc_st_raw": "CRC-ST (un-monotonised)", "bh_ref": "BH-ref (plain)", "by_ref": "BY-ref",
    "bh_iid": "BH-iid (per incident)", "bh_iid_pooled": "BH-iid (pooled)", "b7": "B7 isotonic@0.5",
    "b8": "B8 Platt plug-in", "lr_a1": "LR-A1 (1 draw)", "lr_a1b": "LR-A1b (1 draw)", "lr_a1avg": "LR-A1",
    "lr_a1avgb": "LR-A1b", "lr_a2": "LR-A2", "lr_a2b": "LR-A2b", "chain_composite": "composite CRC",
    "chain_naive": "chain-naive", "chain_rt": "chain-RT", "chain_rt_bh": "chain-RT (BH family)",
    "B1": "B1 King-style", "B2": "B2 BotHunter-style", "B4": "B4 lag-LLR>0", "B5": "B5 LLM only",
    "oracle": "r_C-limited oracle",
}
GUARANTEE = dict(RN.GUARANTEE)
GUARANTEE.update({"crc_bh_raw": "none (ablation)", "crc_st_raw": "none (ablation)", "bh_ref": "none",
                  "by_ref": "none", "b7": "none", "b8": "none (asymptotic heuristic)",
                  "chain_naive": "edge FDR only", "B1": "none", "B2": "none", "B4": "none", "B5": "none",
                  "oracle": "uses test labels"})

# metrics carried through bootstraps (all [S] outputs of runner.evaluate / selection_metrics + sums)
BOOT_METRICS = ("fdr", "fdr_lenient", "pooled_fdr", "pooled_fdr_lenient", "p_fdp_gt_2a", "fdx_020", "fdx_030",
                "n_claims", "frac_empty", "str_mean", "str_pooled", "ostr_mean", "cer_link_mean",
                "cer_link_pooled", "recall_pooled", "bridge_recall", "ucr", "wrongful_benign", "dev_false_impl",
                "abstain", "maxchain_fdr", "maxchain_fdr_lenient", "prefix_fdr", "chain_fwer", "ccr_mean",
                "ccr_pooled", "load_ratio")


# =========================================================================== registry / paths / banks
def prereg_path() -> Path:
    return results_root() / PREREG_NAME


def load_prereg(verify: bool = True) -> dict:
    p = prereg_path()
    if verify and not verify_prereg(p, PREREG_BLAKE3):
        raise RuntimeError(f"{p} does not match the frozen BLAKE3 {PREREG_BLAKE3}")
    return json.loads(p.read_text())


def check_registry(pr: dict) -> None:
    """The dataset registry must agree with the PREREG K / test sizes."""
    kd, tp = pr["K_per_dataset"], pr["test_per_split"]
    for ds in ("cf_ton", "cf_ton_h", "cf_gotham"):
        if DATASETS[ds]["K"] != kd[ds] or DATASETS[ds]["n_test"] != tp[ds]:
            raise RuntimeError(f"registry K/test for {ds} disagrees with the PREREG")
    if list(pr["split_seeds"]["seeds"]) != list(range(N_SPLITS)):
        raise RuntimeError("PREREG split seeds are not 0..199")


def scorer_id(ds: str, kind: str = "stat") -> str:
    return f"{DATASETS[ds]['scorer']}.{kind}"


def bank_dir(ds: str, sid: str) -> Path:
    """results/losses/<ds>/<sid>/bank (P2-F); falls back to P2-G's exploratory banks under results/exp/X1/banks."""
    p = results_root() / "losses" / ds / sid / "bank"
    if not (p / "manifest.json").exists():
        q = results_root() / "exp" / "X1" / "banks" / ds / sid / "bank"
        if (q / "manifest.json").exists():
            return q
    return p


def has_bank(ds: str, sid: str) -> bool:
    return (bank_dir(ds, sid) / "manifest.json").exists()


_BANKS: dict[tuple[str, str], LossBank] = {}


def load_bank(ds: str, sid: str) -> LossBank:
    key = (ds, sid)
    if key not in _BANKS:
        _BANKS[key] = LossBank.load(bank_dir(ds, sid))
    return _BANKS[key]


def bank_hash(bank: LossBank) -> str:
    return str(bank.meta.get("input_hash"))


def exp_root(exp: str) -> Path:
    return results_root() / "exp" / exp


def summary_root() -> Path:
    return results_root() / "summary"


def inc_of(incident_id: str) -> str:
    """Bank incident id '<ds>:<inc>' -> driver incident id."""
    return incident_id.split(":", 1)[1]


def scenario_of(incident_id: str) -> str:
    """IoT-23 window id '<scenario>_b<k>' -> scenario."""
    return inc_of(incident_id).rsplit("_", 1)[0]


# =========================================================================== selection + metrics
def _rows(pool_rows: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = pos >= 0
    rows = np.where(mask, pool_rows[np.where(mask, pos, 0)], 0)
    return rows, mask


@dataclass
class Selection:
    cal_rows: np.ndarray        # [S, K] bank rows
    cal_mask: np.ndarray
    test_rows: np.ndarray       # [S, M]
    tmask: np.ndarray
    R: np.ndarray               # [S, M] selection sizes (0 where masked)
    col: np.ndarray | None      # [S] grid index (loss-table methods)
    family: str | None


def select(bank: LossBank, method: str, splits: Splits, alpha: float, *, pool_rows=None, aux=None,
           cache: dict | None = None, delta: float = 0.1, lr_r: int = 100,
           test_bank: LossBank | None = None) -> Selection:
    """Selection sizes per test incident; the dispatch mirrors eval/runner.evaluate exactly.

    ``test_bank`` (E4 test-only injection): calibration columns come from ``bank`` (clean), the selection
    sizes from ``test_bank`` (row-aligned, injected) at those columns. Only loss-table methods."""
    aux = aux or {}
    cache = {} if cache is None else cache
    pool_rows = np.arange(bank.n) if pool_rows is None else np.asarray(pool_rows, dtype=np.int64)
    cal_rows, cal_mask = _rows(pool_rows, splits.cal)
    test_rows, tmask = _rows(pool_rows, splits.test)
    col = fam = None
    if method in RN.COLUMN_METHODS:
        spec = RN.COLUMN_METHODS[method]
        if spec.get("chain") and not bank.has_chains:
            raise ValueError(f"{method} needs chain tables")
        col = RN.choose_columns(bank, spec, cal_rows, cal_mask, alpha, delta)
        fam = spec["family"]
        tb = bank if test_bank is None else test_bank
        R = tb.t("R", fam)[test_rows, col[:, None]].astype(np.int64)
    else:
        if test_bank is not None:
            raise ValueError("test_bank is only supported for loss-table methods")
        if method in ("bh_ref", "by_ref", "b7"):
            key = (method, alpha)
            if key not in cache:
                if method == "bh_ref":
                    cache[key] = bh_ref_plain_R(bank, alpha)
                elif method == "by_ref":
                    cache[key] = by_ref_R(bank, alpha)
                else:
                    if "b7" not in aux:
                        raise ValueError("b7 needs aux['b7']")
                    cache[key] = b7_R(bank, aux["b7"])
            R = cache[key][test_rows]
        elif method in RN.R_METHODS:
            R = np.zeros(test_rows.shape, dtype=np.int64)
            if method.startswith("bh_iid") and "null_index" not in cache:
                cache["null_index"] = NullIndex(bank)
            for s in range(splits.S):
                cr = cal_rows[s][cal_mask[s]]
                tr = test_rows[s][tmask[s]]
                seed = int(splits.seeds[s])
                if method == "bh_iid":
                    r = bh_iid_R(bank, cr, tr, alpha, pooled=False, null_index=cache["null_index"])
                elif method == "bh_iid_pooled":
                    r = bh_iid_R(bank, cr, tr, alpha, pooled=True, null_index=cache["null_index"])
                elif method == "b8":
                    r = b8_R(bank, cr, tr, alpha)
                else:
                    variant, boost, rr = RN.LR_METHODS[method]
                    r = lee_ren_R(bank, cr, tr, alpha, variant, boost, r=lr_r if rr is None else rr, seed=seed)
                R[s, : r.size] = r
        else:
            raise ValueError(f"unknown method {method!r}")
    R = np.where(tmask, R, 0)
    return Selection(cal_rows, cal_mask, test_rows, tmask, R, col, fam)


def strata_arrays(bank: LossBank, pool_rows: np.ndarray, strata: Sequence[str]) -> dict[str, np.ndarray]:
    """Same as runner._strata_arrays (bool / small-integer index columns, N tercile over the pool)."""
    out = {}
    cols = set(bank.index.columns)
    for s in strata:
        if s == "N_tercile":
            N = bank.col("N")
            q = np.quantile(N[pool_rows], [1 / 3, 2 / 3])
            out[s] = np.searchsorted(q, N, side="right").astype(np.int64)
        elif s in cols:
            v = bank.index[s]
            if v.dtype == pl.Boolean:
                out[s] = v.fill_null(False).to_numpy().astype(np.int64)
            elif v.dtype.is_integer() and v.n_unique() <= 6:
                out[s] = v.fill_null(-1).to_numpy().astype(np.int64)
    return out


def selection_metrics(bank: LossBank, sel: Selection, alpha: float, *, mask: np.ndarray | None = None,
                      gammas=(0.2, 0.3), strata: Sequence[str] = RN.STRATA_DEFAULT, pool_rows=None,
                      cert_gap: bool = False, count_bank: LossBank | None = None) -> dict[str, np.ndarray]:
    """runner.evaluate's metrics for a Selection, optionally restricted to test incidents with ``mask``
    ([S, M] bool, e.g. a stratum). ``count_bank``: bank whose CSR counts the selection (E4 injection)."""
    cb = bank if count_bank is None else count_bank
    tmask = sel.tmask if mask is None else (sel.tmask & mask)
    R = np.where(tmask, sel.R, 0)
    counts = cb.counts(sel.test_rows, R)
    meta = {k: cb.col(k)[sel.test_rows] for k in RN.META_COLS}
    meta["benign_only"] = cb.col("benign_only")[sel.test_rows]
    out = split_metrics(counts, meta, tmask, alpha, gammas)
    out["K"] = sel.cal_mask.sum(axis=1).astype(float)
    out["M"] = tmask.sum(axis=1).astype(float)
    fs = counts["V"] / np.maximum(counts["R"], 1)
    pool_rows = np.arange(bank.n) if pool_rows is None else np.asarray(pool_rows, dtype=np.int64)
    for sname, arr in strata_arrays(cb, pool_rows, strata).items():
        st = arr[sel.test_rows]
        for v in np.unique(st[tmask]):
            out[f"fdr|{sname}={int(v)}"] = masked_mean(fs, tmask & (st == v))
    out["sum_R"] = np.where(tmask, counts["R"], 0).sum(axis=1).astype(float)
    out["sum_TP"] = np.where(tmask, counts["TP"], 0).sum(axis=1).astype(float)
    out["sum_V"] = np.where(tmask, counts["V"], 0).sum(axis=1).astype(float)
    out["sum_TR"] = np.where(tmask, counts["TR"], 0).sum(axis=1).astype(float)
    if sel.col is not None:
        fam = sel.family
        out["lam_idx"] = sel.col.astype(float)
        out["lam"] = cb.grids[fam][sel.col]
        out["abstain"] = (sel.col == 0).astype(float)
        if cb.has_chains:
            ch = {k: cb.t(k, fam)[sel.test_rows, sel.col[:, None]] for k in CHAIN_KEYS}
            out.update(chain_metrics(ch, meta["n_gt_paths"], tmask))
        if cert_gap:
            den = (meta["n_elink"] + meta["n_trans_hid"])[..., None]
            ftab = cb.t("fdp_strict", fam)[sel.test_rows]
            with np.errstate(invalid="ignore", divide="ignore"):
                rtab = np.where(den > 0, cb.t("TP", fam)[sel.test_rows] / np.where(den > 0, den, 1.0), np.nan)
            out["cert_gap_cer_link"] = certification_gap(ftab, rtab, sel.col, tmask)
    return out


def add_sums(res: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """sum_R / sum_TP per split from runner.evaluate output (n_claims x M; load = sum R / sum TP)."""
    if "sum_R" in res:
        return res
    sR = res["n_claims"] * res["M"]
    load = res["analyst_load"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sTP = np.where(np.isfinite(load) & (load > 0), sR / np.where(load > 0, load, 1.0), 0.0)
    res = dict(res)
    res["sum_R"] = np.nan_to_num(sR)
    res["sum_TP"] = np.round(sTP)
    return res


def load_ratio(res: dict[str, np.ndarray]) -> float:
    r = float(np.nansum(res["sum_R"]))
    t = float(np.nansum(res["sum_TP"]))
    return r / t if t > 0 else (math.inf if r > 0 else math.nan)


def evaluate_cell(bank: LossBank, method: str, splits: Splits, alpha: float, *, aux=None, cache=None,
                  pool_rows=None, cert_gap: bool | None = None) -> dict[str, np.ndarray]:
    """P2-C's runner.evaluate (single source of truth for the standard metrics) + sums."""
    res = RN.evaluate(bank, method, splits, alpha, pool_rows=pool_rows, aux=aux, cache=cache, cert_gap=cert_gap)
    return add_sums(res)


# =========================================================================== B7 / aux
def b7_model(scorer: str, variant: str = "stat") -> IsotonicB7:
    """B7 isotonic map on OUT-OF-FOLD training scores of the scorer (stat: oof.parquet; fuse_s0:
    oof_fuse_s0.parquet). DESIGN's B7 is the fused variant; the stat variant is supplementary."""
    md = results_root() / "models" / scorer
    name = "oof.parquet" if variant == "stat" else f"oof_{variant}.parquet"
    df = pl.read_parquet(md / name)
    scol = "s_oof" if "s_oof" in df.columns else ("s" if "s" in df.columns else df.columns[3])
    return IsotonicB7().fit(df[scol].to_numpy(), df["y"].to_numpy())


def aux_for(ds: str, sid: str) -> dict:
    scorer, variant = sid.split(".", 1)
    try:
        return {"b7": b7_model(scorer, variant.split("__", 1)[0])}
    except (FileNotFoundError, OSError):
        return {}


# =========================================================================== cell storage
def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def cfg_hash(cfg: dict, *bank_hashes: str) -> str:
    h = blake3.blake3()
    h.update(_canon(cfg).encode())
    for b in bank_hashes:
        h.update(str(b).encode())
    return h.hexdigest()[:16]


def cell_dir(exp: str, ds: str, method: str, sid: str, h: str) -> Path:
    return exp_root(exp) / ds / f"{method}__{sid}" / h


def frame_from(res: dict[str, np.ndarray], method: str, sid: str, alpha: float, seeds: np.ndarray,
               extra: dict | None = None) -> pl.DataFrame:
    rows = []
    for i in range(int(np.asarray(seeds).size)):
        row = {"method": method, "scorer_id": sid, "seed": int(seeds[i]), "alpha": float(alpha),
               "guarantee": GUARANTEE.get(method, "none")}
        row.update(extra or {})
        for k, v in res.items():
            v = np.asarray(v)
            if v.ndim == 1 and v.size == np.asarray(seeds).size:
                row[k] = float(v[i])
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def summarize_frame(df: pl.DataFrame) -> dict[str, Any]:
    """Mean / sd / se / split quantiles of every numeric column; analyst load as a ratio of sums."""
    out: dict[str, Any] = {"n_splits": df.height}
    for c, t in zip(df.columns, df.dtypes):
        if not t.is_numeric() or c in ("seed", "alpha"):
            continue
        v = df[c].to_numpy().astype(float)
        f = v[np.isfinite(v)]
        if f.size == 0:
            out[c] = {"mean": None, "sd": None, "se": None, "q05": None, "q50": None, "q95": None, "n": 0}
            continue
        sd = float(f.std(ddof=1)) if f.size > 1 else 0.0
        out[c] = {"mean": float(f.mean()), "sd": sd, "se": sd / math.sqrt(f.size), "q05": float(np.quantile(f, .05)),
                  "q50": float(np.quantile(f, .5)), "q95": float(np.quantile(f, .95)), "n": int(f.size),
                  "n_inf": int(np.isinf(v).sum())}
    if "sum_R" in df.columns and "sum_TP" in df.columns:
        r, t = float(df["sum_R"].sum()), float(df["sum_TP"].sum())
        out["load_ratio"] = r / t if t > 0 else (None if r == 0 else math.inf)
    return out


def summary_parquet(df: pl.DataFrame) -> pl.DataFrame:
    s = summarize_frame(df)
    row: dict[str, Any] = {"method": df["method"][0] if df.height else None, "n_splits": s["n_splits"],
                           "load_ratio": s.get("load_ratio")}
    for k, v in s.items():
        if isinstance(v, dict):
            for q in ("mean", "sd", "se", "q05", "q50", "q95"):
                row[f"{k}_{q}"] = v[q]
    return pl.DataFrame([row], infer_schema_length=None)


def write_cell(path: Path, df: pl.DataFrame, cfg: dict, meta: dict) -> Path:
    """Atomic directory write: split_<sss>.json for every split + per_split/summary parquet + cell.json."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.mkdir(parents=True)
    for i, r in enumerate(df.iter_rows(named=True)):
        sid = int(r.get("seed", i))
        (tmp / f"split_{sid:03d}.json").write_text(json.dumps(r, default=float, allow_nan=True))
    df.write_parquet(tmp / "per_split.parquet")
    summary_parquet(df).write_parquet(tmp / "summary.parquet")
    summ = summarize_frame(df)
    (tmp / "cell.json").write_text(json.dumps({"config": cfg, **meta, "summary": summ,
                                               "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                              indent=1, default=str, allow_nan=True))
    if path.exists():
        old = path.with_name(f".{path.name}.old-{os.getpid()}-{time.time_ns()}")
        os.replace(path, old)
        os.replace(tmp, path)
        shutil.rmtree(old, ignore_errors=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, path)
    return path


def read_cell(path: Path) -> tuple[pl.DataFrame, dict]:
    return pl.read_parquet(path / "per_split.parquet"), json.loads((path / "cell.json").read_text())


def run_cell(exp: str, ds: str, sid: str, method: str, alpha: float, *, K: int, n_test: int | None,
             seeds: Iterable[int] = range(N_SPLITS), splits: Splits | None = None, split_tag: str | None = None,
             aux: dict | None = None, aux_tag: str | None = None, extra_cfg: dict | None = None,
             overwrite: bool = False, cert_gap: bool | None = None,
             test_sid: str | None = None, bank: LossBank | None = None) -> tuple[pl.DataFrame, dict, Path]:
    """One (dataset, scorer, method, alpha, K) cell over the given splits; resumable by config hash."""
    bank = load_bank(ds, sid) if bank is None else bank
    seeds = np.asarray(list(seeds) if splits is None else splits.seeds, dtype=np.int64)
    sp = splits if splits is not None else random_splits(bank.n, K, seeds, n_test)
    tb = load_bank(ds, test_sid) if test_sid else None
    cfg = {"exp": exp, "ds": ds, "scorer_id": sid, "method": method, "alpha": float(alpha), "K": int(K),
           "n_test": n_test, "seeds": [int(seeds[0]), int(seeds[-1]), int(seeds.size)] if seeds.size else [],
           "split_tag": split_tag, "aux": aux_tag, "test_scorer_id": test_sid, "prereg": PREREG_BLAKE3,
           "code": CODE_VERSION, **(extra_cfg or {})}
    hashes = [bank_hash(bank)] + ([bank_hash(tb)] if tb is not None else [])
    h = cfg_hash(cfg, *hashes)
    path = cell_dir(exp, ds, method, sid if test_sid is None else f"{sid}~{test_sid}", h)
    if (path / "cell.json").exists() and not overwrite:
        df, meta = read_cell(path)
        return df, meta, path
    t0 = time.perf_counter()
    if tb is None:
        res = evaluate_cell(bank, method, sp, alpha, aux=aux, cert_gap=cert_gap)
    else:
        sel = select(bank, method, sp, alpha, aux=aux, test_bank=tb)
        res = selection_metrics(bank, sel, alpha, count_bank=tb, cert_gap=bool(cert_gap))
    sec = time.perf_counter() - t0
    df = frame_from(res, method, sid, alpha, sp.seeds, {"ds": ds, "K": int(K)})
    meta = {"bank_input_hash": hashes, "seconds": sec, "prereg_blake3": PREREG_BLAKE3}
    write_cell(path, df, cfg, meta)
    return df, {"config": cfg, **meta, "summary": summarize_frame(df)}, path


def collect(exp: str) -> pl.DataFrame:
    """Index of all finished cells of an experiment (one row per cell)."""
    rows = []
    for p in sorted(exp_root(exp).glob("*/*/*/cell.json")):
        c = json.loads(p.read_text())
        cfg = c["config"]
        s = c.get("summary", {})
        row = {k: cfg.get(k) for k in ("ds", "scorer_id", "method", "alpha", "K", "n_test", "split_tag",
                                         "test_scorer_id")}
        row["path"] = str(p.parent)
        for m in ("fdr", "fdr_lenient", "pooled_fdr", "str_mean", "str_pooled", "cer_link_mean", "abstain"):
            row[m] = (s.get(m) or {}).get("mean") if isinstance(s.get(m), dict) else None
        row["load_ratio"] = s.get("load_ratio")
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# =========================================================================== bootstrap
def boot_pools(n: int, B: int, seed: int = BOOT_SEED) -> Iterable[np.ndarray]:
    """The resampled pools of eval/bootstrap.cluster_bootstrap (identical RNG stream)."""
    rng = np.random.default_rng((int(seed), BOOT_STREAM))
    pool = np.arange(n, dtype=np.int64)
    for _ in range(B):
        yield pool[rng.integers(0, pool.size, pool.size)]


def _boot_vector(res: dict[str, np.ndarray], metrics: Sequence[str]) -> np.ndarray:
    out = np.full(len(metrics), np.nan)
    for i, m in enumerate(metrics):
        if m == "load_ratio":
            out[i] = load_ratio(res) if "sum_R" in res else np.nan
        elif m in res:
            v = np.asarray(res[m], float)
            v = v[np.isfinite(v)]
            out[i] = float(v.mean()) if v.size else np.nan
    return out


def bootstrap_reps(bank: LossBank, method: str, alpha: float, *, K: int, n_test: int | None,
                   metrics: Sequence[str] = BOOT_METRICS, B: int = BOOT_B, n_splits: int = BOOT_SPLITS,
                   seed: int = BOOT_SEED, aux=None, bank_b: LossBank | None = None, test_bank=None,
                   mask_col: np.ndarray | None = None, progress: Callable[[int], None] | None = None) -> dict:
    """Incident-pool cluster bootstrap (B replicates x n_splits random splits of the resampled pool).

    Returns replicate matrices [B, len(metrics)] for ``bank`` (and ``bank_b`` on the SAME resampled pools and
    splits, for paired differences). ``mask_col`` [bank.n] bool restricts the metrics to test incidents of a
    stratum (calibration unchanged). ``test_bank``: E4 test-only injection (calibration from ``bank``)."""
    seeds = np.arange(n_splits)
    ca, cb = {}, {}
    reps_a = np.full((B, len(metrics)), np.nan)
    reps_b = np.full((B, len(metrics)), np.nan) if bank_b is not None else None
    custom = mask_col is not None or test_bank is not None
    for b, rows in enumerate(boot_pools(bank.n, B, seed)):
        sp = random_splits(rows.size, K, seeds, n_test)
        for bk, cache, reps in ((bank, ca, reps_a), (bank_b, cb, reps_b)):
            if bk is None:
                continue
            if custom:
                sel = select(bk, method, sp, alpha, pool_rows=rows, aux=aux, cache=cache, test_bank=test_bank)
                mask = None if mask_col is None else mask_col[sel.test_rows]
                res = selection_metrics(bk, sel, alpha, mask=mask, pool_rows=rows, count_bank=test_bank,
                                        strata=())
            else:
                res = evaluate_cell(bk, method, sp, alpha, aux=aux, cache=cache, pool_rows=rows, cert_gap=False)
            reps[b] = _boot_vector(res, metrics)
        if progress is not None and (b + 1) % 50 == 0:
            progress(b + 1)
    return {"metrics": list(metrics), "reps_a": reps_a, "reps_b": reps_b, "B": B, "n_splits": n_splits,
            "seed": seed}


def ci_from_reps(reps: np.ndarray, level: float = 0.95) -> dict[str, float]:
    f = reps[np.isfinite(reps)]
    if f.size < 2:
        return {"se": math.nan, "lo": math.nan, "hi": math.nan, "n": int(f.size)}
    a = (1 - level) / 2
    return {"se": float(f.std(ddof=1)), "lo": float(np.quantile(f, a)), "hi": float(np.quantile(f, 1 - a)),
            "n": int(f.size)}


def write_bootstrap(path: Path, name: str, out: dict, point: dict[str, float] | None = None,
                    extra: dict | None = None) -> dict:
    """bootstrap_<name>.json (+ .npy replicates) inside a cell directory; returns the summary dict."""
    mets = out["metrics"]
    summ: dict[str, Any] = {"B": out["B"], "n_splits_per_replicate": out["n_splits"], "seed": out["seed"],
                            "stream": "default_rng((seed, 0xB00757)) as eval/bootstrap.cluster_bootstrap",
                            "metrics": {}}
    for j, m in enumerate(mets):
        d = ci_from_reps(out["reps_a"][:, j])
        if point and m in point:
            d["estimate"] = point[m]
        if out.get("reps_b") is not None:
            diff = out["reps_b"][:, j] - out["reps_a"][:, j]
            d = {"a": d, "b": ci_from_reps(out["reps_b"][:, j]), "delta": ci_from_reps(diff)}
            d["delta"]["p_one_sided_le0"] = float((1 + np.sum(diff[np.isfinite(diff)] <= 0))
                                                  / (1 + np.isfinite(diff).sum()))
        summ["metrics"][m] = d
    summ.update(extra or {})
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f".bootstrap_{name}.tmp.npy", out["reps_a"])
    os.replace(path / f".bootstrap_{name}.tmp.npy", path / f"bootstrap_{name}_a.npy")
    if out.get("reps_b") is not None:
        np.save(path / f".bootstrap_{name}.tmp.npy", out["reps_b"])
        os.replace(path / f".bootstrap_{name}.tmp.npy", path / f"bootstrap_{name}_b.npy")
    atomic_write_json(path / f"bootstrap_{name}.json", summ)
    return summ


def read_bootstrap(path: Path, name: str) -> dict | None:
    p = path / f"bootstrap_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# =========================================================================== validity endpoint
def validity_family(cells: dict[str, dict]) -> list[dict]:
    """Primary endpoint 1 (PREREG): per cell z = (mean - alpha) / SE (SE_boot; jackknife for P-LOSO),
    one-sided p = 1 - Phi(z), Holm at 0.05 over the cells that were run; missing cells listed."""
    names = [n for n, c in cells.items() if c.get("mean") is not None and c.get("se") is not None]
    zs, ps = [], []
    for n in names:
        z, p = validity_z(float(cells[n]["mean"]), float(cells[n]["se"]), float(cells[n]["alpha"]))
        zs.append(z)
        ps.append(p)
    adj, rej = holm(ps, 0.05) if ps else (np.array([]), np.array([], bool))
    out = []
    for i, n in enumerate(names):
        out.append({"cell": n, **cells[n], "z": zs[i], "p": ps[i], "p_holm": float(adj[i]),
                    "violation": bool(rej[i])})
    for n, c in cells.items():
        if n not in names:
            out.append({"cell": n, **c, "missing": True})
    return out


# =========================================================================== fixed-claim baselines
def baseline_dir(ds: str, b: str) -> Path:
    return results_root() / "baselines" / ds / b


def has_baseline(ds: str, b: str) -> bool:
    return (baseline_dir(ds, b) / "per_incident.parquet").exists()


def _csr_flags(bank: LossBank, row: int) -> dict[int, tuple[int, int]]:
    """cand id -> (bridge-true flag, decoy-endpoint flag) for one incident from the CSR prefix counts."""
    o, o2 = bank.csr["offsets"], bank.csr["off2"]
    a, b = int(o[row]), int(o[row + 1])
    c = bank.csr["cand"][a:b]
    j = int(o2[row])
    n = b - a
    br = np.diff(bank.csr["c_bridge"][j: j + n + 1])
    dc = np.diff(bank.csr["c_decoy"][j: j + n + 1])
    return {int(ci): (int(x), int(y)) for ci, x, y in zip(c, br, dc)}


def baseline_counts(bank: LossBank, ds: str, b: str) -> dict[str, np.ndarray]:
    """Per-bank-row counts (R, V, A, TP, TR, BR, DC; FD = NaN) of a fixed-claim baseline (B1/B2/B4/B5)
    from results/baselines/<ds>/<b>/{per_incident,claims}.parquet. Rows without baseline output -> NaN."""
    pi = pl.read_parquet(baseline_dir(ds, b) / "per_incident.parquet")
    cl = pl.read_parquet(baseline_dir(ds, b) / "claims.parquet")
    ids = [inc_of(i) for i in bank.index["incident_id"].to_list()]
    pos = {inc: k for k, inc in enumerate(ids)}
    n = bank.n
    out = {k: np.full(n, np.nan) for k in ("R", "V", "A", "TP", "TR", "BR", "DC", "FD")}
    ntrans = bank.col("n_trans")
    for r in pi.iter_rows(named=True):
        k = pos.get(r["incident"])
        if k is None:
            continue
        for key in ("R", "V", "A", "TP"):
            out[key][k] = float(r[key])
        nT = float(r["n_T"]) if r.get("n_T") is not None else ntrans[k]
        st = r.get("STR")
        out["TR"][k] = float(round(st * nT)) if (st is not None and np.isfinite(st) and nT > 0) else 0.0
        if r.get("n_T") is not None and np.isfinite(ntrans[k]) and int(r["n_T"]) != int(ntrans[k]):
            raise RuntimeError(f"{ds}/{b}: |T*| of {r['incident']} differs from the bank ({r['n_T']} vs {ntrans[k]})")
    by_inc = cl.group_by("incident").agg(pl.col("cand_id"))
    for inc, cands in zip(by_inc["incident"].to_list(), by_inc["cand_id"].to_list()):
        k = pos.get(inc)
        if k is None:
            continue
        fl = _csr_flags(bank, k)
        br = dc = 0
        for c in cands:
            f = fl.get(int(c))
            if f is not None:
                br += f[0]
                dc += f[1]
        out["BR"][k] = br
        out["DC"][k] = dc
    for k in range(n):
        if np.isfinite(out["R"][k]) and not np.isfinite(out["BR"][k]):
            out["BR"][k] = 0.0
            out["DC"][k] = 0.0
    return out


def fixed_metrics(bank: LossBank, counts: dict[str, np.ndarray], splits: Splits, alpha: float,
                  mask_col: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """split_metrics of a fixed (split-independent) claim set over the test incidents of each split."""
    test_rows, tmask = _rows(np.arange(bank.n), splits.test)
    ok = np.isfinite(counts["R"])[test_rows]
    tmask = tmask & ok
    if mask_col is not None:
        tmask = tmask & mask_col[test_rows]
    c = {k: np.where(tmask, np.nan_to_num(v[test_rows]), 0).astype(np.int64) for k, v in counts.items() if k != "FD"}
    meta = {k: bank.col(k)[test_rows] for k in RN.META_COLS}
    meta["benign_only"] = bank.col("benign_only")[test_rows]
    meta["n_noncamp_dev"] = np.full(test_rows.shape, np.nan)
    c["FD"] = np.zeros(test_rows.shape, np.int64)
    out = split_metrics(c, meta, tmask, alpha)
    out["dev_false_impl"] = np.full(splits.S, np.nan)
    out["M"] = tmask.sum(axis=1).astype(float)
    out["sum_R"] = np.where(tmask, c["R"], 0).sum(1).astype(float)
    out["sum_TP"] = np.where(tmask, c["TP"], 0).sum(1).astype(float)
    out["sum_V"] = np.where(tmask, c["V"], 0).sum(1).astype(float)
    out["sum_TR"] = np.where(tmask, c["TR"], 0).sum(1).astype(float)
    fs = c["V"] / np.maximum(c["R"], 1)
    for sname, arr in strata_arrays(bank, np.arange(bank.n), RN.STRATA_DEFAULT).items():
        st = arr[test_rows]
        for v in np.unique(st[tmask]):
            out[f"fdr|{sname}={int(v)}"] = masked_mean(fs, tmask & (st == v))
    return out


def oracle_counts(bank: LossBank) -> dict[str, np.ndarray]:
    """r_C-limited oracle: certify exactly the true links in C (FDR 0; STR = (1 - rho_T) r_C,T per incident)."""
    o2 = bank.csr["off2"]
    N = bank.sizes()
    end = o2[:-1] + N
    tp = bank.csr["c_tp"][end].astype(float)
    tr = bank.csr["c_trans"][end].astype(float)
    # bridges: c_bridge counts true bridge candidates; all true links are certified
    br = bank.csr["c_bridge"][end].astype(float)
    return {"R": tp, "V": np.zeros(bank.n), "A": np.zeros(bank.n), "TP": tp, "TR": tr, "BR": br,
            "DC": np.zeros(bank.n), "FD": np.zeros(bank.n)}


def transitions_in_C(bank: LossBank) -> np.ndarray:
    """Distinct observable transitions whose enabling pair is a (true) candidate of C, per incident."""
    o2 = bank.csr["off2"]
    return bank.csr["c_trans"][o2[:-1] + bank.sizes()].astype(float)


def descriptors(bank: LossBank) -> dict[str, float]:
    """Table III descriptors of a bank's pool (pooled sums; DESIGN §2.3, §5.5)."""
    N = bank.col("N")
    nt = bank.col("n_trans")
    no = bank.col("n_trans_obs")
    nh = bank.col("n_trans_hid")
    tc = transitions_in_C(bank)
    att = tc > 0
    r_ct = float(np.nansum(tc) / np.nansum(no)) if np.nansum(no) > 0 else math.nan
    rho = float(np.nansum(nh) / np.nansum(nt)) if np.nansum(nt) > 0 else math.nan
    return {"n_incidents": int(bank.n), "N_median": float(np.median(N)), "N_mean": float(np.mean(N)),
            "N_total": float(np.sum(N)), "N_over_r_pooled": float(N[att].sum() / tc[att].sum()) if att.any() else math.nan,
            "N_over_r_all": float(N.sum() / tc.sum()) if tc.sum() > 0 else math.nan,
            "N_over_r_median": float(np.median(N[att] / tc[att])) if att.any() else math.nan,
            "r_CT": r_ct, "rho_T": rho, "ceiling": (1 - rho) * r_ct if np.isfinite(rho) and np.isfinite(r_ct) else math.nan,
            "n_T": float(np.nansum(nt)), "n_T_obs": float(np.nansum(no)), "n_T_hid": float(np.nansum(nh)),
            "n_T_obs_in_C": float(np.nansum(tc)), "n_true": float(bank.col("n_true").sum()),
            "n_amb": float(bank.col("n_amb").sum()), "amb_share": float(bank.col("n_amb").sum() / max(N.sum(), 1)),
            "n_benign_only": int(bank.col("benign_only").sum())}


# =========================================================================== IoT-23 protocols
def scenario_groups(bank: LossBank) -> np.ndarray:
    return np.asarray([scenario_of(i) for i in bank.index["incident_id"].to_list()])


def _per_window(bank: LossBank, rows: np.ndarray, R: np.ndarray) -> dict[str, np.ndarray]:
    c = bank.counts(rows, R)
    Rf = c["R"].astype(float)
    V, A = c["V"].astype(float), c["A"].astype(float)
    return {"R": Rf, "V": V, "A": A, "TP": c["TP"].astype(float), "TR": c["TR"].astype(float),
            "fdp": V / np.maximum(Rf, 1.0), "fdp_lenient": (V - A) / np.maximum(Rf - A, 1.0)}


def loso(bank: LossBank, alpha: float, *, method: str = "crc_bh", scenarios: Sequence[str] | None = None,
         fixed: dict[str, np.ndarray] | None = None, groups: np.ndarray | None = None) -> pl.DataFrame:
    """P-LOSO (DESIGN §5.1.1): incident = scenario, scenario loss path = mean of its windows' monotonised
    losses; leave-one-scenario-out CRC with K = (#scenarios - 1). One row per held-out scenario with the
    scenario FDP (mean window FDP, strict / lenient) and power counts. ``method``: crc_bh | crc_st |
    crc_fdx020 | crc_fdx030 (scenario-level CRC), bh_iid (window-level nulls of the other scenarios), or
    'fixed' with ``fixed`` per-row counts (B2). ``scenarios``: the population (jackknife deletions)."""
    g = scenario_groups(bank) if groups is None else np.asarray(groups)
    scen = sorted(set(g.tolist())) if scenarios is None else list(scenarios)
    scen_rows = {s: np.flatnonzero(g == s) for s in scen}
    Kc = len(scen) - 1
    rows_out = []
    if method in RN.COLUMN_METHODS:
        spec = RN.COLUMN_METHODS[method]
        fam, tab = spec["family"], spec["table"]
        T = bank.t(tab, fam)
        Ls = np.stack([T[scen_rows[s]].mean(axis=0) for s in scen])
    ni = NullIndex(bank) if method == "bh_iid" else None
    nt, no, nh = bank.col("n_trans"), bank.col("n_trans_obs"), bank.col("n_trans_hid")
    ne = bank.col("n_elink")
    tc = transitions_in_C(bank)
    for j, s in enumerate(scen):
        rows = scen_rows[s]
        col = None
        if method in RN.COLUMN_METHODS:
            others = [i for i in range(len(scen)) if i != j]
            col = int(crc_threshold_from_sums(Ls[others].sum(axis=0)[None, :], np.array([Kc]), alpha)[0])
            R = bank.t("R", fam)[rows, col].astype(np.int64)
            w = _per_window(bank, rows, R)
        elif method == "bh_iid":
            cal = np.concatenate([scen_rows[x] for x in scen if x != s])
            R = bh_iid_R(bank, cal, rows, alpha, pooled=False, null_index=ni)
            w = _per_window(bank, rows, R)
        elif method == "fixed":
            w = {k: np.nan_to_num(fixed[k][rows]) for k in ("R", "V", "A", "TP", "TR")}
            w["fdp"] = w["V"] / np.maximum(w["R"], 1.0)
            w["fdp_lenient"] = (w["V"] - w["A"]) / np.maximum(w["R"] - w["A"], 1.0)
        else:
            raise ValueError(method)
        sT, sTo, sTh = float(np.nansum(nt[rows])), float(np.nansum(no[rows])), float(np.nansum(nh[rows]))
        rows_out.append({
            "scenario": s, "n_windows": int(rows.size), "lam_idx": col, "abstain": (col == 0) if col is not None else None,
            "fdp": float(w["fdp"].mean()), "fdp_lenient": float(w["fdp_lenient"].mean()),
            "n_claims": float(w["R"].mean()), "sum_R": float(w["R"].sum()), "sum_V": float(w["V"].sum()),
            "sum_TP": float(w["TP"].sum()), "sum_TR": float(w["TR"].sum()),
            "any_claim": float((w["R"] > 0).mean()),
            "n_T": sT, "n_T_obs": sTo, "n_T_hid": sTh, "n_T_obs_in_C": float(np.nansum(tc[rows])),
            "n_elink": float(np.nansum(ne[rows])),
            "str": float(w["TR"].sum() / sT) if sT > 0 else None,
            "rho_T": sTh / sT if sT > 0 else None,
            "r_CT": float(np.nansum(tc[rows]) / sTo) if sTo > 0 else None,
            "benign_only": bool(bank.col("benign_only")[rows].all()),
        })
    return pl.DataFrame(rows_out, infer_schema_length=None)


def jackknife_loso(bank: LossBank, alpha: float, *, method: str = "crc_bh", fixed=None, metric: str = "fdp",
                   level: float = 0.95) -> dict[str, Any]:
    """LOSO mean of ``metric`` over scenarios and its delete-one-scenario jackknife (each deletion re-runs
    LOSO on the remaining scenarios, K = n - 2). CI = estimate +- t_{n-1} SE_jack; the naive SE (SD of
    the per-scenario values / sqrt(n)) is reported too."""
    full = loso(bank, alpha, method=method, fixed=fixed)
    vals = full[metric].to_numpy().astype(float)
    est = float(np.mean(vals))
    scen = full["scenario"].to_list()
    n = len(scen)
    th = np.empty(n)
    for d in range(n):
        sub = loso(bank, alpha, method=method, fixed=fixed, scenarios=[s for s in scen if s != scen[d]])
        th[d] = float(sub[metric].to_numpy().astype(float).mean())
    se = float(math.sqrt((n - 1) / n * np.sum((th - th.mean()) ** 2)))
    q = float(_tdist.ppf(1 - (1 - level) / 2, n - 1))
    return {"estimate": est, "se_jack": se, "lo": est - q * se, "hi": est + q * se, "n_scenarios": n,
            "t_quantile": q, "se_naive": float(np.std(vals, ddof=1) / math.sqrt(n)), "jack_values": th.tolist(),
            "per_scenario": full}


def shift_splits(groups: np.ndarray, n_splits: int = 100, n_cal_groups: int | None = None,
                 salt: int = 7) -> Splits:
    """P-SHIFT: per split seed s, a random half of the scenarios (default ceil(G/2)) calibrates at the window
    level and all windows of the other scenarios are test windows. Positions padded with -1."""
    g = np.asarray(groups)
    scen = sorted(set(g.tolist()))
    G = len(scen)
    kc = int(math.ceil(G / 2)) if n_cal_groups is None else int(n_cal_groups)
    cal, test, seeds = [], [], []
    for s in range(n_splits):
        perm = np.random.default_rng((s, salt)).permutation(G)
        A = {scen[i] for i in perm[:kc]}
        cal.append(np.flatnonzero(np.isin(g, list(A))))
        test.append(np.flatnonzero(~np.isin(g, list(A))))
        seeds.append(s)

    def pad(rows):
        m = max(len(r) for r in rows)
        out = np.full((len(rows), m), -1, dtype=np.int64)
        for i, r in enumerate(rows):
            out[i, : len(r)] = r
        return out
    return Splits(np.asarray(seeds, dtype=np.int64), pad(cal), pad(test))


# =========================================================================== Gotham case study
def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided CI for a binomial proportion k / n (NaN if n = 0)."""
    if n <= 0:
        return (math.nan, math.nan)
    a = 1 - level
    lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def crc_full_bank(bank: LossBank, alpha: float, method: str = "crc_bh") -> int:
    """lambda-hat index from ALL incidents of a calibration bank (K = bank.n)."""
    spec = RN.COLUMN_METHODS[method]
    sums = bank.t(spec["table"], spec["family"]).sum(axis=0)[None, :]
    return int(crc_threshold_from_sums(sums, np.array([bank.n]), alpha)[0])


# =========================================================================== chains
def longest_gt_path(transitions: Sequence[Sequence[str]]) -> int:
    """Number of transitions on the longest root->leaf path of the episode DAG (0 if none)."""
    succ: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for a, b in transitions:
        succ.setdefault(a, []).append(b)
        nodes |= {a, b}
    memo: dict[str, int] = {}

    def depth(x: str, stack: frozenset) -> int:
        if x in memo:
            return memo[x]
        best = 0
        for y in succ.get(x, []):
            if y in stack:
                continue
            best = max(best, 1 + depth(y, stack | {y}))
        memo[x] = best
        return best
    return max((depth(n, frozenset({n})) for n in nodes), default=0)


def chain_length_strata(ds: str, bank: LossBank) -> np.ndarray:
    """GT longest-path length L per bank row (GT side, via the driver's guarded reader)."""
    from ..scoring import driver as D
    out = np.zeros(bank.n, dtype=np.int64)
    for k, iid in enumerate(bank.index["incident_id"].to_list()):
        gt = D.load_gt_json(ds, inc_of(iid))
        out[k] = longest_gt_path([tuple(t) for t in gt.get("transitions", [])])
    return out


# =========================================================================== content-rich stratum
def content_rich(ds: str, bank: LossBank) -> np.ndarray:
    """>= 1 unit with a content flag (label-free; PREREG strata.content_rich)."""
    from ..scoring import driver as D
    out = np.zeros(bank.n, dtype=bool)
    for k, iid in enumerate(bank.index["incident_id"].to_list()):
        u = D.load_units(ds, inc_of(iid))
        cf = u["content_flags"].fill_null(0).to_numpy() if "content_flags" in u.columns else np.zeros(1)
        out[k] = bool(np.any(cf != 0))
    return out


# =========================================================================== numbers registry
class Numbers:
    """results/summary/<exp>_numbers.json: key -> {value, ci, se, definition, source, status, ...}."""

    def __init__(self, exp: str):
        self.exp = exp
        self.path = summary_root() / f"{exp}_numbers.json"
        self.d: dict[str, Any] = {}
        if self.path.exists():
            try:
                self.d = json.loads(self.path.read_text()).get("numbers", {})
            except json.JSONDecodeError:
                self.d = {}

    def add(self, key: str, value, *, definition: str, ci=None, se=None, source=None, status: str = "secondary",
            **extra) -> None:
        def clean(x):
            if isinstance(x, (np.floating, np.integer)):
                x = x.item()
            if isinstance(x, float) and not math.isfinite(x):
                return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
            return x
        ent = {"value": clean(value), "definition": definition, "status": status}
        if ci is not None:
            ent["ci"] = [clean(ci[0]), clean(ci[1])]
        if se is not None:
            ent["se"] = clean(se)
        if source is not None:
            ent["source"] = str(source)
        for k, v in extra.items():
            ent[k] = clean(v) if not isinstance(v, (list, dict)) else v
        self.d[key] = ent

    def write(self) -> Path:
        return atomic_write_json(self.path, {"exp": self.exp, "prereg_blake3": PREREG_BLAKE3,
                                             "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "numbers": self.d})


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(P2_ROOT))
    except ValueError:
        return str(p)


# =========================================================================== task layer (process pools)
def cell_point(path: Path) -> dict[str, float]:
    """200-split means of every metric of a finished cell (point estimates for bootstrap summaries)."""
    df, meta = read_cell(path)
    out = {}
    for c, t in zip(df.columns, df.dtypes):
        if t.is_numeric() and c not in ("seed", "alpha"):
            v = df[c].to_numpy().astype(float)
            v = v[np.isfinite(v)]
            out[c] = float(v.mean()) if v.size else math.nan
    s = meta.get("summary", {}).get("load_ratio")
    out["load_ratio"] = float(s) if isinstance(s, (int, float)) else (math.inf if s == "inf" else math.nan)
    return out


def _mask_from_spec(bank: LossBank, ds: str, spec: dict | None) -> np.ndarray | None:
    """Stratum mask over bank rows: {'col': <index column>, 'value': v} or {'kind': 'content_rich'} or
    {'kind': 'chain_len', 'L': [..]}."""
    if not spec:
        return None
    if spec.get("kind") == "content_rich":
        return content_rich(ds, bank)
    if spec.get("kind") == "chain_len":
        L = chain_length_strata(ds, bank)
        return np.isin(L, spec["L"])
    col = bank.index[spec["col"]]
    v = col.fill_null(False if col.dtype == pl.Boolean else -1).to_numpy()
    return v == spec["value"]


def run_task(task: dict) -> dict:
    """Worker entry point. kinds: cell | boot (single or paired, optional stratum mask / test bank)."""
    kind = task["kind"]
    t0 = time.perf_counter()
    if kind == "cell":
        aux = aux_for(task["ds"], task["sid"]) if task["method"] == "b7" else None
        if task["method"] == "b7" and not aux:
            return {"task": task, "skipped": "no OOF scores for B7"}
        df, meta, path = run_cell(task["exp"], task["ds"], task["sid"], task["method"], task["alpha"],
                                  K=task["K"], n_test=task.get("n_test"), seeds=range(task.get("splits", N_SPLITS)),
                                  aux=aux, aux_tag=("b7:oof" if aux else None), test_sid=task.get("test_sid"),
                                  overwrite=task.get("overwrite", False), cert_gap=task.get("cert_gap"),
                                  extra_cfg=task.get("extra_cfg"))
        s = meta.get("summary", {})
        return {"task": task, "path": str(path), "fdr": (s.get("fdr") or {}).get("mean"),
                "str_mean": (s.get("str_mean") or {}).get("mean"), "seconds": time.perf_counter() - t0}
    if kind == "boot":
        path = Path(task["cell"])
        name = task["name"]
        if (path / f"bootstrap_{name}.json").exists() and not task.get("overwrite", False):
            return {"task": task, "path": str(path), "cached": True}
        bank = load_bank(task["ds"], task["sid"])
        bank_b = load_bank(task["ds"], task["sid_b"]) if task.get("sid_b") else None
        if bank_b is not None and (bank_b.index["incident_id"].to_list() != bank.index["incident_id"].to_list()):
            raise RuntimeError(f"paired banks {task['sid']} / {task['sid_b']} are not row-aligned")
        tb = load_bank(task["ds"], task["test_sid"]) if task.get("test_sid") else None
        aux = aux_for(task["ds"], task["sid"]) if task["method"] == "b7" else None
        mask = _mask_from_spec(bank, task["ds"], task.get("mask"))
        out = bootstrap_reps(bank, task["method"], task["alpha"], K=task["K"], n_test=task.get("n_test"),
                             metrics=tuple(task.get("metrics", BOOT_METRICS)), B=task.get("B", BOOT_B),
                             n_splits=task.get("boot_splits", BOOT_SPLITS), seed=task.get("seed", BOOT_SEED),
                             aux=aux, bank_b=bank_b, test_bank=tb, mask_col=mask)
        point = cell_point(path) if (path / "cell.json").exists() and not task.get("mask") else None
        summ = write_bootstrap(path, name, out, point, extra={"task": {k: v for k, v in task.items()
                                                                          if k not in ("kind",)},
                                                                "seconds": time.perf_counter() - t0})
        return {"task": task, "path": str(path), "seconds": time.perf_counter() - t0,
                "fdr": summ["metrics"].get("fdr")}
    if kind == "synth_pool":
        return _synth_pool_task(task)
    if kind == "probe":
        return _probe_task(task)
    raise ValueError(kind)


SYNTH_SEED_BASE = {"decoy_rich": 777_000, "decoy_free": 888_000}


def _synth_pool_task(task: dict) -> dict:
    """E7 synthetic panel: one pool of P2-C's generator (eval/synth.py), all (method, K, alpha) on
    ``splits_per_pool`` splits with salt = pool (as P2-C's few-shot runs)."""
    from ..certify.losses import build_loss_tables
    from . import synth
    from .gates import bank_within_auc
    gen, p, sc = task["gen"], int(task["pool"]), task["cfg"]
    out = exp_root("E7") / "synth" / gen / f"p{p}"
    if (out / "rows.parquet").exists() and not task.get("overwrite", False):
        return {"task": task, "cached": True}
    prm = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in task["params"].items()}
    cfg = synth.SynthConfig().with_(**prm)
    base = SYNTH_SEED_BASE.get(gen, 999_000)
    s0 = synth.synth_null_pool(int(sc["null_incidents"]), base + 2 * p, cfg)
    pool = synth.synth_pool(int(sc["pool_size"]), base + 2 * p + 1, cfg)
    bank = build_loss_tables(pool, s0, out / "bank", n_jobs=1, overwrite=True)
    frames = []
    for K in task["K_grid"]:
        sp = random_splits(bank.n, int(K), range(int(sc["splits_per_pool"])), int(sc["n_test"]), salt=p)
        cache: dict = {}
        for alpha in task["alphas"]:
            for m in task["methods"]:
                res = evaluate_cell(bank, m, sp, float(alpha), cache=cache, cert_gap=False)
                frames.append(frame_from(res, m, "synth", float(alpha), sp.seeds,
                                         {"K": int(K), "pool": p, "gen": gen}))
    df = pl.concat(frames, how="diagonal_relaxed")
    auc = bank_within_auc(bank)["mean"]
    df = df.with_columns(pl.lit(float(auc)).alias("within_auc"))
    tmp = out / f".rows.tmp-{os.getpid()}.parquet"
    df.write_parquet(tmp)
    os.replace(tmp, out / "rows.parquet")
    return {"task": task, "rows": df.height, "within_auc": auc}


def _probe_task(task: dict) -> dict:
    ds, role, pc = task["ds"], task["role"], task["cfg"]
    out = exp_root("E15") / "probe_units" / f"{ds}_{role}.parquet"
    if out.exists() and not task.get("overwrite", False):
        df = pl.read_parquet(out)
    else:
        df = probe_table(ds, role)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
        df.write_parquet(tmp)
        os.replace(tmp, out)
    aucs = {}
    if df.height == 0 or "kind" not in df.columns:      # e.g. a bank-only variant without a training role
        return {"task": task, "aucs": aucs, "kinds": []}
    sets = {"all": PROBE_TIMING + PROBE_SCORER, "timing": PROBE_TIMING, "scorer": PROBE_SCORER}
    for target in ("i", "ii", "i_attack", "ii_typematched"):
        for name, feats in sets.items():
            aucs[f"{ds}|{role}|{target}|{name}"] = probe_auc(df, target, feats, folds=int(pc.get("folds", 5)),
                                                             params=pc.get("lightgbm"), rounds=int(pc.get("rounds", 200)),
                                                             B=int(pc.get("bootstrap_B", 200)))
    kinds = df.group_by("kind").len().to_dicts() if df.height else []
    return {"task": task, "aucs": aucs, "kinds": kinds}


# =========================================================================== E15 artifact probes (DESIGN §5.4.4)
PROBE_TIMING = ("frac_ts", "frac_rec_mean", "frac_rec_zero", "t_since_t0", "t_norm", "rank_norm", "n_rec_probe",
                "gap_mean", "gap_median", "gap_std", "gap_min", "gap_max", "gap_cv", "gap_zero")
PROBE_SCORER = ("type", "lobytes", "lrbytes", "lpkts", "lnrec", "lndst", "fail", "ldur", "startcv", "lndports",
                "svc", "port", "dir", "c_exec", "c_inj", "c_toolua", "c_nxdom", "c_hient", "seed")
PROBE_CATEGORICAL = ("type", "svc", "port", "dir")


def _incident_dir(ds: str, inc: str) -> Path:
    from ..scoring import driver as D
    parts = inc.split("/")
    return D.interim_dir(ds) / "incidents" / Path(*parts[1:])


def probe_units(ds: str, inc: str) -> pl.DataFrame:
    """One row per unit: timing/position probe features, the scorer's unit-level features, and the GT kind of
    the unit (main | D1 | D2 | D3 | background | mixed). GT side (guarded reader)."""
    from ..features.stats import unit_features
    from ..scoring import driver as D
    units = D.load_units(ds, inc)
    if units.height == 0:
        return pl.DataFrame()
    side = D.load_side(ds, inc)
    uf = unit_features(units, side)
    meta = json.loads((_incident_dir(ds, inc) / "meta.json").read_text())
    t0, t1 = float(meta["t0"]), float(meta["t1"])
    rec = pl.read_parquet(_incident_dir(ds, inc) / "records.parquet", columns=["rec_id", "ts"])
    ts_of = dict(zip(rec["rec_id"].to_list(), rec["ts"].to_list()))
    rows = []
    order = np.argsort(units["ts"].to_numpy(), kind="stable")
    rank = np.empty(units.height)
    rank[order] = (np.arange(units.height) + 0.5) / units.height
    for k, (uid, ts, rids) in enumerate(zip(units["unit_id"].to_list(), units["ts"].to_list(),
                                             units["rec_ids"].to_list())):
        rts = np.sort(np.array([ts_of[r] for r in (rids or []) if r in ts_of], dtype=np.float64))
        fr = rts - np.floor(rts) if rts.size else np.array([0.0])
        gaps = np.diff(rts) if rts.size > 1 else np.array([0.0])
        gm = float(gaps.mean())
        rows.append({"unit_id": int(uid), "frac_ts": float(ts - math.floor(ts)), "frac_rec_mean": float(fr.mean()),
                     "frac_rec_zero": float(np.mean(fr == 0)), "t_since_t0": float(ts - t0),
                     "t_norm": float((ts - t0) / max(t1 - t0, 1e-9)), "rank_norm": float(rank[k]),
                     "n_rec_probe": float(rts.size), "gap_mean": gm, "gap_median": float(np.median(gaps)),
                     "gap_std": float(gaps.std()), "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
                     "gap_cv": float(gaps.std() / gm) if gm > 0 else 0.0, "gap_zero": float(np.mean(gaps == 0))})
    pf = pl.DataFrame(rows)
    gtu = D.load_gt_units(ds, inc)
    gtj = D.load_gt_json(ds, inc)
    camps = (gtj.get("meta") or {}).get("campaigns") or {}
    kind_of = {c: (d or {}).get("kind", "main") for c, d in camps.items()}
    kinds = {}
    for uid, camp, mixed in zip(gtu["unit_id"].to_list(), gtu["campaign"].to_list(), gtu["mixed"].to_list()):
        kinds[int(uid)] = "mixed" if mixed else (kind_of.get(camp, "main") if camp is not None else "background")
    kcol = [kinds.get(int(u), "background") for u in pf["unit_id"].to_list()]
    out = pf.join(uf.select("unit_id", *[c for c in PROBE_SCORER if c in uf.columns]), on="unit_id", how="left")
    return out.with_columns(pl.Series("kind", kcol), pl.lit(inc).alias("incident"))


def probe_table(ds: str, role: str, limit: int | None = None) -> pl.DataFrame:
    from ..scoring import driver as D
    incs = D.list_incidents(ds, role)
    if limit:
        incs = incs[:limit]
    parts = [probe_units(ds, i) for i in incs]
    parts = [p for p in parts if p.height]
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def probe_auc(df: pl.DataFrame, target: str, features: Sequence[str], *, folds: int = 5, params: dict | None = None,
              rounds: int = 200, B: int = 200, seed: int = 0) -> dict[str, Any]:
    """Incident-grouped K-fold LightGBM probe; pooled out-of-fold AUC with an incident-cluster bootstrap CI.
    target 'i' = campaign (main) vs decoy (D1/D2/D3) units; 'ii' = D3 look-alike vs native background units."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from .gates import auc
    if target == "i":
        d = df.filter(pl.col("kind").is_in(["main", "D1", "D2", "D3"]))
        y = (d["kind"] == "main").to_numpy().astype(int)
    elif target == "ii":
        d = df.filter(pl.col("kind").is_in(["D3", "background"]))
        y = (d["kind"] == "D3").to_numpy().astype(int)
    elif target == "i_attack":          # exploratory: campaign vs ATTACK decoys only (D1, D2; D3 is benign traffic)
        d = df.filter(pl.col("kind").is_in(["main", "D1", "D2"]))
        y = (d["kind"] == "main").to_numpy().astype(int)
    elif target == "ii_typematched":    # exploratory: D3 vs native background units of the same unit types
        types = set(df.filter(pl.col("kind") == "D3")["type"].drop_nulls().to_list())
        d = df.filter(pl.col("kind").is_in(["D3", "background"]) & pl.col("type").is_in(list(types)))
        y = (d["kind"] == "D3").to_numpy().astype(int)
    else:
        raise ValueError(target)
    feats = [f for f in features if f in d.columns]
    X_ = d.select(feats).to_numpy().astype(np.float64)
    g = d["incident"].to_numpy()
    out: dict[str, Any] = {"target": target, "n": int(y.size), "n_pos": int(y.sum()), "features": feats,
                           "n_incidents": int(np.unique(g).size)}
    if y.size < 20 or y.sum() < 5 or (y.size - y.sum()) < 5:
        out.update(auc=math.nan, lo=math.nan, hi=math.nan)
        return out
    prm = dict(params or {})
    prm.setdefault("objective", "binary")
    prm.setdefault("verbose", -1)
    cats = [i for i, f in enumerate(feats) if f in PROBE_CATEGORICAL]
    oof = np.full(y.size, np.nan)
    gains = np.zeros(len(feats))
    nf = min(folds, int(np.unique(g).size))
    for tr, te in GroupKFold(n_splits=nf).split(X_, y, g):
        if y[tr].min() == y[tr].max():
            continue
        ds_ = lgb.Dataset(X_[tr], y[tr], categorical_feature=cats or "auto", free_raw_data=True)
        bst = lgb.train(prm, ds_, num_boost_round=rounds)
        oof[te] = bst.predict(X_[te])
        gains += bst.feature_importance("gain")
    ok = np.isfinite(oof)
    a = auc(oof[ok & (y == 1)], oof[ok & (y == 0)])
    rng = np.random.default_rng((seed, 15))
    inc_u = np.unique(g)
    idx_of = {i: np.flatnonzero(g == i) for i in inc_u}
    reps = []
    for _ in range(B):
        pick = rng.choice(inc_u, size=inc_u.size, replace=True)
        ii = np.concatenate([idx_of[i] for i in pick])
        ii = ii[ok[ii]]
        reps.append(auc(oof[ii][y[ii] == 1], oof[ii][y[ii] == 0]))
    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    imp = sorted(zip(feats, gains.tolist()), key=lambda t: -t[1])
    out.update(auc=float(a), lo=float(np.quantile(reps, 0.025)) if reps.size else math.nan,
               hi=float(np.quantile(reps, 0.975)) if reps.size else math.nan,
               top_features=[(f, round(v / max(gains.sum(), 1e-12), 4)) for f, v in imp[:6]])
    return out


# =========================================================================== fixed-claim cells (E2, E11, E15)
def run_fixed_cell(exp: str, ds: str, name: str, alpha: float, *, K: int, n_test: int | None,
                   seeds: Iterable[int] = range(N_SPLITS), splits: Splits | None = None, split_tag: str | None = None,
                   sid_bank: str | None = None, overwrite: bool = False) -> tuple[pl.DataFrame, dict, Path]:
    """A split-independent claim set (B1 / B2 / B4 / B5 or the r_C-limited 'oracle') evaluated on the SAME
    test incidents as the calibrated methods (the calibration part of each split is unused)."""
    sid_bank = sid_bank or scorer_id(ds, "stat")
    bank = load_bank(ds, sid_bank)
    seeds = np.asarray(list(seeds) if splits is None else splits.seeds, dtype=np.int64)
    sp = splits if splits is not None else random_splits(bank.n, K, seeds, n_test)
    src = None
    if name == "oracle":
        cnt = oracle_counts(bank)
    else:
        cnt = baseline_counts(bank, ds, name)
        src = str((baseline_dir(ds, name) / "summary.json").read_text()) if (baseline_dir(ds, name) / "summary.json").exists() else None
    cfg = {"exp": exp, "ds": ds, "scorer_id": "fixed", "method": name, "alpha": float(alpha), "K": int(K),
           "n_test": n_test, "seeds": [int(seeds[0]), int(seeds[-1]), int(seeds.size)], "split_tag": split_tag,
           "bank_for_labels": sid_bank, "prereg": PREREG_BLAKE3, "code": CODE_VERSION}
    h = cfg_hash(cfg, bank_hash(bank), blake3.blake3((src or "").encode()).hexdigest())
    path = cell_dir(exp, ds, name, "fixed", h)
    if (path / "cell.json").exists() and not overwrite:
        df, meta = read_cell(path)
        return df, meta, path
    res = fixed_metrics(bank, cnt, sp, alpha)
    res["n_incidents_with_output"] = np.full(sp.S, float(np.isfinite(cnt["R"]).sum()))
    df = frame_from(res, name, "fixed", alpha, sp.seeds, {"ds": ds, "K": int(K)})
    meta = {"bank_input_hash": [bank_hash(bank)], "baseline_summary": json.loads(src) if src else None,
            "prereg_blake3": PREREG_BLAKE3}
    write_cell(path, df, cfg, meta)
    return df, {"config": cfg, **meta, "summary": summarize_frame(df)}, path


# =========================================================================== E3 paired tests
def holm_bootstrap(cells: dict[str, np.ndarray], level: float = 0.05) -> dict[str, dict]:
    """Holm over one-sided bootstrap tests of H0: Delta <= 0. ``cells``: name -> replicate array of Delta.
    p_j = (1 + #{Delta* <= 0}) / (B + 1); Holm step-down; the Holm-adjusted lower confidence bound of cell j is
    the (level / (m - rank_j + 1))-quantile of its replicates for the cells reached by the step-down, and the
    level/m quantile otherwise. Also the normal-theory version (z = mean / SE) for reference."""
    names = list(cells)
    m = len(names)
    reps = {n: np.asarray(cells[n], float)[np.isfinite(cells[n])] for n in names}
    p = np.array([(1 + np.sum(reps[n] <= 0)) / (1 + reps[n].size) for n in names])
    adj, rej = holm(p, level)
    order = np.argsort(p, kind="stable")
    out = {}
    stop = False
    for rank, j in enumerate(order):
        n = names[j]
        lev = level / (m - rank) if not stop else level / m
        r = reps[n]
        lcb = float(np.quantile(r, lev)) if r.size else math.nan
        se = float(r.std(ddof=1)) if r.size > 1 else math.nan
        out[n] = {"p_boot": float(p[j]), "p_holm": float(adj[j]), "reject": bool(rej[j]), "holm_level": lev,
                  "lcb_holm": lcb, "se": se, "lcb_holm_normal": float(np.mean(r) - norm.ppf(1 - lev) * se) if r.size > 1 else math.nan}
        if not rej[j]:
            stop = True
    return out


# =========================================================================== E4 test-only manipulation bound
def r3_bound_terms(bank_inj: LossBank, sel: Selection, q: int, alpha: float) -> np.ndarray:
    """Remark R3 (score-threshold family): E[FDP'] <= alpha + E[min(1, q / max(|S'|, 1))]; per split the
    mean over test incidents of min(1, q_i / max(R'_i, 1)) with q_i = min(q, #nulls of the incident)."""
    n0 = bank_inj.col("n_null0")[sel.test_rows]
    qi = np.minimum(float(q), n0)
    term = np.minimum(1.0, qi / np.maximum(sel.R.astype(float), 1.0))
    return alpha + masked_mean(term, sel.tmask)


# =========================================================================== E5 Example 1
def example1_curve(L_values: Sequence[int], *, n: int = 20, K: int = 150, alpha: float = 0.1,
                   n_incidents: int | None = None, splits: int = 5, seed: int = 0) -> pl.DataFrame:
    """P2-C's Example 1 construction (eval/synth.example1_incident) for each chain length L: naive composition
    of an edge-valid CRC set vs composite CRC vs chain-RT; theory: f = example1_f, naive maximal-chain FDR =
    f / n, edge FDR = f / (nL + f), asymptote min(1, alpha L / (1 - alpha))."""
    from ..certify.losses import build_loss_tables
    from ..certify.pvalues import sort_null_pool
    from . import synth
    s0 = sort_null_pool(np.random.default_rng(seed).standard_normal(10000))
    rows = []
    for L in L_values:
        f = synth.example1_f(int(L), n, alpha, K)
        incs = [synth.example1_incident(s0, L=int(L), n=n, f=f, seed=seed, incident_id=f"e{i}")
                for i in range((n_incidents or K + 50))]
        bank = build_loss_tables(incs, s0)
        sp = random_splits(bank.n, K, range(splits), None)
        row = {"L": int(L), "f": int(f), "alpha": alpha, "theory_naive_maxchain_fdr": f / n,
               "theory_edge_fdr": f / (n * L + f), "asymptote": min(1.0, alpha * L / (1 - alpha))}
        for m in ("chain_naive", "chain_composite", "chain_rt"):
            res = RN.evaluate(bank, m, sp, alpha, cert_gap=False)
            for k in ("fdr", "maxchain_fdr", "prefix_fdr", "chain_fwer", "ccr_mean", "n_claims"):
                row[f"{m}|{k}"] = float(np.nanmean(res[k])) if k in res else math.nan
        rows.append(row)
    return pl.DataFrame(rows)


# =========================================================================== E6 streaming (SHOULD; DESIGN §3.7)
def stream_incidents(ds: str, role: str, scorer: str, limit: int | None = None):
    """StreamIncident list for P2-C's e-LOND bank: prefix-only scores (training: OOF from
    models/<scorer>/oof_prefix.parquet; other roles: results/scores/<ds>/<scorer>.stat_prefix), labels, and the
    look index of each candidate from its v unit's start relative to the case trigger t_0 (meta.json)."""
    from ..certify.elond import StreamIncident, look_index
    from ..scoring import driver as D
    incs = D.list_incidents(ds, role)
    if limit:
        incs = incs[:limit]
    oof = None
    if role in ("train",):
        oof = pl.read_parquet(results_root() / "models" / scorer / "oof_prefix.parquet")
        oof = {k: v for k, v in oof.group_by("inc_id")}
    out = []
    for inc in incs:
        cands = D.load_cands(ds, inc)
        if cands.height == 0:
            continue
        lab = D.load_labels(ds, inc)
        if oof is not None:
            g = oof.get((f"{ds}:{inc}",))
            if g is None:
                continue
            m = dict(zip(g["cand_id"].to_list(), g["s_oof"].to_list()))
            s = np.array([m.get(int(c), np.nan) for c in cands["cand_id"].to_list()], dtype=np.float64)
        else:
            sc = D.read_scores(ds, f"{scorer}.stat_prefix", inc)
            m = dict(zip(sc["cand_id"].to_list(), sc["s"].to_list()))
            s = np.array([m.get(int(c), np.nan) for c in cands["cand_id"].to_list()], dtype=np.float64)
        units = D.load_units(ds, inc)
        uts = dict(zip(units["unit_id"].to_list(), units["ts"].to_list()))
        vstart = np.array([uts[int(v)] for v in cands["v"].to_list()], dtype=np.float64)
        meta = json.loads((_incident_dir(ds, inc) / "meta.json").read_text())
        look = look_index(vstart, float(meta["t0"]))
        ok = np.isfinite(s)
        out.append(StreamIncident(f"{ds}:{inc}", s[ok], lab["y"].to_numpy().astype(np.int8)[ok], look[ok]))
    return out


# =========================================================================== E10 transfer / few-shot (SHOULD; DESIGN §3.8)
def cross_select(bank_cal: LossBank, cal_rows: np.ndarray, bank_test: LossBank, test_rows: np.ndarray,
                 method: str, alpha: float) -> Selection:
    """Transfer: lambda-hat from calibration incidents of ``bank_cal`` (rows [S, K]), selection sizes on test
    incidents of ``bank_test`` (rows [S, M]); both banks share the scorer's null pool and the Lambda grid."""
    spec = RN.COLUMN_METHODS[method]
    cal_rows = np.asarray(cal_rows, dtype=np.int64)
    test_rows = np.asarray(test_rows, dtype=np.int64)
    cmask = np.ones(cal_rows.shape, bool)
    tmask = np.ones(test_rows.shape, bool)
    col = RN.choose_columns(bank_cal, spec, cal_rows, cmask, alpha)
    R = bank_test.t("R", spec["family"])[test_rows, col[:, None]].astype(np.int64)
    return Selection(cal_rows, cmask, test_rows, tmask, R, col, spec["family"])


def phi_matrix(ds: str, role: str | None = None, incs: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """phi(I) (features/descriptors.descriptor, 42-d, label-free) for the incidents of a pool."""
    from ..features.descriptors import descriptor
    from ..scoring import driver as D
    incs = list(incs) if incs is not None else D.list_incidents(ds, role)
    rows, ids = [], []
    for inc in incs:
        u = D.load_units(ds, inc)
        c = D.load_cands(ds, inc)
        rows.append(np.asarray(descriptor(u, c, D.load_side(ds, inc)), dtype=np.float64))
        ids.append(inc)
    return np.vstack(rows) if rows else np.zeros((0, 42)), ids


def two_sample_auc(Xa: np.ndarray, Xb: np.ndarray, folds: int = 5, seed: int = 0) -> dict[str, float]:
    """Cross-validated logistic two-sample classifier (bank = 0 vs deployment = 1) on phi; pooled OOF AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .gates import auc
    X_ = np.vstack([Xa, Xb])
    X_ = np.where(np.isfinite(X_), X_, 0.0)
    y = np.r_[np.zeros(len(Xa), int), np.ones(len(Xb), int)]
    oof = np.zeros(y.size)
    nf = int(min(folds, min(len(Xa), len(Xb))))
    for tr, te in StratifiedKFold(nf, shuffle=True, random_state=seed).split(X_, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        m.fit(X_[tr], y[tr])
        oof[te] = m.predict_proba(X_[te])[:, 1]
    return {"auc": float(auc(oof[y == 1], oof[y == 0])), "n_bank": int(len(Xa)), "n_deploy": int(len(Xb)),
            "folds": nf}


# =========================================================================== Gotham default-rule labels (E11 side by side)
def gotham_default_labels(inc: str) -> pl.DataFrame:
    """Pair labels of a Gotham-real incident under the DEFAULT unit->episode rule (P2-B gt.gotham_episodes with
    seed_priority=False; the banks use the seed-priority rule). Reads the incident's quarantined label table via the
    GT module (post-PREREG evaluation only)."""
    import yaml
    from ..gt import semantics as SEM
    from ..gt.gotham_episodes import GothamRoles, gotham_gt
    from ..scoring import driver as D
    from ..units.scope import ScopeResult
    D.check_label_access("gotham_real", inc)
    cfg = yaml.safe_load((P2_ROOT / "configs" / "units.yaml").read_text())
    roles = GothamRoles.from_config((cfg.get("gotham") or {}).get("roles"))
    idir = _incident_dir("gotham_real", inc)
    meta = json.loads((idir / "meta.json").read_text())
    recs = pl.read_parquet(idir / "records.parquet")
    units = D.load_units("gotham_real", inc)
    cands = D._raw_cands("gotham_real", inc) if hasattr(D, "_raw_cands") else D.load_cands("gotham_real", inc)
    scope = ScopeResult.from_json(D.load_scope("gotham_real", inc))
    seeds = [meta["seed_device"]] if isinstance(meta.get("seed_device"), str) else list(meta.get("seed_device") or [])
    gt_d = gotham_gt(recs, units, meta["session"], [str(idir / "labels.parquet")], roles, seeds=seeds,
                     seed_priority=False)
    return SEM.label_pairs(cands, units, gt_d, scope=scope)
