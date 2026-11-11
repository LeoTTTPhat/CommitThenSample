"""Split runner on cached loss tables (DESIGN §6.1-6.4).

``run_splits(loss_tables, method, K, seeds)`` evaluates one or more methods on S random
calibration/test partitions of the pool and returns one row per (method, split) with every
§6.2 metric; with ``out_dir`` it also writes per_split.parquet, splits.jsonl (one JSON record per
split and method), summary.parquet and run_manifest.json, atomically and resumably (a rerun with
the same config hash and bank input hash just reads the results back).

Methods
-------
Loss-table ("column") methods pick one grid index per split from calibration-row sums, fully
vectorised across splits (S x K_all membership matrix @ table):
  crc_bh            CRC on L~ = running max of strict FDP, BH family          (Thm 1)
  crc_bh_raw        same formula on the UN-monotonised FDP (ablation, no guarantee)
  crc_fdx020/030    CRC on 1{L~ > gamma}                                       (FDX)
  ltt               Hoeffding-Bentkus + fixed sequence on raw FDP, delta        (PAC)
  crc_st/_raw       score-threshold family (ablation)
  chain_composite   CRC on L~cmp (Alg. 5; Cor. 2)
  chain_naive       edge-CRC lambda-hat, chains read off S_e(lambda-hat)
  chain_rt          coherent-factuality style on the score-threshold family (FWER-type)
  chain_rt_bh       the same on the BH family
Raw-score ("R") methods compute selection sizes per test incident:
  bh_ref (plain BH at alpha on reference p), by_ref, bh_iid, bh_iid_pooled, b7 (needs
  aux['b7'] = fitted IsotonicB7), b8, lr_a1 / lr_a1b (single draw), lr_a1avg / lr_a1avgb (Eq. 12
  average of r = lr_r draws), lr_a2 / lr_a2b (hierarchical); 'b' = boosted (U-eBH).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Sequence

import blake3
import numpy as np
import polars as pl

from ..baselines.bh_iid import NullIndex, bh_iid_R
from ..baselines.by_ref import by_ref_R
from ..baselines.lee_ren import lee_ren_R
from ..baselines.naive import bh_ref_plain_R
from ..certify.chains import rt_r_index
from ..certify.crc import crc_threshold_from_sums
from ..certify.losses import CHAIN_KEYS, LossBank
from ..certify.ltt import ltt_threshold_from_sums
from ..scoring.calibrate import b7_R, b8_R
from .metrics import certification_gap, chain_metrics, masked_mean, split_metrics, summarize
from .splits import Splits, random_splits

COLUMN_METHODS: dict[str, dict] = {
    "crc_bh": {"family": "bh", "table": "Lmono", "calib": "crc"},
    "crc_bh_raw": {"family": "bh", "table": "fdp_strict", "calib": "crc"},
    "crc_fdx020": {"family": "bh", "table": "fdx020", "calib": "crc"},
    "crc_fdx030": {"family": "bh", "table": "fdx030", "calib": "crc"},
    "ltt": {"family": "bh", "table": "fdp_strict", "calib": "ltt"},
    "crc_st": {"family": "st", "table": "Lmono", "calib": "crc"},
    "crc_st_raw": {"family": "st", "table": "fdp_strict", "calib": "crc"},
    "chain_composite": {"family": "bh", "table": "Lcmp", "calib": "crc", "chain": True},
    "chain_naive": {"family": "bh", "table": "Lmono", "calib": "crc", "chain": True},
    "chain_rt": {"family": "st", "table": "fwer", "calib": "rt", "chain": True},
    "chain_rt_bh": {"family": "bh", "table": "fwer", "calib": "rt", "chain": True},
}
LR_METHODS = {"lr_a1": ("A1", False, 1), "lr_a1b": ("A1", True, 1), "lr_a1avg": ("A1", False, None),
              "lr_a1avgb": ("A1", True, None), "lr_a2": ("A2", False, 1), "lr_a2b": ("A2", True, 1)}
R_METHODS = {"bh_ref", "by_ref", "bh_iid", "bh_iid_pooled", "b7", "b8"} | set(LR_METHODS)
GUARANTEE = {"crc_bh": "per-incident FDR", "crc_fdx020": "FDX(0.2)", "crc_fdx030": "FDX(0.3)",
             "ltt": "PAC (1-delta)", "crc_st": "per-incident FDR", "chain_composite": "edge+maxchain FDR",
             "chain_rt": "per-incident all-chains-correct (FWER)", "chain_rt_bh": "per-incident FWER",
             "bh_iid": "none (unit exchangeability)", "bh_iid_pooled": "pooled FDR (unit exchangeability)",
             "lr_a1": "pooled FDR", "lr_a1b": "pooled FDR", "lr_a1avg": "pooled FDR", "lr_a1avgb": "pooled FDR",
             "lr_a2": "pooled FDR", "lr_a2b": "pooled FDR"}
DEFAULT_METHODS = ("crc_bh", "crc_bh_raw", "crc_fdx020", "crc_fdx030", "ltt", "crc_st", "bh_ref", "by_ref",
                   "bh_iid", "bh_iid_pooled", "lr_a2", "lr_a2b", "lr_a1", "lr_a1b", "lr_a1avg", "lr_a1avgb")
META_COLS = ("n_true", "n_trans", "n_trans_obs", "n_trans_hid", "n_elink", "n_bridges", "n_noncamp_dev",
             "n_gt_paths")
STRATA_DEFAULT = ("decoy_rich", "benign_only", "short_chain", "N_tercile")


# --------------------------------------------------------------------------- helpers
def _rows(pool_rows: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = pos >= 0
    rows = np.where(mask, pool_rows[np.where(mask, pos, 0)], 0)
    return rows, mask


def _membership(cal_rows: np.ndarray, cal_mask: np.ndarray, n: int) -> np.ndarray:
    S = cal_rows.shape[0]
    W = np.zeros((S, n))
    sidx = np.repeat(np.arange(S), cal_rows.shape[1])
    m = cal_mask.ravel()
    np.add.at(W, (sidx[m], cal_rows.ravel()[m]), 1.0)
    return W


def choose_columns(bank: LossBank, spec: dict, cal_rows: np.ndarray, cal_mask: np.ndarray, alpha: float,
                   delta: float = 0.1, B: float = 1.0) -> np.ndarray:
    """Grid index per split for a column method."""
    fam, tab, calib = spec["family"], spec["table"], spec["calib"]
    K = cal_mask.sum(axis=1)
    if calib == "rt":
        jr = rt_r_index(bank.t(tab, fam))
        out = np.zeros(cal_rows.shape[0], dtype=np.int64)
        for s in range(cal_rows.shape[0]):
            j = np.sort(jr[cal_rows[s][cal_mask[s]]])
            q = int(np.floor(alpha * (j.size + 1) + 1e-12))
            out[s] = int(j[min(q, j.size) - 1]) if q >= 1 and j.size else 0
        return out
    W = _membership(cal_rows, cal_mask, bank.n)
    sums = W @ bank.t(tab, fam)
    if calib == "crc":
        return crc_threshold_from_sums(sums, K, alpha, B)
    if calib == "ltt":
        return ltt_threshold_from_sums(sums, K, alpha, delta)
    raise ValueError(calib)


def _strata_arrays(bank: LossBank, pool_rows: np.ndarray, strata: Sequence[str]) -> dict[str, np.ndarray]:
    """Label-free or GT-derived categorical strata (bool / small-integer columns, N tercile)."""
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


def evaluate(bank: LossBank, method: str, splits: Splits, alpha: float, *, pool_rows=None, aux=None,
             delta: float = 0.1, lr_r: int = 100, gammas=(0.2, 0.3), strata: Sequence[str] = STRATA_DEFAULT,
             cert_gap: bool | None = None, cache: dict | None = None) -> dict[str, np.ndarray]:
    """Per-split metrics [S] for one method. ``pool_rows`` maps split positions to bank rows."""
    aux = aux or {}
    cache = {} if cache is None else cache
    pool_rows = np.arange(bank.n) if pool_rows is None else np.asarray(pool_rows, dtype=np.int64)
    cal_rows, cal_mask = _rows(pool_rows, splits.cal)
    test_rows, tmask = _rows(pool_rows, splits.test)
    S = splits.S
    col = None
    if method in COLUMN_METHODS:
        spec = COLUMN_METHODS[method]
        if spec.get("chain") and not bank.has_chains:
            raise ValueError(f"{method} needs chain tables (endpoints) in the bank")
        col = choose_columns(bank, spec, cal_rows, cal_mask, alpha, delta)
        R = bank.t("R", spec["family"])[test_rows, col[:, None]].astype(np.int64)
    elif method in ("bh_ref", "by_ref", "b7"):
        key = (method, alpha)
        if key not in cache:
            if method == "bh_ref":
                cache[key] = bh_ref_plain_R(bank, alpha)
            elif method == "by_ref":
                cache[key] = by_ref_R(bank, alpha)
            else:
                if "b7" not in aux:
                    raise ValueError("b7 needs aux['b7'] = IsotonicB7 fitted on OOF training scores")
                cache[key] = b7_R(bank, aux["b7"])
        R = cache[key][test_rows]
    elif method in R_METHODS:
        R = np.zeros(test_rows.shape, dtype=np.int64)
        if method.startswith("bh_iid") and "null_index" not in cache:
            cache["null_index"] = NullIndex(bank)
        for s in range(S):
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
                variant, boost, rr = LR_METHODS[method]
                r = lee_ren_R(bank, cr, tr, alpha, variant, boost, r=lr_r if rr is None else rr, seed=seed)
            R[s, : r.size] = r       # test rows of a split are contiguous at the front (padding at the end)
    else:
        raise ValueError(f"unknown method {method!r}")
    R = np.where(tmask, R, 0)
    counts = bank.counts(test_rows, R)
    meta = {k: bank.col(k)[test_rows] for k in META_COLS}
    meta["benign_only"] = bank.col("benign_only")[test_rows]
    out = split_metrics(counts, meta, tmask, alpha, gammas)
    out["K"] = cal_mask.sum(axis=1).astype(float)
    out["M"] = tmask.sum(axis=1).astype(float)
    fs = counts["V"] / np.maximum(counts["R"], 1)
    for sname, arr in _strata_arrays(bank, pool_rows, strata).items():
        st = arr[test_rows]
        for v in np.unique(st[tmask]):
            out[f"fdr|{sname}={int(v)}"] = masked_mean(fs, tmask & (st == v))
    if col is not None:
        spec = COLUMN_METHODS[method]
        fam = spec["family"]
        out["lam_idx"] = col.astype(float)
        out["lam"] = bank.grids[fam][col]
        out["abstain"] = (col == 0).astype(float)
        if bank.has_chains and (spec.get("chain") or method in ("crc_bh", "crc_st")):
            ch = {k: bank.t(k, fam)[test_rows, col[:, None]] for k in CHAIN_KEYS}
            out.update(chain_metrics(ch, meta["n_gt_paths"], tmask))
        if cert_gap if cert_gap is not None else method == "crc_bh":
            den = (meta["n_elink"] + meta["n_trans_hid"])[..., None]
            ftab = bank.t("fdp_strict", fam)[test_rows]
            with np.errstate(invalid="ignore", divide="ignore"):
                rtab = np.where(den > 0, bank.t("TP", fam)[test_rows] / np.where(den > 0, den, 1.0), np.nan)
            out["cert_gap_cer_link"] = certification_gap(ftab, rtab, col, tmask)
    return out


# --------------------------------------------------------------------------- driver
def _config_hash(cfg: dict, bank: LossBank) -> str:
    h = blake3.blake3()
    h.update(json.dumps(cfg, sort_keys=True, default=str).encode())
    h.update(str(bank.meta.get("input_hash")).encode())
    return h.hexdigest()


def run_splits(loss_tables, method, K: int, seeds: Iterable[int] = range(200), *, alpha: float = 0.1,
               n_test: int | None = None, splits: Splits | None = None, out_dir: Path | str | None = None,
               aux: dict | None = None, delta: float = 0.1, lr_r: int = 100, salt: int = 0,
               overwrite: bool = False, tag: str = "") -> pl.DataFrame:
    """Evaluate method(s) over splits; one row per (method, split). See module docstring."""
    bank = loss_tables if isinstance(loss_tables, LossBank) else LossBank.load(loss_tables)
    methods = [method] if isinstance(method, str) else list(method)
    seeds = list(seeds)
    sp = splits if splits is not None else random_splits(bank.n, K, seeds, n_test, salt)
    cfg = {"methods": methods, "K": K, "seeds": [int(s) for s in sp.seeds], "alpha": alpha, "n_test": n_test,
           "delta": delta, "lr_r": lr_r, "salt": salt, "custom_splits": splits is not None, "tag": tag,
           "aux": sorted((aux or {}).keys())}
    chash = _config_hash(cfg, bank)
    if out_dir is not None:
        out_dir = Path(out_dir)
        man = out_dir / "run_manifest.json"
        if man.exists() and not overwrite and json.loads(man.read_text()).get("config_hash") == chash:
            return pl.read_parquet(out_dir / "per_split.parquet")
    cache: dict = {}
    rows = []
    timings = {}
    for m in methods:
        t0 = time.perf_counter()
        res = evaluate(bank, m, sp, alpha, aux=aux, delta=delta, lr_r=lr_r, cache=cache)
        timings[m] = time.perf_counter() - t0
        for i in range(sp.S):
            row = {"method": m, "seed": int(sp.seeds[i]), "alpha": float(alpha), "guarantee": GUARANTEE.get(m, "none")}
            row.update({k: float(v[i]) for k, v in res.items()})
            rows.append(row)
    df = pl.DataFrame(rows, infer_schema_length=None)
    if out_dir is not None:
        _write_outputs(out_dir, df, cfg, chash, timings)
    return df


def summary_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per method: mean / sd / naive se over splits of every numeric metric, plus split-distribution
    quantiles of the realised FDR (training-conditional view)."""
    out = []
    num = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric() and c not in ("seed",)]
    for m in df["method"].unique(maintain_order=True).to_list():
        sub = df.filter(pl.col("method") == m)
        row = {"method": m}
        for c in num:
            s = summarize(sub[c].to_numpy())
            row[f"{c}_mean"], row[f"{c}_sd"], row[f"{c}_se"] = s["mean"], s["sd"], s["se"]
        f = sub["fdr"].to_numpy()
        for q in (0.05, 0.5, 0.95):
            row[f"fdr_q{int(q * 100):02d}"] = float(np.quantile(f, q)) if f.size else float("nan")
        out.append(row)
    return pl.DataFrame(out, infer_schema_length=None)


def _write_outputs(out_dir: Path, df: pl.DataFrame, cfg: dict, chash: str, timings: dict) -> None:
    tmp = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.mkdir(parents=True)
    df.write_parquet(tmp / "per_split.parquet")
    with open(tmp / "splits.jsonl", "w") as f:
        for r in df.iter_rows(named=True):
            f.write(json.dumps(r, default=float, allow_nan=True) + "\n")
    summary_table(df).write_parquet(tmp / "summary.parquet")
    (tmp / "run_manifest.json").write_text(json.dumps({"config": cfg, "config_hash": chash,
                                                       "seconds": timings, "created_unix": time.time()},
                                                      indent=1, default=str))
    if out_dir.exists():
        old = out_dir.with_name(f".{out_dir.name}.old-{os.getpid()}-{time.time_ns()}")
        os.replace(out_dir, old)
        os.replace(tmp, out_dir)
        shutil.rmtree(old)
    else:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, out_dir)
