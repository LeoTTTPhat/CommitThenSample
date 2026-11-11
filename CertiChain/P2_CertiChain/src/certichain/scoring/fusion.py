"""LightGBM scorers, incident-grouped cross-fitting and the reference null pool S_0 (DESIGN §3.4.2).

Models: M_stat (FEATURES), M_stat^prefix (PREFIX_FEATURES; SHOULD, E6) and M_fuse (FEATURES +
``s_llm`` + ``llm_called``; s_llm is NaN outside the cascade band). Hyper-parameters are fixed
(DESIGN §3.4.2 / §3.10): binary objective, 63 leaves, lr 0.05, <= 1000 rounds with 50-round early
stopping on an incident-grouped validation fold, min_child_samples 50, feature_fraction 0.8,
bagging 0.8 / 1, lambda_l2 1, scale_pos_weight 20, seed 0 (M_fuse: seeds 0-4).

Cross-fitting (``cross_fit``) on the training incidents, 5 incident-grouped folds:
  for each outer fold f: validation fold = (f + 1) mod 5 (early stopping), training = the other 3;
    prior features (transition prior, lag-LLR) of rows outside f are themselves out-of-fold over the
    remaining folds, rows of f get priors fitted on all incidents outside f;
    OOF score of every row of f = raw margin (log-odds) of the early-stopped fold model.
  final model: all training incidents (prior features 5-fold OOF), rounds = round(mean best iteration);
  final priors (for scoring dev/cal/test) = fitted on all training incidents.
  S_0 = sorted OOF scores of the training STRICT nulls (y != 1: null and AMB, as in the strict
  calibration loss). S_0 is a fixed function of training data (A2 preserved).
Training target = 1{y = 1}; AMB rows are negatives (the strict loss counts them as false), priors
exclude AMB (features/priors.py). Scores are raw margins (monotone, no saturation ties).
Labels enter only as arguments; this module never reads label files.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import polars as pl

from certichain.features import stats as F
from certichain.features.priors import PairPriors

LGB_PARAMS: dict = {
    "objective": "binary", "num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "scale_pos_weight": 20.0, "seed": 0, "num_threads": 16, "verbose": -1, "deterministic": True,
    "force_row_wise": True, "metric": "auc", "first_metric_only": True,
}
MAX_ROUNDS = 1000
EARLY_STOP = 50
N_FOLDS = 5
FUSE_EXTRA = ("s_llm", "llm_called")


@dataclass
class IncidentData:
    """One incident: label-free base features (stats.base_features, sorted by cand_id), labels for
    training incidents, and (M_fuse) the LLM score + band flag per candidate."""
    inc_id: str
    feats: pl.DataFrame
    y: np.ndarray | None = None
    s_llm: np.ndarray | None = None          # NaN outside the band / not scored
    in_band: np.ndarray | None = None        # bool: candidate is in the cascade band (llm_called)

    @property
    def n(self) -> int:
        return int(self.feats.height)


def make_folds(inc_ids: Sequence[str], n_folds: int = N_FOLDS, seed: int = 0) -> dict[str, int]:
    """Deterministic incident -> fold map (sorted ids, permuted by default_rng(seed), round robin)."""
    ids = sorted(set(inc_ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {ids[j]: int(i % n_folds) for i, j in enumerate(perm)}


def _rank_llr(inc_codes: np.ndarray, v: np.ndarray, llr: np.ndarray) -> np.ndarray:
    df = pl.DataFrame({"g": inc_codes, "v": v, "l": llr})
    return df.select(pl.col("l").rank("min", descending=True).over(["g", "v"]))["l"].to_numpy().astype(np.float64)


def _extra_cols(incs: Sequence[IncidentData]) -> np.ndarray:
    s = np.concatenate([(i.s_llm if i.s_llm is not None else np.full(i.n, np.nan)) for i in incs])
    b = np.concatenate([(i.in_band if i.in_band is not None else np.zeros(i.n, bool)) for i in incs])
    s = np.where(b, s, np.nan)
    return np.column_stack([s.astype(np.float32), b.astype(np.float32)])


@dataclass
class Stacked:
    """Row-stacked training incidents."""
    big: pl.DataFrame
    inc_ids: list[str]
    sizes: np.ndarray
    groups: np.ndarray            # inc id per row
    gcode: np.ndarray             # incident index per row
    y: np.ndarray
    extra: np.ndarray | None

    @classmethod
    def build(cls, incs: Sequence[IncidentData], with_extra: bool) -> "Stacked":
        sizes = np.array([i.n for i in incs], dtype=np.int64)
        big = pl.concat([i.feats for i in incs], how="vertical") if incs else pl.DataFrame()
        ids = [i.inc_id for i in incs]
        return cls(big=big, inc_ids=ids, sizes=sizes, groups=np.repeat(np.array(ids, dtype=object), sizes),
                   gcode=np.repeat(np.arange(len(incs)), sizes),
                   y=np.concatenate([np.asarray(i.y) for i in incs]).astype(np.int8) if incs else np.zeros(0, np.int8),
                   extra=_extra_cols(incs) if with_extra else None)


def _matrix(st: Stacked, rows: np.ndarray, tp: np.ndarray, ll: np.ndarray, names: Sequence[str],
            with_extra: bool) -> np.ndarray:
    base = [n for n in names if n not in FUSE_EXTRA]
    sub = st.big.filter(pl.Series(rows)) if rows.dtype == bool else st.big[rows]
    rk = _rank_llr(st.gcode[rows], sub["v"].to_numpy(), ll[rows])
    sub = sub.with_columns(pl.Series("trans_prior", tp[rows]), pl.Series("lag_llr", ll[rows]),
                           pl.Series("rank_llr_u", rk))
    X = F.feature_matrix(sub, base)
    if with_extra:
        X = np.column_stack([X, st.extra[rows]])
    return X


def _prior_cols_oof(st: Stacked, fold_arr: np.ndarray, rows_mask: np.ndarray, W: float) -> tuple[np.ndarray, np.ndarray]:
    """Priors for rows in rows_mask, out-of-fold over the folds present among them."""
    tp = np.full(st.y.size, np.nan)
    ll = np.full(st.y.size, np.nan)
    tu = st.big["type_u_str"].to_numpy()
    tv = st.big["type_v_str"].to_numpy()
    lg = st.big["lag_gap"].to_numpy().astype(np.float64)
    for g in np.unique(fold_arr[rows_mask]):
        tgt = rows_mask & (fold_arr == g)
        src = rows_mask & (fold_arr != g)
        pr = PairPriors.fit(tu[src], tv[src], lg[src], st.y[src], W)
        tp[tgt] = pr.transition_prior(tu[tgt], tv[tgt])
        ll[tgt] = pr.lag_llr(tu[tgt], tv[tgt], lg[tgt])
    return tp, ll


def _train(X, y01, cat, Xv, yv, params, max_rounds, es, n_rounds=None):
    import lightgbm as lgb
    dtr = lgb.Dataset(X, label=y01, categorical_feature=cat, free_raw_data=False)
    if n_rounds is not None:
        b = lgb.train(params, dtr, num_boost_round=int(n_rounds))
        return b, int(n_rounds)
    dva = lgb.Dataset(Xv, label=yv, categorical_feature=cat, reference=dtr, free_raw_data=False)
    b = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(es, first_metric_only=True, verbose=False)])
    bi = int(b.best_iteration or max_rounds)
    return b, bi


@dataclass
class CrossFitResult:
    names: tuple[str, ...]
    inc_ids: list[str]
    folds: dict[str, int]
    oof: np.ndarray                    # OOF raw margin per stacked row
    y: np.ndarray
    groups: np.ndarray
    sizes: np.ndarray
    best_iters: list[int]
    final_rounds: int
    booster: object                    # final lightgbm.Booster
    fold_boosters: list
    priors: PairPriors                 # fitted on all training incidents
    params: dict
    with_extra: bool
    timings: dict = field(default_factory=dict)

    def null_pool(self) -> np.ndarray:
        """S_0: sorted OOF scores of the training strict nulls (y != 1)."""
        return np.sort(self.oof[self.y != 1]).astype(np.float64)

    def oof_frame(self) -> pl.DataFrame:
        cand = np.concatenate([np.arange(n) for n in self.sizes]) if self.sizes.size else np.zeros(0, np.int64)
        return pl.DataFrame({"inc_id": self.groups.astype(str), "row_in_inc": cand.astype(np.int64),
                             "y": self.y.astype(np.int8), "s_oof": self.oof,
                             "fold": np.array([self.folds[g] for g in self.groups], dtype=np.int8)})


def cross_fit(incs: Sequence[IncidentData], names: Sequence[str] = F.FEATURES, *, params: dict | None = None,
              n_folds: int = N_FOLDS, fold_seed: int = 0, max_rounds: int = MAX_ROUNDS, es_rounds: int = EARLY_STOP,
              W: float = 1800.0, log: Callable[[str], None] = print, check: Callable[[], None] = lambda: None
              ) -> CrossFitResult:
    """5-fold incident-grouped cross-fitting + final refit (see module docstring)."""
    params = dict(LGB_PARAMS if params is None else params)
    names = tuple(names)
    with_extra = any(n in FUSE_EXTRA for n in names)
    if with_extra and (names[-len(FUSE_EXTRA):] != FUSE_EXTRA or sum(n in FUSE_EXTRA for n in names) != 2):
        raise ValueError(f"fusion features must end with {FUSE_EXTRA}")
    st = Stacked.build(incs, with_extra)
    folds = make_folds(st.inc_ids, n_folds, fold_seed)
    fold_arr = np.array([folds[g] for g in st.groups], dtype=np.int64)
    y01 = (st.y == 1).astype(np.float32)
    cat = F.categorical_indices([n for n in names if n not in FUSE_EXTRA])
    oof = np.full(st.y.size, np.nan)
    best, boosters, tim = [], [], {}
    t_all = time.perf_counter()
    for f in range(n_folds):
        t0 = time.perf_counter()
        test = fold_arr == f
        val = fold_arr == (f + 1) % n_folds
        T = ~test
        tr = T & ~val
        tp, ll = _prior_cols_oof(st, fold_arr, T, W)            # rows outside f: OOF over T's folds
        tu = st.big["type_u_str"].to_numpy()
        tv = st.big["type_v_str"].to_numpy()
        lg = st.big["lag_gap"].to_numpy().astype(np.float64)
        prT = PairPriors.fit(tu[T], tv[T], lg[T], st.y[T], W)
        tp[test] = prT.transition_prior(tu[test], tv[test])
        ll[test] = prT.lag_llr(tu[test], tv[test], lg[test])
        Xtr, Xva, Xte = (_matrix(st, m, tp, ll, names, with_extra) for m in (tr, val, test))
        b, bi = _train(Xtr, y01[tr], cat, Xva, y01[val], params, max_rounds, es_rounds)
        oof[test] = b.predict(Xte, num_iteration=bi, raw_score=True)
        best.append(bi)
        boosters.append(b)
        tim[f"fold{f}_s"] = round(time.perf_counter() - t0, 2)
        log(f"[fit] fold {f}: train {int(tr.sum())} val {int(val.sum())} test {int(test.sum())} rows, "
            f"best_iter {bi}, {tim[f'fold{f}_s']}s")
        check()
    # final refit on all training incidents
    t0 = time.perf_counter()
    allm = np.ones(st.y.size, dtype=bool)
    tp, ll = _prior_cols_oof(st, fold_arr, allm, W)
    X = _matrix(st, allm, tp, ll, names, with_extra)
    n_final = max(1, int(round(float(np.mean(best)))))
    booster, _ = _train(X, y01, cat, None, None, params, max_rounds, es_rounds, n_rounds=n_final)
    tu = st.big["type_u_str"].to_numpy()
    tv = st.big["type_v_str"].to_numpy()
    lg = st.big["lag_gap"].to_numpy().astype(np.float64)
    priors = PairPriors.fit(tu, tv, lg, st.y, W)
    tim["final_s"] = round(time.perf_counter() - t0, 2)
    tim["total_s"] = round(time.perf_counter() - t_all, 2)
    log(f"[fit] final: {st.y.size} rows, {n_final} rounds, {tim['final_s']}s")
    return CrossFitResult(names=names, inc_ids=st.inc_ids, folds=folds, oof=oof, y=st.y, groups=st.groups,
                          sizes=st.sizes, best_iters=best, final_rounds=n_final, booster=booster,
                          fold_boosters=boosters, priors=priors, params=params, with_extra=with_extra, timings=tim)


# ----------------------------------------------------------------------------- scoring
def score_incident(booster, priors: PairPriors, inc: IncidentData, names: Sequence[str]) -> np.ndarray:
    """Raw margin of a fitted model on one (non-training) incident."""
    if inc.n == 0:
        return np.zeros(0)
    names = tuple(names)
    df = F.add_prior_features(inc.feats, priors)
    X = F.feature_matrix(df, [n for n in names if n not in FUSE_EXTRA])
    if any(n in FUSE_EXTRA for n in names):
        X = np.column_stack([X, _extra_cols([inc])])
    return np.asarray(booster.predict(X, raw_score=True), dtype=np.float64)


def load_booster(path) -> object:
    import lightgbm as lgb
    return lgb.Booster(model_file=str(path))


def within_auc(scores: Sequence[np.ndarray], ys: Sequence[np.ndarray], null: str = "strict") -> dict:
    from certichain.eval.gates import within_incident_auc
    return within_incident_auc(scores, ys, null=null)


def split_by_incident(values: np.ndarray, sizes: np.ndarray) -> list[np.ndarray]:
    cuts = np.cumsum(sizes)[:-1]
    return list(np.split(values, cuts)) if sizes.size else []
