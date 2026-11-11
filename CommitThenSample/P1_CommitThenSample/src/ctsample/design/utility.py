"""Utility models and the bit-exact integer aggregated utility (DESIGN §3.4, Prop. 3(b), Remark 1).

Models (LightGBM, <= 200 trees, num_leaves 31, lr 0.05, min_data_in_leaf 100, feature_fraction 0.8,
class-balanced, compiled to design/treeeval.IntForest):
  g_mal       P(unit malicious | m) on payload units with a known label (U3 = P_mal; B4, B13 rank by it);
  g_q^hit     P(f_q > 0 | m) per estimand q, trained on the q-eligible payload units;
  m_q         E[f_q^2 | f_q > 0, proto3, dport bucket] lookup, shrunk to the global mean with prior weight 50
              (= 1 for indicator estimands);
  U8 table    p̄_q(stratum = proto3 x dport bucket x direction class), no ML, shrunk with prior weight 50;
  Q̃_q         mean per-epoch total of q on the TRAINING LP stream (calibration, utility/04 script).
Class balancing uses explicit weights w1/w0 = n_neg/n_pos; the forests carry the exact prior-shift bias
-log(w1/w0) so that p is a calibrated probability (treeeval.compile_lgbm(bias=...)).

Integer pipeline (gateway = verifier = experiments; Python ints, no overflow anywhere):
    p_iq   = forest_q.predict_int(x_i)  (Q0.16)   if unit i is q-eligible by its METADATA, else 0
    ĝ_iq   = (p_iq · M_q[cell_i]) >> 16           M_q = round(m_q · 2^16)            (Q16.16)
    A_i    = Σ_q (W_q · ĝ_iq << 32) // Q2_q       W_q = round(w_q 2^16) (Σ ≈ 2^16), Q2_q = max(1, round(Q̃_q² 2^32))
    a_fix  = min(2^32 − 1, isqrt(A_i))            = ⌊sqrt(Σ_q w_q ĝ_q / Q̃_q²) · 2^16⌋ up to the per-term floors
    â_i    = a_fix · κ_b^{1[device(i) ∈ boost set]} + 1        (ε_a = 2^-16; utility a = â 2^-16)
  U3: â = p_mal + 1.  U8: p_iq = P8_q[stratum_i] (Q0.16 lookup).  U7 (oracle, headroom only):
  ĝ_iq = f_iq² << 16 (true second moment).  Float twin: a = sqrt(Σ w p m / Q̃²) κ + 2^-16.
Every input is committed (θ: forests, M, P8, W, Q2, feature version, LUT; theta_digest()).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from blake3 import blake3

from ..units import features as F
from . import treeeval as te

__all__ = ["UTILITY_VERSION", "DEFAULT_QUERIES", "LGB_PARAMS", "MIN_POS", "PRIOR_WEIGHT", "N_CELLS", "N_STRATA",
           "meta_eligible", "eligibility_matrix", "cell_index", "stratum_index", "train_forest",
           "second_moment_table", "stratum_table", "UtilityBundle", "ghat_int", "a_fix_int", "utility_int",
           "utility_float", "equal_weights", "q2_fix", "oracle_ghat", "auc", "kind_of", "dedup_rows",
           "a_fix_dedup", "predict_all", "utilities_from_P", "calibrate_lambda_adaptive", "BOOST_DETECTORS",
           "BOOST_QUERIES", "BOOST_FP_MAX", "DETECTOR_ESTIMANDS", "boost_trigger_set"]

UTILITY_VERSION = "ctsample.utility/v1"
LGB_PARAMS = dict(objective="binary", num_leaves=31, learning_rate=0.05, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=1.0, bagging_freq=0, lambda_l2=0.0, max_bin=255, deterministic=True,
                  force_row_wise=True, verbose=-1)
N_TREES = 200
MIN_POS = 30                 # fewer positives in training -> constant (smoothed-rate) model
PRIOR_WEIGHT = 50.0
N_CELLS = 3 * 16             # proto3 x dport bucket
N_STRATA = N_CELLS * 4       # x direction class
TWO16 = 1 << 16
TWO32 = 1 << 32
U32MAX = (1 << 32) - 1

# Utility query class: detector estimands, Suricata families and the malicious-unit label estimand L1.
# (L2-L5 / L_<class> are sub-families of L1 and are evaluated, but not weighted separately in the utility.)
DEFAULT_QUERIES = ("D1o", "D1c", "D2o", "D2c", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11o", "D12", "D13",
                   "S1", "S2", "S3", "S4", "S5", "S6", "L1")
_COUNTS = {"D1c", "D2c", "D5", "D6", "D9"}


def kind_of(q: str) -> str:
    return "count" if q in _COUNTS else "indicator"


# ------------------------------------------------------------------------------------------ #
# metadata eligibility, cells and strata (verifier-recomputable)
# ------------------------------------------------------------------------------------------ #
def meta_eligible(q: str, c: dict) -> np.ndarray:
    """Units that can have f_q > 0, decided from committed metadata only (detector protocol/port rules,
    fps for opening-based estimands; payload units for S and L estimands)."""
    from ..content.detectors import DET_ESTIMANDS, RAW_BY_ID
    pay = c["pay"].astype(bool)
    if q.startswith("S") or q.startswith("L"):
        return pay.copy()
    est = {e.id: e for e in DET_ESTIMANDS}[q]
    r = RAW_BY_ID[est.raw]
    el = pay & np.isin(c["proto"], sorted(r.protos))
    if r.ports is not None:
        ports = sorted(r.ports)
        el &= np.isin(c["port_a"], ports) | np.isin(c["port_b"], ports)
    if est.cls == "indicator_opening":
        el &= c["fps"].astype(bool)
    return el


def eligibility_matrix(queries, c: dict) -> np.ndarray:
    return np.stack([meta_eligible(q, c) for q in queries], axis=1) if len(queries) else np.zeros((len(c["pay"]), 0), bool)


def cell_index(c: dict) -> np.ndarray:
    return F.proto3(c["proto"]) * 16 + F.dport_bucket(c["proto"], c["port_a"], c["port_b"])


def stratum_index(c: dict) -> np.ndarray:
    _, dc = F.side_and_dir(c)
    return cell_index(c) * 4 + dc


# ------------------------------------------------------------------------------------------ #
# training
# ------------------------------------------------------------------------------------------ #
def train_forest(X: np.ndarray, y: np.ndarray, seed: int, threads: int = 8, n_trees: int = N_TREES,
                 min_pos: int = MIN_POS, meta: dict | None = None):
    """Class-balanced LightGBM binary forest -> (IntForest, booster | None, info). Constant model if < min_pos
    positives (smoothed rate (n_pos + 0.5)/(n + 1))."""
    y = np.asarray(y, dtype=np.int8)
    n, n_pos = len(y), int(y.sum())
    info = {"n": n, "n_pos": n_pos, "seed": int(seed)}
    if n == 0 or n_pos < min_pos or n_pos == n:
        p = (n_pos + 0.5) / (n + 1.0)
        info["model"] = "constant"
        return te.constant_forest(X.shape[1], p, F.FEATURE_NAMES, dict(meta or {}, **info)), None, info
    import lightgbm as lgb
    n_neg = n - n_pos
    w = np.where(y == 1, 0.5 * n / n_pos, 0.5 * n / n_neg)
    params = dict(LGB_PARAMS, seed=int(seed), num_threads=int(threads))
    ds = lgb.Dataset(np.asarray(X, dtype=np.float64), label=y, weight=w, feature_name=list(F.FEATURE_NAMES),
                     free_raw_data=True)
    bst = lgb.train(params, ds, num_boost_round=n_trees)
    bias = -math.log(n_neg / n_pos)            # prior shift: logit p = logit p_bal - log(w1/w0)
    info.update(model="lgbm", n_trees=bst.num_trees(), prior_shift=bias)
    f = te.compile_lgbm(bst, bias=bias, feature_names=F.FEATURE_NAMES, meta=dict(meta or {}, **info))
    return f, bst, info


def second_moment_table(cell: np.ndarray, f: np.ndarray, kind: str, prior: float = PRIOR_WEIGHT) -> dict:
    """m[cell] = E[f^2 | f > 0, cell] shrunk to the global mean (prior weight 50); 1 for indicators."""
    cell = np.asarray(cell, dtype=np.int64)
    f = np.asarray(f, dtype=np.float64)
    hit = f > 0
    n = np.bincount(cell[hit], minlength=N_CELLS).astype(np.float64)
    if kind == "indicator" or not hit.any():
        m = np.ones(N_CELLS)
        mu = 1.0
    else:
        s2 = np.bincount(cell[hit], weights=f[hit] ** 2, minlength=N_CELLS)
        mu = float(s2.sum() / n.sum())
        m = (s2 + prior * mu) / (n + prior)
    return {"m": m, "n_hits": n, "global": mu}


def stratum_table(stratum: np.ndarray, hit: np.ndarray, elig: np.ndarray, prior: float = PRIOR_WEIGHT) -> dict:
    """p̄[stratum] = (hits + prior * rate) / (eligible + prior), rate = global hit rate among eligible units."""
    stratum = np.asarray(stratum, dtype=np.int64)
    elig = np.asarray(elig, dtype=bool)
    hit = np.asarray(hit, dtype=bool) & elig
    ne = np.bincount(stratum[elig], minlength=N_STRATA).astype(np.float64)
    nh = np.bincount(stratum[hit], minlength=N_STRATA).astype(np.float64)
    rate = float(nh.sum() / ne.sum()) if ne.sum() > 0 else 0.0
    p = (nh + prior * rate) / (ne + prior)
    return {"p": p, "n_elig": ne, "n_hits": nh, "rate": rate}


# ------------------------------------------------------------------------------------------ #
# integer pipeline
# ------------------------------------------------------------------------------------------ #
def m_fix(m: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(m, dtype=np.float64) * TWO16).astype(np.int64)


def p_fix(p: np.ndarray) -> np.ndarray:
    return np.minimum(np.rint(np.asarray(p, dtype=np.float64) * TWO16), 65535).astype(np.int64)


def q2_fix(Qt) -> list[int]:
    """Q2_q = max(1, round(Q̃_q² 2^32)) as Python ints (from the decimal repr of Q̃, see intpi.to_fraction)."""
    from .intpi import to_fraction
    out = []
    for v in np.atleast_1d(Qt):
        fr = to_fraction(float(v))
        out.append(max(1, round(fr * fr * TWO32)))
    return out


def equal_weights(n: int) -> list[int]:
    """W_q = round(2^16 / n) for n active estimands (Σ W ≈ 2^16)."""
    if n <= 0:
        return []
    return [round(TWO16 / n)] * n


def ghat_int(P: np.ndarray, M_cells: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """ĝ = (p · M_q[cell]) >> 16; P (N, Q) Q0.16 ints, M_cells (Q, N_CELLS) Q16.16 ints."""
    P = np.asarray(P, dtype=np.int64)
    M = np.asarray(M_cells, dtype=np.int64)[:, np.asarray(cell, dtype=np.int64)].T     # (N, Q)
    if P.size and (M.max(initial=0) >= (1 << 46)):
        raise OverflowError("second-moment table too large for int64 products")
    return (P * M) >> 16


def a_fix_int(G: np.ndarray, W, Q2) -> np.ndarray:
    """a_fix = min(2^32 - 1, isqrt(Σ_q (W_q ĝ_q << 32) // Q2_q)) per row (exact Python-int arithmetic)."""
    G = np.asarray(G, dtype=np.int64)
    N, Q = G.shape
    if Q == 0:
        return np.zeros(N, dtype=np.int64)
    acc = np.zeros(N, dtype=object)
    for j in range(Q):
        if W[j] == 0:
            continue
        col = G[:, j].astype(object)
        acc = acc + (col * int(W[j]) * TWO32) // int(Q2[j])
    out = np.fromiter((min(U32MAX, math.isqrt(int(v))) for v in acc), dtype=np.int64, count=N)
    return out


def utility_int(a_fix: np.ndarray, boost_mult=None) -> np.ndarray:
    """â = a_fix · κ_b(boosted) + 1 (int64; â < 2^35 for κ_b <= 8)."""
    a = np.asarray(a_fix, dtype=np.int64)
    if boost_mult is not None:
        a = a * np.asarray(boost_mult, dtype=np.int64)
    return a + 1


def utility_float(Pf: np.ndarray, M: np.ndarray, cell: np.ndarray, w, Qt, boost_mult=None) -> np.ndarray:
    """Float twin: sqrt(Σ_q w_q p_q m_q[cell] / Q̃_q²) κ + 2^-16."""
    Mf = np.asarray(M, dtype=np.float64)[:, np.asarray(cell, dtype=np.int64)].T
    w = np.asarray(w, dtype=np.float64)
    Qt = np.asarray(Qt, dtype=np.float64)
    a = np.sqrt(np.sum(np.asarray(Pf, dtype=np.float64) * Mf * (w / Qt ** 2)[None, :], axis=1))
    if boost_mult is not None:
        a = a * np.asarray(boost_mult, dtype=np.float64)
    return a + 2.0 ** -16


def oracle_ghat(f: np.ndarray) -> np.ndarray:
    """U7: ĝ = f² in Q16.16 (true second moment)."""
    f = np.asarray(f, dtype=np.int64)
    return (f * f) << 16


def dedup_rows(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(index of the first occurrence of each distinct row, inverse map) of an integer matrix (exact)."""
    M = np.ascontiguousarray(M)
    if M.ndim != 2 or M.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(M.shape[0] if M.ndim else 0, np.int64)
    v = M.view(np.dtype((np.void, M.dtype.itemsize * M.shape[1]))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return first, inv.ravel()


def a_fix_dedup(G: np.ndarray, W, Q2) -> np.ndarray:
    """a_fix_int evaluated once per distinct ĝ row (identical integers)."""
    G = np.asarray(G, dtype=np.int64)
    if G.shape[0] == 0 or G.shape[1] == 0:
        return a_fix_int(G, W, Q2)
    first, inv = dedup_rows(G)
    return a_fix_int(G[first], W, Q2)[inv]


def predict_all(b: "UtilityBundle", X: np.ndarray, elig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P (N, len(b.queries)) Q0.16 (0 where the unit is not q-eligible) and p_mal (N,), every forest evaluated
    once per distinct (feature row, eligibility bits) — identical to b.p_int / b.pmal_int row by row."""
    X = np.asarray(X)
    elig = np.asarray(elig, dtype=bool)
    if X.shape[0] == 0:
        return np.zeros((0, len(b.queries)), np.int64), np.zeros(0, np.int64)
    key = np.concatenate([X.astype(np.int32), elig.astype(np.int32)], axis=1)
    first, inv = dedup_rows(key)
    Xu, Eu = X[first], elig[first]
    return b.p_int(Xu, Eu)[inv], b.pmal_int(Xu)[inv]


def utilities_from_P(b: "UtilityBundle", c: dict, P_all: np.ndarray, pmal: np.ndarray, active: list, W, Q2,
                     f_true: dict | None = None, which=("U3", "U5", "U8", "U7")) -> dict:
    """The integers of UtilityBundle.utilities for PAYLOAD-ONLY columns ``c`` from precomputed predict_all
    outputs (P over b.queries, eligibility already applied); ĝ rows deduplicated before the big-int sums."""
    out = {}
    cell = cell_index(c)
    el = eligibility_matrix(active, c)
    if "U3" in which:
        out["U3"] = utility_int(pmal)
    if "U5" in which:
        qi = [b.queries.index(q) for q in active]
        G = ghat_int(np.asarray(P_all)[:, qi], b.M_cells(active), cell)
        out["U5"] = utility_int(a_fix_dedup(G, W, Q2))
        out["_G_U5"] = G
    if "U8" in which:
        st = stratum_index(c)
        P8 = (np.stack([b.P8[q][st] for q in active], axis=1) if active
              else np.zeros((len(cell), 0), np.int64))
        G = ghat_int(np.where(el, P8, 0), b.M_cells(active), cell)
        out["U8"] = utility_int(a_fix_dedup(G, W, Q2))
    if "U7" in which:
        if f_true is None:
            raise ValueError("U7 needs the true f values")
        Fm = (np.stack([np.asarray(f_true[q]) for q in active], axis=1) if active
              else np.zeros((len(cell), 0), np.int64))
        out["U7"] = utility_int(a_fix_dedup(oracle_ghat(np.where(el, Fm, 0)), W, Q2))
    return out


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
    """Mann-Whitney AUC with ties (None if one class is empty)."""
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(np.asarray(score, dtype=np.float64))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------ #
# the committed model bundle θ
# ------------------------------------------------------------------------------------------ #
@dataclass
class UtilityBundle:
    queries: list                         # estimands with a hit model (training positives >= 1)
    hit: dict                             # q -> IntForest
    gmal: te.IntForest
    M: dict                               # q -> (N_CELLS,) int Q16.16
    P8: dict                              # q -> (N_STRATA,) int Q0.16
    m_float: dict = field(default_factory=dict)
    p8_float: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    # ---- per-unit pieces ------------------------------------------------------------------- #
    def p_int(self, X: np.ndarray, elig: np.ndarray, queries=None) -> np.ndarray:
        qs = list(self.queries if queries is None else queries)
        N = X.shape[0]
        P = np.zeros((N, len(qs)), dtype=np.int64)
        for j, q in enumerate(qs):
            m = np.asarray(elig[:, self.queries.index(q)] if queries is None else elig[:, j], dtype=bool)
            if m.any():
                P[m, j] = self.hit[q].predict_int(X[m])
        return P

    def p_float(self, X: np.ndarray, elig: np.ndarray) -> np.ndarray:
        P = np.zeros((X.shape[0], len(self.queries)))
        for j, q in enumerate(self.queries):
            m = np.asarray(elig[:, j], dtype=bool)
            if m.any():
                P[m, j] = self.hit[q].predict_float(X[m])
        return P

    def pmal_int(self, X: np.ndarray) -> np.ndarray:
        return self.gmal.predict_int(X)

    def p8_int(self, stratum: np.ndarray, elig: np.ndarray) -> np.ndarray:
        S = np.stack([self.P8[q][np.asarray(stratum, dtype=np.int64)] for q in self.queries], axis=1) \
            if self.queries else np.zeros((len(stratum), 0), np.int64)
        return np.where(np.asarray(elig, dtype=bool), S, 0)

    def M_cells(self, queries=None) -> np.ndarray:
        qs = self.queries if queries is None else queries
        return np.stack([self.M[q] for q in qs], axis=0) if qs else np.zeros((0, N_CELLS), np.int64)

    # ---- aggregated utilities (integer) ----------------------------------------------------- #
    def utilities(self, c: dict, X: np.ndarray, active: list, W: list, Q2: list, f_true: dict | None = None,
                  which=("U3", "U5", "U8")) -> dict:
        """Integer â (int64) of the requested utilities for the units of ``c``/``X`` (zero-payload units
        get â = 0). ``active`` = estimands with weight (subset of self.queries for U5/U8; U7 uses f_true)."""
        pay = c["pay"].astype(bool)
        N = len(pay)
        out = {}
        idx = np.nonzero(pay)[0]
        Xp = X[idx]
        cp = F.take_cols(c, idx)
        cell = cell_index(cp)
        el = eligibility_matrix(active, cp)
        if "U3" in which:
            a = np.zeros(N, np.int64)
            a[idx] = utility_int(self.pmal_int(Xp))
            out["U3"] = a
        if "U5" in which:
            P = np.zeros((len(idx), len(active)), dtype=np.int64)
            for j, q in enumerate(active):
                m = el[:, j]
                if m.any():
                    P[m, j] = self.hit[q].predict_int(Xp[m])
            G = ghat_int(P, self.M_cells(active), cell)
            a = np.zeros(N, np.int64)
            a[idx] = utility_int(a_fix_int(G, W, Q2))
            out["U5"] = a
            out["_P_U5"] = P
            out["_G_U5"] = G
        if "U8" in which:
            st = stratum_index(cp)
            P8 = np.stack([self.P8[q][st] for q in active], axis=1) if active else np.zeros((len(idx), 0), np.int64)
            P8 = np.where(el, P8, 0)
            G = ghat_int(P8, self.M_cells(active), cell)
            a = np.zeros(N, np.int64)
            a[idx] = utility_int(a_fix_int(G, W, Q2))
            out["U8"] = a
        if "U7" in which:
            if f_true is None:
                raise ValueError("U7 needs the true f values")
            Fm = np.stack([np.asarray(f_true[q])[idx] for q in active], axis=1) if active else np.zeros((len(idx), 0))
            G = oracle_ghat(np.where(el, Fm, 0))
            a = np.zeros(N, np.int64)
            a[idx] = utility_int(a_fix_int(G, W, Q2))
            out["U7"] = a
        return out

    # ---- serialisation ---------------------------------------------------------------------- #
    def save(self, d: Path) -> None:
        from ..utils.io import atomic_write_json, atomic_write_parquet, atomic_write_text
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(d / "gmal.json", self.gmal.to_json())
        for q, f in self.hit.items():
            atomic_write_text(d / f"hit_{q}.json", f.to_json())
        import pyarrow as pa
        rows = {"q": [], "cell": [], "m": [], "M_fix": []}
        for q in self.queries:
            for k in range(N_CELLS):
                rows["q"].append(q)
                rows["cell"].append(k)
                rows["m"].append(float(self.m_float.get(q, np.ones(N_CELLS))[k]))
                rows["M_fix"].append(int(self.M[q][k]))
        atomic_write_parquet(d / "m2.parquet", pa.table(rows))
        rows = {"q": [], "stratum": [], "p": [], "P_fix": []}
        for q in self.queries:
            for k in range(N_STRATA):
                rows["q"].append(q)
                rows["stratum"].append(k)
                rows["p"].append(float(self.p8_float.get(q, np.zeros(N_STRATA))[k]))
                rows["P_fix"].append(int(self.P8[q][k]))
        atomic_write_parquet(d / "u8.parquet", pa.table(rows))
        atomic_write_json(d / "bundle.json", {"version": UTILITY_VERSION, "queries": list(self.queries),
                                             "feature_version": F.FEATURE_VERSION, "lut_hash": te.lut_hash(),
                                             "theta": self.theta_digest(), "info": self.info})

    @classmethod
    def load(cls, d: Path, attach_lgbm: bool = False) -> "UtilityBundle":
        import pyarrow.parquet as pq
        d = Path(d)
        b = json.loads((d / "bundle.json").read_text())
        if b["version"] != UTILITY_VERSION or b["feature_version"] != F.FEATURE_VERSION:
            raise ValueError("utility bundle version mismatch")
        qs = list(b["queries"])
        hit = {q: te.IntForest.from_json((d / f"hit_{q}.json").read_text()) for q in qs}
        gmal = te.IntForest.from_json((d / "gmal.json").read_text())
        if attach_lgbm:
            import lightgbm as lgb
            for name, f in [("gmal", gmal)] + [(f"hit_{q}", hit[q]) for q in qs]:
                p = d / f"{name}.lgb.txt"
                if p.exists() and f.n_trees:
                    f.attach(lgb.Booster(model_file=str(p)))
        m2 = pq.read_table(d / "m2.parquet").to_pandas()
        u8 = pq.read_table(d / "u8.parquet").to_pandas()
        M, mf, P8, pf = {}, {}, {}, {}
        for q in qs:
            g = m2[m2.q == q].sort_values("cell")
            M[q] = g.M_fix.to_numpy().astype(np.int64)
            mf[q] = g.m.to_numpy()
            g = u8[u8.q == q].sort_values("stratum")
            P8[q] = g.P_fix.to_numpy().astype(np.int64)
            pf[q] = g.p.to_numpy()
        out = cls(queries=qs, hit=hit, gmal=gmal, M=M, P8=P8, m_float=mf, p8_float=pf, info=b.get("info", {}))
        if out.theta_digest() != b["theta"]:
            raise ValueError("utility bundle digest mismatch")
        return out

    def theta_digest(self) -> str:
        """BLAKE3 over every integer that determines â (the H(θ) payload; Q̃/W are policy parameters)."""
        h = blake3(b"CTS/v1/utility-bundle/" + UTILITY_VERSION.encode())
        h.update(F.FEATURE_VERSION.encode())
        h.update(self.gmal.theta_hash().encode())
        for q in self.queries:
            h.update(q.encode() + b"\x00" + self.hit[q].theta_hash().encode())
            h.update(np.asarray(self.M[q], dtype="<i8").tobytes())
            h.update(np.asarray(self.P8[q], dtype="<i8").tobytes())
        return h.hexdigest()


# ------------------------------------------------------------------------------------------ #
# adaptive-arm calibration (boost replay on the TRAINING streams)
# ------------------------------------------------------------------------------------------ #
# Boost trigger set F: attack-indicative content detectors (Mirai telnet prompts, loader/dropper strings, HTTP
# injection, IRC C&C, webshell commands). Benign-protocol detectors (D5 HTTP requests, D6 DNS questions,
# D7 NXDOMAIN, D8/D9 MQTT, D10 Modbus writes, D11 SSH banners) would boost nearly every device.
BOOST_DETECTORS = ("D1", "D2", "D3", "D4", "D12", "D13")      # fallback only; the committed set is per fold:
BOOST_QUERIES = ("D1o", "D1c", "D2o", "D2c", "D3", "D4", "D12", "D13")
# Pre-registered training-data rule (coordinator, 18 Oct): a raw detector enters the trigger set F iff its firing
# rate on the BENIGN-labelled payload units of the fold's TRAINING streams is <= BOOST_FP_MAX.
BOOST_FP_MAX = 1e-3
DETECTOR_ESTIMANDS = {"D1": ("D1o", "D1c"), "D2": ("D2o", "D2c"), "D3": ("D3",), "D4": ("D4",), "D5": ("D5",),
                      "D6": ("D6",), "D7": ("D7",), "D8": ("D8",), "D9": ("D9",), "D10": ("D10",), "D11": ("D11o",),
                      "D12": ("D12",), "D13": ("D13",)}


def boost_trigger_set(streams, fp_max: float = BOOST_FP_MAX) -> dict:
    """Firing rate of every raw detector on the benign-labelled payload units of the given (training)
    StreamTables, and the trigger set F = {d : rate_d <= fp_max} with its estimand ids."""
    n_ben = 0
    fired = {d: 0 for d in DETECTOR_ESTIMANDS}
    for T in streams:
        lab = np.asarray(T.label)
        ben = np.asarray(T.pay, dtype=bool) & (lab == 0)
        n_ben += int(ben.sum())
        for d, qs in DETECTOR_ESTIMANDS.items():
            m = np.zeros(T.n_units, bool)
            for q in qs:
                if q in T.query_names:
                    m[T.hits(q)[0]] = True
            fired[d] += int((m & ben).sum())
    rates = {d: fired[d] / max(n_ben, 1) for d in fired}
    F = [d for d in DETECTOR_ESTIMANDS if rates[d] <= fp_max]
    return {"rule": f"rate on benign payload units of the training streams <= {fp_max:g}", "n_benign_payload": n_ben,
            "fired": fired, "rates": rates, "F": F, "queries": [q for d in F for q in DETECTOR_ESTIMANDS[d]]}


def calibrate_lambda_adaptive(s_pay, a_pay, dev_pay, ep_pay, trig_pay, n_epochs: int, configs: list[dict],
                              kappa_b: float = 5.0, W_b: int = 20, R: int = 4, seed: int = 0, lo: float = 0.15,
                              hi: float = 1.2, n_grid: int = 8, rounds: int = 2) -> list[tuple[float, dict]]:
    """λ_ref^A of the ADAPTIVE arm for several configs at once (K = len(configs); each config: lam_A, lam_S
    (None for a pure-A design), tau_mix, phi0, Bbar). With the follow-the-evidence boost replayed on the
    training payload units (Z ~ Bernoulli(π) of the full design; units with trig_pay boost their device for
    epochs e+2..e+1+W_b, multiplier κ_b on a), the mean over R replicates of the A component's per-epoch expected
    cost equals Bbar. λ^S keeps its static value. All configs and grid candidates c = λ^A/λ^A_static share
    common random numbers, so the cost is monotone in c per replicate (u < π coupling). Grid search on c with
    ``rounds`` zoom-ins, then log-linear interpolation. Returns [(λ^A_adaptive, info)] per config."""
    from .boost import BoostState
    from .probs import pi_A, pi_S, sqrt_eff
    s_pay = np.asarray(s_pay, dtype=np.float64)
    a_pay = np.asarray(a_pay, dtype=np.float64)
    dev_pay = np.asarray(dev_pay, dtype=np.int64)
    ep_pay = np.asarray(ep_pay, dtype=np.int64)
    trig_pay = np.asarray(trig_pay, dtype=bool)
    K = len(configs)
    lamA0 = np.array([float(c["lam_A"]) for c in configs])
    lamS = np.array([float(c["lam_S"]) if c.get("lam_S") is not None else 0.0 for c in configs])
    tau = np.array([float(c.get("tau_mix", 0.0)) if c.get("lam_S") is not None else 0.0 for c in configs])
    phi0 = np.array([float(c["phi0"]) for c in configs])
    Bbar = np.array([float(c["Bbar"]) for c in configs])
    sq_all = sqrt_eff(s_pay)
    # Only devices with >= 1 trigger unit can ever be boosted: their units are replayed epoch by epoch; every
    # other unit's π^A does not depend on the boost state, so its cost is closed form (same result, far faster).
    boostable = np.unique(dev_pay[trig_pay])
    mb = np.isin(dev_pay, boostable)
    s_nb, a_nb, sq_nb = s_pay[~mb], a_pay[~mb], sq_all[~mb]
    s_b, a_b, sq = s_pay[mb], a_pay[mb], sq_all[mb]
    ep_b, trig_b = ep_pay[mb], trig_pay[mb]
    dev_b = np.searchsorted(boostable, dev_pay[mb])
    n_dev = max(1, len(boostable))
    eptr = np.searchsorted(ep_b, np.arange(n_epochs + 1))
    tset = set(np.unique(ep_b[trig_b]).tolist())

    def cost_nb(cs: np.ndarray) -> np.ndarray:              # (K, C) closed-form part
        out = np.zeros(cs.shape)
        if not len(s_nb):
            return out
        for k in range(K):
            for j in range(cs.shape[1]):
                pA = pi_A(lamA0[k] * cs[k, j], a_nb, s_nb, phi0[k], sqrt_s=sq_nb)
                out[k, j] = float(np.dot(pA, s_nb))
        return out

    def cost(cs: np.ndarray) -> np.ndarray:                 # cs (K, C) -> mean A cost per epoch (K, C)
        C = cs.shape[1]
        shape = (K, C, R)
        bs = BoostState(n_dev, shape=shape, kappa_b=kappa_b, W_b=W_b)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 0xB0057])))
        lamA = (lamA0[:, None] * cs)[:, :, None, None]
        ph = phi0[:, None, None, None]
        lS = lamS[:, None, None, None]
        tm = tau[:, None, None, None]
        tot = np.zeros(shape)
        none = np.zeros(shape + (n_dev,), bool)
        for e in range(n_epochs):
            bs.begin_epoch(e)
            a0, a1 = int(eptr[e]), int(eptr[e + 1])
            if a1 > a0:
                mult = bs.multiplier(dev_b[a0:a1])
                pA = pi_A(lamA, a_b[a0:a1] * mult, s_b[a0:a1], ph, sqrt_s=sq[a0:a1])
                tot += (pA * s_b[a0:a1]).sum(-1)
                if e in tset:
                    ti = np.nonzero(trig_b[a0:a1])[0]
                    p = pA[..., ti]
                    pS = pi_S(lS, s_b[a0:a1][ti], ph, sqrt_s=sq[a0:a1][ti])
                    p = (1.0 - tm) * p + tm * pS
                    U_ = rng.random((R, len(ti)))
                    bs.record_units(U_[None, None] < p, np.ones(len(ti), bool), dev_b[a0:a1][ti])
                    continue
            bs.end_epoch(none)
        return (tot.mean(axis=2) + cost_nb(cs)) / n_epochs

    lo_c = np.full(K, float(lo))
    hi_c = np.full(K, float(hi))
    if not trig_pay.any():
        return [(float(lamA0[k]), {"note": "no trigger units: adaptive = static", "c": 1.0}) for k in range(K)]
    for _ in range(rounds + 3):
        cs = np.exp(np.linspace(np.log(lo_c), np.log(hi_c), n_grid, axis=1))      # (K, n_grid)
        cst = cost(cs)
        done = True
        for k in range(K):
            if cst[k, 0] > Bbar[k]:
                lo_c[k], hi_c[k] = lo_c[k] / 4.0, cs[k, 0]
                done = False
            elif cst[k, -1] < Bbar[k]:
                lo_c[k], hi_c[k] = cs[k, -1], hi_c[k] * 4.0
                done = False
            else:
                j = int(np.searchsorted(cst[k], Bbar[k]))
                j = min(max(j, 1), n_grid - 1)
                lo_c[k], hi_c[k] = cs[k, j - 1], cs[k, j]
        rounds -= 1
        if done and rounds <= 0:
            break
    f = cost(np.stack([lo_c, hi_c], axis=1))
    c1 = cost(np.ones((K, 1)))[:, 0]
    out = []
    for k in range(K):
        fl, fh = f[k]
        t = (Bbar[k] - fl) / (fh - fl) if fh > fl else 0.0
        c = math.exp(math.log(lo_c[k]) + min(max(t, 0.0), 1.0) * (math.log(hi_c[k]) - math.log(lo_c[k])))
        out.append((float(lamA0[k] * c), {"c": c, "bracket": [float(lo_c[k]), float(hi_c[k])],
                                          "cost_bracket": [float(fl), float(fh)], "Bbar": float(Bbar[k]),
                                          "cost_static_c1": float(c1[k]), "R": R, "seed": seed,
                                          "kappa_b": kappa_b, "W_b": W_b, "n_trigger_units": int(trig_pay.sum())}))
    return out
