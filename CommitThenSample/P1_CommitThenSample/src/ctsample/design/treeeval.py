"""Bit-exact integer evaluator for LightGBM binary forests (DESIGN §3.4 "Bit-exact integer evaluator").

    compile_lgbm(booster) -> IntForest
      * leaf values v -> round(v * 2^16) as int32 (Q15.16);
      * thresholds snapped to the int32 feature grid: x <= t  <=>  x <= floor(t) for integer x;
      * score S = bias + sum of the integer leaves, int64, in tree order (exact, order-free);
      * p = LUT[clip(S >> 8, -2^15, 2^15 - 1)] with a 65,536-entry u16 table (Q0.16) committed in
        H(policy): LUT[j + 2^15] = min(65535, round_half_even(2^16 * sigmoid((j + 1/2)/256))), computed
        with correctly rounded decimal arithmetic, so it is platform independent; ``lut_hash()`` is its
        BLAKE3 digest and every IntForest carries it.
    predict_int(X)   -> int64 p in [0, 65535]  (vectorised numpy, deterministic; bit-exact)
    predict_score(X) -> int64 S
    predict_float(X) -> float64 sigmoid(bias_f + sum of the float leaves) with the float thresholds
                        (the LightGBM reference; tests compare it with booster.predict(raw_score=True))
    to_json / from_json; theta_hash() = BLAKE3("CTS/v1/intforest" || canonical JSON) = H(theta) part.

Fast path: ``predict_int(X, leaves=L)`` with L = booster.predict(X, pred_leaf=True) (LightGBM's C++
traversal) gives the SAME integers (tests check it on 10^6 rows); ``attach(booster)`` makes it the
default. The pure numpy traversal is the protocol / verifier path.

``bias`` holds the prior-shift calibration of a class-balanced model: training with positive weight
w1 and negative weight w0 makes the model estimate logit(p) + log(w1/w0), so bias = -log(w1/w0)
returns calibrated probabilities (utility.py). An IntForest with no trees is a constant model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

import numpy as np
from blake3 import blake3

__all__ = ["IntForest", "compile_lgbm", "constant_forest", "sigmoid_lut", "lut_hash", "LUT_ID", "LEAF_SCALE",
           "LUT_SHIFT", "lut_max_step"]

LEAF_SCALE = 1 << 16
LUT_SHIFT = 8
LUT_HALF = 1 << 15
LUT_ID = "sigmoid_q8_mid_v1"
DOM_FOREST = b"CTS/v1/intforest"
_I32 = (-(1 << 31), (1 << 31) - 1)

_LUT: np.ndarray | None = None
_LUT_HASH: str | None = None


def sigmoid_lut() -> np.ndarray:
    """The committed 65,536-entry u16 sigmoid table (cached)."""
    global _LUT, _LUT_HASH
    if _LUT is None:
        out = np.empty(1 << 16, dtype=np.uint16)
        with localcontext() as ctx:
            ctx.prec = 40
            one = Decimal(1)
            sc = Decimal(1 << 16)
            den = Decimal(2 << LUT_SHIFT)            # (j + 1/2) / 256 = (2j + 1) / 512
            for k in range(1 << 16):
                z = Decimal(2 * (k - LUT_HALF) + 1) / den
                v = int((sc / (one + (-z).exp())).to_integral_value(rounding=ROUND_HALF_EVEN))
                out[k] = min(v, 65535)
        _LUT = out
        _LUT_HASH = blake3(b"CTS/v1/lut/" + LUT_ID.encode() + out.astype("<u2").tobytes()).hexdigest()
    return _LUT


def lut_hash() -> str:
    sigmoid_lut()
    return _LUT_HASH


def lut_max_step() -> float:
    """Largest difference between adjacent LUT entries, as a probability (the 'LUT step')."""
    L = sigmoid_lut().astype(np.int64)
    return float(np.max(np.abs(np.diff(L)))) / LEAF_SCALE


@dataclass
class _Tree:
    feat: np.ndarray        # (n_int,) int32 split feature
    thr_i: np.ndarray       # (n_int,) int64 threshold on the integer grid
    thr_f: np.ndarray       # (n_int,) float64 original threshold
    left: np.ndarray        # (n_int,) int32 child: >= 0 internal node, < 0 -> leaf ~child
    right: np.ndarray
    leaf_i: np.ndarray      # (n_leaves,) int64 Q15.16
    leaf_f: np.ndarray      # (n_leaves,) float64

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_i)


@dataclass
class IntForest:
    n_features: int
    trees: list
    bias_i: int = 0
    bias_f: float = 0.0
    feature_names: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    lut_id: str = LUT_ID
    booster: object = field(default=None, repr=False, compare=False)

    # ---------------------------------------------------------------- evaluation
    def _check_X(self, X) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (N, {self.n_features})")
        if not np.issubdtype(X.dtype, np.integer):
            raise TypeError("IntForest needs integer features")
        return X.astype(np.int64, copy=False)

    def _flat(self):
        """All trees in global flat arrays (children: >= 0 global internal node, < 0 -> ~global leaf)."""
        if getattr(self, "_flat_cache", None) is None:
            feat, thr, left, right, roots, loff = [], [], [], [], [], []
            n_int = n_leaf = 0
            for tr in self.trees:
                loff.append(n_leaf)
                if len(tr.feat) == 0:
                    roots.append(~n_leaf)
                else:
                    roots.append(n_int)
                    feat.append(tr.feat.astype(np.int64))
                    thr.append(tr.thr_i.astype(np.int64))
                    for src, dst in ((tr.left, left), (tr.right, right)):
                        c = src.astype(np.int64)
                        dst.append(np.where(c >= 0, c + n_int, ~(~c + n_leaf)))
                    n_int += len(tr.feat)
                n_leaf += tr.n_leaves
            cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64)  # noqa: E731
            leaf_val = (np.concatenate([tr.leaf_i for tr in self.trees]) if self.trees else np.zeros(0, np.int64))
            self._flat_cache = (cat(feat), cat(thr), cat(left), cat(right), np.asarray(roots, np.int64),
                                np.asarray(loff, np.int64), leaf_val.astype(np.int64))
        return self._flat_cache

    def leaves(self, X, chunk: int = 16384) -> np.ndarray:
        """(N, n_trees) leaf indices by the integer traversal (numpy, all trees at once; protocol path)."""
        X = self._check_X(X)
        N = X.shape[0]
        T = len(self.trees)
        out = np.zeros((N, T), dtype=np.int32)
        if T == 0 or N == 0:
            return out
        feat, thr, left, right, roots, loff, _ = self._flat()
        for a in range(0, N, chunk):
            Xc = X[a:a + chunk]
            n = Xc.shape[0]
            flatx = Xc.ravel()
            node = np.broadcast_to(roots, (n, T)).copy()
            rowbase = (np.arange(n, dtype=np.int64) * self.n_features)[:, None]
            act = np.nonzero(node.ravel() >= 0)[0]
            nflat = node.ravel()
            rb = np.broadcast_to(rowbase, (n, T)).ravel()
            while act.size:
                nd = nflat[act]
                xv = flatx[rb[act] + feat[nd]]
                nxt = np.where(xv <= thr[nd], left[nd], right[nd])
                nflat[act] = nxt
                act = act[nxt >= 0]
            out[a:a + n] = (~nflat.reshape(n, T)) - loff[None, :]
        return out

    def predict_score(self, X, leaves=None) -> np.ndarray:
        """S = bias + sum_t leaf_int (int64)."""
        if leaves is None and self.booster is not None:
            X = self._check_X(X)
            leaves = np.asarray(self.booster.predict(X.astype(np.float64), pred_leaf=True), dtype=np.int64)
            if leaves.ndim == 1:
                leaves = leaves[:, None]
        if leaves is None:
            leaves = self.leaves(X)
        leaves = np.asarray(leaves, dtype=np.int64)
        N = leaves.shape[0]
        if leaves.shape[1] != len(self.trees):
            raise ValueError("leaf matrix does not match the forest")
        S = np.full(N, int(self.bias_i), dtype=np.int64)
        for t, tr in enumerate(self.trees):
            S += tr.leaf_i[leaves[:, t]]
        return S

    def predict_int(self, X, leaves=None) -> np.ndarray:
        """Integer probability p in [0, 65535] (Q0.16), bit-exact."""
        S = self.predict_score(X, leaves)
        idx = np.clip(S >> LUT_SHIFT, -LUT_HALF, LUT_HALF - 1) + LUT_HALF
        return sigmoid_lut()[idx].astype(np.int64)

    def raw_float(self, X) -> np.ndarray:
        """float64 raw score with the ORIGINAL float thresholds and leaves (LightGBM semantics)."""
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        raw = np.full(N, float(self.bias_f))
        for tr in self.trees:
            if len(tr.feat) == 0:
                raw += tr.leaf_f[0]
                continue
            node = np.zeros(N, dtype=np.int64)
            res = np.zeros(N, dtype=np.int64)
            rows = np.arange(N)
            while rows.size:
                xv = X[rows, tr.feat[node]]
                nxt = np.where(xv <= tr.thr_f[node], tr.left[node], tr.right[node]).astype(np.int64)
                leaf = nxt < 0
                res[rows[leaf]] = ~nxt[leaf]
                rows = rows[~leaf]
                node = nxt[~leaf]
            raw += tr.leaf_f[res]
        return raw

    def predict_float(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.raw_float(X)))

    def attach(self, booster) -> "IntForest":
        """Use LightGBM's pred_leaf traversal for predict_int (same integers, faster)."""
        self.booster = booster
        return self

    # ---------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        trees = []
        for tr in self.trees:
            trees.append({"feat": tr.feat.tolist(), "thr_i": [int(x) for x in tr.thr_i],
                          "thr_f": [float(x) for x in tr.thr_f], "left": tr.left.tolist(), "right": tr.right.tolist(),
                          "leaf_i": [int(x) for x in tr.leaf_i], "leaf_f": [float(x) for x in tr.leaf_f]})
        return {"format": "ctsample.intforest", "version": 1, "n_features": int(self.n_features),
                "feature_names": list(self.feature_names), "bias_i": int(self.bias_i), "bias_f": float(self.bias_f),
                "lut_id": self.lut_id, "lut_hash": lut_hash(), "leaf_scale": LEAF_SCALE, "lut_shift": LUT_SHIFT,
                "trees": trees, "meta": self.meta}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "IntForest":
        if d.get("format") != "ctsample.intforest":
            raise ValueError("not an IntForest")
        if d.get("lut_id") != LUT_ID or d.get("lut_hash") != lut_hash():
            raise ValueError("IntForest was compiled against a different sigmoid LUT")
        if d.get("leaf_scale") != LEAF_SCALE or d.get("lut_shift") != LUT_SHIFT:
            raise ValueError("fixed-point scale mismatch")
        trees = [_Tree(np.asarray(t["feat"], np.int32), np.asarray(t["thr_i"], np.int64), np.asarray(t["thr_f"], np.float64),
                       np.asarray(t["left"], np.int32), np.asarray(t["right"], np.int32),
                       np.asarray(t["leaf_i"], np.int64), np.asarray(t["leaf_f"], np.float64)) for t in d["trees"]]
        return cls(n_features=int(d["n_features"]), trees=trees, bias_i=int(d["bias_i"]), bias_f=float(d["bias_f"]),
                   feature_names=list(d.get("feature_names", [])), meta=dict(d.get("meta", {})), lut_id=d["lut_id"])

    @classmethod
    def from_json(cls, s: str) -> "IntForest":
        return cls.from_dict(json.loads(s))

    def integer_payload(self) -> dict:
        """The integer content that determines predict_int (what H(theta) commits)."""
        return {"n_features": int(self.n_features), "bias_i": int(self.bias_i), "lut_id": self.lut_id,
                "lut_hash": lut_hash(), "trees": [{"feat": tr.feat.tolist(), "thr_i": [int(x) for x in tr.thr_i],
                                                   "left": tr.left.tolist(), "right": tr.right.tolist(),
                                                   "leaf_i": [int(x) for x in tr.leaf_i]} for tr in self.trees]}

    def theta_hash(self) -> str:
        js = json.dumps(self.integer_payload(), sort_keys=True, separators=(",", ":")).encode()
        return blake3(DOM_FOREST + js).hexdigest()

    @property
    def n_trees(self) -> int:
        return len(self.trees)


# ------------------------------------------------------------------------------------------ #
def _snap(t: float) -> int:
    """Largest integer k with k <= t (x <= t <=> x <= k for integers), clipped to the int32 grid."""
    if not math.isfinite(t):
        return _I32[1] if t > 0 else _I32[0] - 1
    k = math.floor(t)
    return max(_I32[0] - 1, min(_I32[1], k))


def _leaf_int(v: float) -> int:
    x = round(v * LEAF_SCALE)
    if not _I32[0] <= x <= _I32[1]:
        raise ValueError(f"leaf value {v} does not fit Q15.16 int32")
    return int(x)


def _compile_tree(ts: dict) -> _Tree:
    if "leaf_value" in ts and "split_index" not in ts:            # single-leaf tree
        v = float(ts["leaf_value"])
        return _Tree(np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros(0, np.float64), np.zeros(0, np.int32),
                     np.zeros(0, np.int32), np.array([_leaf_int(v)], np.int64), np.array([v], np.float64))
    feats, thr_i, thr_f, left, right = {}, {}, {}, {}, {}
    leaf_v: dict[int, float] = {}

    def walk(nd) -> int:
        if "split_index" not in nd:
            li = int(nd["leaf_index"])
            leaf_v[li] = float(nd["leaf_value"])
            return ~li
        j = int(nd["split_index"])
        if nd.get("decision_type", "<=") != "<=":
            raise ValueError("only numerical '<=' splits are supported (no categorical features)")
        mt = nd.get("missing_type", "None")
        if mt not in ("None", None):
            raise ValueError(f"missing_type {mt!r} not supported (features are never missing)")
        feats[j] = int(nd["split_feature"])
        t = float(nd["threshold"])
        thr_f[j] = t
        thr_i[j] = _snap(t)
        left[j] = walk(nd["left_child"])
        right[j] = walk(nd["right_child"])
        return j

    root = walk(ts)
    if root != 0:
        raise ValueError("root must be split 0")
    n_int = len(feats)
    n_leaf = len(leaf_v)
    if sorted(feats) != list(range(n_int)) or sorted(leaf_v) != list(range(n_leaf)) or n_leaf != n_int + 1:
        raise ValueError("unexpected LightGBM tree numbering")
    ar = lambda d, dt: np.array([d[i] for i in range(len(d))], dtype=dt)  # noqa: E731
    lf = ar(leaf_v, np.float64)
    return _Tree(ar(feats, np.int32), ar(thr_i, np.int64), ar(thr_f, np.float64), ar(left, np.int32),
                 ar(right, np.int32), np.array([_leaf_int(v) for v in lf], np.int64), lf)


def compile_lgbm(booster, bias: float = 0.0, feature_names=None, meta=None) -> IntForest:
    """Compile a LightGBM binary Booster (numerical splits only) into an IntForest.

    ``bias`` is added to the raw score (prior-shift calibration); it is stored both as float and as
    round(bias * 2^16)."""
    dump = booster.dump_model()
    if dump.get("num_class", 1) != 1:
        raise ValueError("binary / single-output boosters only")
    obj = str(dump.get("objective", ""))
    if obj and not obj.startswith("binary"):
        raise ValueError(f"objective {obj!r} is not binary")
    if "sigmoid:1" not in obj.replace(" ", "") and obj:
        raise ValueError(f"binary objective must use sigmoid:1 ({obj!r})")
    trees = [_compile_tree(t["tree_structure"]) for t in dump["tree_info"]]
    nf = int(dump["max_feature_idx"]) + 1
    names = list(feature_names) if feature_names is not None else list(dump.get("feature_names", []))
    f = IntForest(n_features=nf, trees=trees, bias_i=_leaf_int(bias), bias_f=float(bias), feature_names=names,
                  meta=dict(meta or {}))
    return f.attach(booster)


def constant_forest(n_features: int, p: float, feature_names=None, meta=None) -> IntForest:
    """A forest without trees that predicts the constant probability p (rare-positive fallback)."""
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    b = math.log(p / (1 - p))
    return IntForest(n_features=int(n_features), trees=[], bias_i=_leaf_int(b), bias_f=b,
                     feature_names=list(feature_names or []), meta=dict(meta or {}, constant=True, p=p))
