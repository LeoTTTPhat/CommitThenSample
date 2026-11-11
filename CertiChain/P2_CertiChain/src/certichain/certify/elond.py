"""Streaming certification at the gateway (DESIGN §3.7 Alg. 6, Thm 4) and its baselines.

Looks l = 1..L at epochs (1, 2, 4, 8, 16, 32, 64, 96) x Delta (Delta = 300 s) after the case
trigger t_0; the look-l batch = candidates whose v starts in (e_{l-1} Delta, e_l Delta].
rule_l = BH at level theta_l on the batch's reference p-values (prefix-only scorer).

Alg. 6 (batch e-LOND with group-conformal e-values):
  Q_{k,l} = FDP of calibration incident k's look-l batch under rule_l;
  E_l = (K+1) / (sum_k Q_{k,l} + 1)                         (Lemma A; test-independent)
  per incident: D <- 0; for each look with S = rule_l(batch) != {}:
      gamma* = gamma_l (per incident) or w_n gamma_l (gateway-pooled, w_n = 1/(n(n+1)))
      certify S (irrevocable) iff E_l >= |S| / (alpha gamma* (D + |S|)); then D <- D + |S|.
  Claim (Thm 4): E[sup_tau V_tau / max(D_tau, 1)] <= alpha sum_l gamma_l <= alpha.

Baselines (E6):
  * CRC-looks: at look l, CRC on the CUMULATIVE prefix (BH family over lambda) at
    alpha_l = alpha gamma_l (alpha-spending); certified sets are unioned (irrevocable).
    sup-FDP <= sum_l FDP_l, so E[sup] <= alpha.
  * naive repeated CRC: the same at alpha at every look (invalid under data-dependent stopping).
  * offline CRC at the end: one CRC on the last look's cumulative prefix at alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .crc import crc_threshold_from_sums
from .evalues import group_evalues
from .losses import monotonize
from .pvalues import ref_pvalues, sort_null_pool
from .rules import LAMBDA_GRID, bh_counts, bh_lam_star, score_order

LOOK_EPOCHS = (1, 2, 4, 8, 16, 32, 64, 96)
GAMMA_LOOKS = (0.25, 0.15, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07)
DELTA_S = 300.0
TOL = 1e-12


def look_index(v_start, t0: float = 0.0, epochs=LOOK_EPOCHS, delta: float = DELTA_S) -> np.ndarray:
    """0-based look index of each candidate (by its v's start time); -1 beyond the last look."""
    dt = (np.asarray(v_start, dtype=np.float64) - t0) / delta
    edges = np.asarray(epochs, dtype=np.float64)
    idx = np.searchsorted(edges, dt, side="left")   # dt in (e_{l-1}, e_l] -> l
    return np.where(idx < edges.size, idx, -1).astype(np.int64)


@dataclass
class StreamIncident:
    incident_id: str
    scores: np.ndarray        # prefix-only scorer
    y: np.ndarray             # {1, 0, -1}
    look: np.ndarray          # 0-based look index, -1 = after the last look (never seen)


@dataclass
class StreamBank:
    lam_grid: np.ndarray
    n_looks: int
    batch_N: np.ndarray       # [K, L]
    batch_R: np.ndarray       # [K, L, G]  BH(lambda) on the look-l batch
    batch_V: np.ndarray
    batch_TP: np.ndarray
    cum_R: np.ndarray         # [K, L, G]  BH(lambda) on the cumulative prefix (looks <= l)
    cum_V: np.ndarray
    cum_TP: np.ndarray
    n_true: np.ndarray        # [K] true links seen by the last look
    csr: dict[str, np.ndarray] = field(default_factory=dict)
    ids: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.batch_N.shape[0])

    def batch_fdp(self) -> np.ndarray:
        return self.batch_V / np.maximum(self.batch_R, 1)

    def cum_fdp(self) -> np.ndarray:
        return self.cum_V / np.maximum(self.cum_R, 1)


def _bh_R_V_TP(s_sorted, p_sorted, y_sorted, lam_grid):
    R = bh_counts(bh_lam_star(p_sorted), lam_grid)
    cf = np.concatenate(([0], np.cumsum(y_sorted != 1)))
    ct = np.concatenate(([0], np.cumsum(y_sorted == 1)))
    return R, cf[R], ct[R]


def build_stream_bank(incidents: Sequence[StreamIncident], s0, lam_grid=LAMBDA_GRID,
                      n_looks: int = len(LOOK_EPOCHS)) -> StreamBank:
    s0s = sort_null_pool(s0)
    lam_grid = np.asarray(lam_grid, dtype=np.float64)
    K, L, G = len(incidents), n_looks, lam_grid.size
    shp = (K, L, G)
    bR, bV, bT = (np.zeros(shp, np.int32) for _ in range(3))
    cR, cV, cT = (np.zeros(shp, np.int32) for _ in range(3))
    bN = np.zeros((K, L), np.int32)
    n_true = np.zeros(K, np.int32)
    sc, yy, lk, off = [], [], [], [0]
    for k, inc in enumerate(incidents):
        s = np.asarray(inc.scores, dtype=np.float64)
        y = np.asarray(inc.y, dtype=np.int8)
        lo = np.asarray(inc.look, dtype=np.int64)
        o = score_order(s)
        s, y, lo = s[o], y[o], lo[o]
        p = ref_pvalues(s, s0s)
        for l in range(L):
            m = lo == l
            bN[k, l] = int(m.sum())
            bR[k, l], bV[k, l], bT[k, l] = _bh_R_V_TP(s[m], p[m], y[m], lam_grid)
            c = (lo >= 0) & (lo <= l)
            cR[k, l], cV[k, l], cT[k, l] = _bh_R_V_TP(s[c], p[c], y[c], lam_grid)
        n_true[k] = int(np.sum((y == 1) & (lo >= 0)))
        sc.append(s); yy.append(y); lk.append(lo); off.append(off[-1] + s.size)
    csr = {"offsets": np.asarray(off, np.int64),
           "score": np.concatenate(sc) if sc else np.empty(0),
           "y": np.concatenate(yy) if yy else np.empty(0, np.int8),
           "look": np.concatenate(lk) if lk else np.empty(0, np.int64)}
    return StreamBank(lam_grid, L, bN, bR, bV, bT, cR, cV, cT, n_true, csr, [i.incident_id for i in incidents])


def choose_look_levels(train: StreamBank, alpha: float, gamma=GAMMA_LOOKS, frac: float = 1.0 / 3.0) -> np.ndarray:
    """theta_l (grid index) per look on TRAINING incidents: the last lambda before the first one whose
    training mean batch-FDP exceeds frac * alpha * gamma_l (DESIGN §3.7: frac = 1/3)."""
    m = train.batch_fdp().mean(axis=0)              # [L, G]
    tgt = frac * alpha * np.asarray(gamma, dtype=np.float64)[:, None]
    bad = m[:, 1:] > tgt + TOL
    first_bad = np.where(bad.any(1), np.argmax(bad, axis=1), bad.shape[1])
    return first_bad.astype(np.int64)                # leading OK columns r -> index r


def q_table(bank: StreamBank, theta_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per incident and look under rule_l = BH(theta_l): (|S|, V, Q = V / max(|S|, 1)), each [K, L]."""
    L = bank.n_looks
    R = bank.batch_R[:, np.arange(L), theta_idx].astype(np.int64)
    V = bank.batch_V[:, np.arange(L), theta_idx].astype(np.int64)
    return R, V, V / np.maximum(R, 1)


# --------------------------------------------------------------------------- Alg. 6
@dataclass
class Decision:
    incident: int          # position n - 1 in the exogenous incident order
    look: int
    n_sel: int
    certified: bool
    D_before: int
    E: float
    gamma_star: float


def gateway_weight(n: int) -> float:
    """w_n = 1/(n(n+1)) for the n-th incident (1-based); sum_n w_n = 1."""
    return 1.0 / (n * (n + 1.0))


def batch_elond(test_batches: Sequence[Sequence[int]], Q_cal_table: np.ndarray, alpha: float,
                gamma=GAMMA_LOOKS, w=None) -> list[Decision]:
    """Alg. 6. ``test_batches[n][l]`` = |S_{n,l}| (rule_l selection size of incident n at look l).
    ``w=None``: per incident (D restarts at 0); ``w='gateway'``: pooled with w_n = 1/(n(n+1)) and
    D accumulated across incidents; a callable or array gives custom w_n."""
    E = group_evalues(Q_cal_table)
    gamma = np.asarray(gamma, dtype=np.float64)
    out: list[Decision] = []
    D = 0
    for n, sizes in enumerate(test_batches):
        if w is None:
            D = 0
            wn = 1.0
        elif isinstance(w, str) and w == "gateway":
            wn = gateway_weight(n + 1)
        elif callable(w):
            wn = float(w(n + 1))
        else:
            wn = float(np.asarray(w)[n])
        for l, s in enumerate(sizes):
            s = int(s)
            if s == 0:
                continue
            g = wn * gamma[l]
            ok = E[l] * alpha * g * (D + s) >= s * (1.0 - TOL)
            out.append(Decision(n, l, s, bool(ok), D, float(E[l]), float(g)))
            if ok:
                D += s
    return out


def elond_paths(sizes: np.ndarray, falses: np.ndarray, trues: np.ndarray, E: np.ndarray, alpha: float,
                gamma=GAMMA_LOOKS, pooled: bool = False) -> dict[str, np.ndarray]:
    """Vectorised e-LOND over M test incidents (rows, in exogenous order) and L looks.
    Returns cumulative certified counts D, V (false), T (true) per look [M, L] and the certify mask.
    Pooled: D accumulates over incidents (gateway), gamma* = w_n gamma_l."""
    sizes = np.asarray(sizes, dtype=np.int64)
    M, L = sizes.shape
    gamma = np.asarray(gamma, dtype=np.float64)
    cert = np.zeros((M, L), dtype=bool)
    if not pooled:
        D = np.zeros(M, dtype=np.int64)
        for l in range(L):
            s = sizes[:, l]
            ok = (s > 0) & (E[l] * alpha * gamma[l] * (D + s) >= s * (1.0 - TOL))
            cert[:, l] = ok
            D = D + np.where(ok, s, 0)
    else:
        D = 0
        for n in range(M):
            wn = gateway_weight(n + 1)
            for l in range(L):
                s = int(sizes[n, l])
                if s == 0:
                    continue
                if E[l] * alpha * wn * gamma[l] * (D + s) >= s * (1.0 - TOL):
                    cert[n, l] = True
                    D += s
    cs = np.where(cert, sizes, 0)
    return {"cert": cert, "D": np.cumsum(cs, axis=1), "V": np.cumsum(np.where(cert, falses, 0), axis=1),
            "T": np.cumsum(np.where(cert, trues, 0), axis=1)}


# --------------------------------------------------------------------------- CRC-looks / naive
def crc_looks_index(bank: StreamBank, cal_rows: np.ndarray, alphas: np.ndarray, B: float = 1.0) -> np.ndarray:
    """lambda-hat_l per look from the cumulative-prefix loss tables of the calibration rows."""
    Lm = monotonize(bank.cum_fdp()[cal_rows])        # [K, L, G]
    sums = Lm.sum(axis=0)                             # [L, G]
    K = len(cal_rows)
    return np.array([crc_threshold_from_sums(sums[l][None], K, float(alphas[l]), B)[0]
                     for l in range(bank.n_looks)], dtype=np.int64)


def union_paths(bank: StreamBank, rows: np.ndarray, lam_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Cumulative union of S_{j,l}(lambda_l) over looks for each test row: D, V, T [M, L]."""
    L = bank.n_looks
    M = len(rows)
    D = np.zeros((M, L), np.int64)
    V = np.zeros((M, L), np.int64)
    T = np.zeros((M, L), np.int64)
    off = bank.csr["offsets"]
    for i, r in enumerate(rows):
        sl = slice(int(off[r]), int(off[r + 1]))
        s, y, lo = bank.csr["score"][sl], bank.csr["y"][sl], bank.csr["look"][sl]
        entered = np.zeros(s.size, dtype=bool)
        for l in range(L):
            Rl = int(bank.cum_R[r, l, lam_idx[l]])
            if Rl > 0:
                pref = np.flatnonzero((lo >= 0) & (lo <= l))   # already in score-desc order
                thr = s[pref[Rl - 1]]
                entered |= (lo >= 0) & (lo <= l) & (s >= thr)
            D[i, l] = int(entered.sum())
            V[i, l] = int(np.sum(entered & (y != 1)))
            T[i, l] = int(np.sum(entered & (y == 1)))
    return {"D": D, "V": V, "T": T}


def stream_metrics(D: np.ndarray, V: np.ndarray, T: np.ndarray, n_true: np.ndarray,
                   epochs=LOOK_EPOCHS, stop_rules: dict[str, Any] | None = None) -> dict[str, float]:
    """sup-FDP, FDR at fixed and data-dependent looks, time-to-first-true-claim (epochs), recall."""
    fdp = V / np.maximum(D, 1)
    out = {"sup_fdp": float(np.mean(fdp.max(axis=1))), "fdr_end": float(np.mean(fdp[:, -1]))}
    ep = np.asarray(epochs, dtype=np.float64)
    for name, l in {"fdr_15min": 2, "fdr_30min": 3, "fdr_60min": 4}.items():
        if l < fdp.shape[1]:
            out[name] = float(np.mean(fdp[:, l]))
    # data-dependent stopping: first look with >= 3 cumulative claims (else the last look)
    hit3 = D >= 3
    tau3 = np.where(hit3.any(1), np.argmax(hit3, axis=1), D.shape[1] - 1)
    out["fdr_first3"] = float(np.mean(fdp[np.arange(D.shape[0]), tau3]))
    hit1 = D >= 1
    tau1 = np.where(hit1.any(1), np.argmax(hit1, axis=1), D.shape[1] - 1)
    out["fdr_first1"] = float(np.mean(fdp[np.arange(D.shape[0]), tau1]))
    for name, fn in (stop_rules or {}).items():
        tau = fn(D, V, T)
        out[f"fdr_{name}"] = float(np.mean(fdp[np.arange(D.shape[0]), tau]))
    anyT = T >= 1
    first_true = np.where(anyT.any(1), np.argmax(anyT, axis=1), -1)
    got = first_true >= 0
    out["frac_with_true_claim"] = float(got.mean())
    out["tte_epochs_mean"] = float(ep[first_true[got]].mean()) if got.any() else float("nan")
    out["recall_end"] = float(T[:, -1].sum() / max(int(np.sum(n_true)), 1))
    out["claims_end"] = float(D[:, -1].mean())
    return out


def evaluate_stream_split(bank: StreamBank, cal_rows: np.ndarray, test_rows: np.ndarray, alpha: float,
                          theta_idx: np.ndarray, gamma=GAMMA_LOOKS, order_seed: int = 0) -> dict[str, dict]:
    """One split: e-LOND per incident and pooled, CRC-looks, naive repeated CRC, offline CRC at end."""
    gamma = np.asarray(gamma, dtype=np.float64)
    L = bank.n_looks
    Rb, Vb, Qb = q_table(bank, theta_idx)
    E = group_evalues(Qb[cal_rows])
    rng = np.random.default_rng(order_seed)
    test_rows = np.asarray(test_rows)[rng.permutation(len(test_rows))]   # exogenous case order
    Tb = bank.batch_TP[:, np.arange(L), theta_idx].astype(np.int64)
    nt = bank.n_true[test_rows]
    res = {}
    for name, pooled in (("elond", False), ("elond_pooled", True)):
        pth = elond_paths(Rb[test_rows], Vb[test_rows], Tb[test_rows], E, alpha, gamma, pooled)
        res[name] = stream_metrics(pth["D"], pth["V"], pth["T"], nt)
        res[name]["n_looks_certified"] = float(pth["cert"].sum(axis=1).mean())
        if pooled:   # gateway stream sup-FDP over the whole emitted claim sequence
            Dg = np.cumsum(pth["D"][:, -1])
            Vg = np.cumsum(pth["V"][:, -1])
            res[name]["gateway_fdp_end"] = float(Vg[-1] / max(Dg[-1], 1))
    for name, alphas in (("crc_looks", alpha * gamma), ("crc_naive_repeated", np.full(L, alpha))):
        lam = crc_looks_index(bank, cal_rows, alphas)
        u = union_paths(bank, test_rows, lam)
        res[name] = stream_metrics(u["D"], u["V"], u["T"], nt)
    lam_end = crc_looks_index(bank, cal_rows, np.full(L, alpha))[-1]
    lam_off = np.zeros(L, np.int64)
    lam_off[-1] = lam_end
    u = union_paths(bank, test_rows, lam_off)
    res["crc_offline_end"] = stream_metrics(u["D"], u["V"], u["T"], nt)
    res["_schedule"] = {"E": E.tolist(), "theta_idx": np.asarray(theta_idx).tolist(),
                        "Qbar_cal": Qb[cal_rows].mean(axis=0).tolist(),
                        "certify_if_D0_Qbar_le": (alpha * gamma - (1 - alpha * gamma) / len(cal_rows)).tolist()}
    return res
