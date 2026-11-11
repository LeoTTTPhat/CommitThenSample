"""Anytime-valid confidence sequences for design-based totals (DESIGN §3.9, Prop. 1(b,c), Prop. 2(a)).

Lower CS (range-free; valid under arbitrary withholding with X := 0 on W):
    M^-_e(r) = sum_k w_k exp(eta_k r - eta_k^2 V_e / 2),         r = Qhat_e - Q_e
    r^-_e    = inf{r >= 0 : M^-_e(r) >= 1/alpha_side}
    L_e = Qhat_e - r^-_e,  L*_e = max_{e'<=e} L_e',  L+_e = max(L*_e, S_e)
Upper CS (floor-bounded weights c_i = b_i/pi_i <= cbar; grid eta < 1/cbar):
    W_e(eta) = sum psi(eta c_i) X_i^2 / c_i^2,   psi(l) = -log(1-l) - l
    M^+_e(r) = sum_k w_k exp(eta_k r - W_e(eta_k)),               r = Q_e - Qhat_e
    U+_e = min(Qhat_e + r^+_e, Dcap_e)

alpha_side is the per-side level (alpha/2 for a two-sided alpha; alpha/(2|F|) under a declared
query family). Everything is vectorised over arbitrary leading axes (reps, estimands); grids have
shape (K,) or (..., K) broadcastable against the state.

Coverage at the truth (MC, DESIGN §6 common settings): Q_e < L_e  <=>  M^-_e(Qhat_e - Q_e) >= 1/alpha_side
(M^- is strictly increasing in r), and likewise Q_e > Qhat_e + r^+_e <=> M^+_e(Q_e - Qhat_e) >= 1/alpha_side.
Because Q_e is non-decreasing, a miss of the running max L* implies a miss of some earlier L_e',
so the time-uniform miss of L* (and of L+, as S <= Q surely) equals the time-uniform miss of L.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import log_ndtr

__all__ = [
    "lcs_grid", "ucs_grid", "psi", "lcs_log_eproc", "ucs_log_eproc", "lcs_radius", "ucs_radius",
    "ucs_increment", "LowerCS", "UpperCS", "NormalMixtureLowerCS", "sure_bounds",
    "retro_tighten", "null_ucs_radius", "logsumexp", "nm_log_eproc", "nm_radius",
    # fixed-time (final-epoch) comparators, rev1 (review R1-M3)
    "ft_eta_lower", "ft_eta_upper", "ft_lower", "ft_upper", "ft_lower_miss", "ft_upper_miss",
    "bernstein_floor_kappa", "bernstein_lower", "bernstein_upper", "bernstein_lower_miss",
    "bernstein_upper_miss", "exact_pb_bounds",
]


# --------------------------------------------------------------------------------------------- #
# grids
# --------------------------------------------------------------------------------------------- #
def lcs_grid(V_ref, alpha_side: float = 0.025, K: int = 16, decades: float = 2.0):
    """LCS grid: K points log-spaced over eta* x 10^[-decades, +decades], weights 1/K.

    eta* = sqrt(2 log(1/alpha_side) / V_ref)  (= sqrt(2 log(2/alpha)/V_ref) for alpha_side = alpha/2).
    V_ref may be an array (one grid per estimand) -> etas of shape V_ref.shape + (K,).
    """
    V_ref = np.asarray(V_ref, dtype=float)
    if np.any(~(V_ref > 0)):
        raise ValueError("V_ref must be > 0")
    eta_star = np.sqrt(2.0 * math.log(1.0 / alpha_side) / V_ref)
    etas = eta_star[..., None] * 10.0 ** np.linspace(-decades, decades, K)
    w = np.full(K, 1.0 / K)
    return etas, w


def ucs_grid(cbar, n_low: int = 15, decades: float = 4.0, w0: float = 0.5, eta0_frac: float = 0.999):
    """UCS grid: w0 on eta0 = eta0_frac/cbar (null-evidence optimum) + n_low points over `decades`
    decades below it (eta0 * 10^(-decades*j/n_low), j = 1..n_low) with weights (1-w0)/n_low.

    Defaults: w0 = 1/2 and 15 points with weight 1/30 (DESIGN §3.9). All etas < 1/cbar.
    """
    cbar = np.asarray(cbar, dtype=float)
    if np.any(~(cbar > 0)):
        raise ValueError("cbar must be > 0")
    if not (0 < eta0_frac < 1):
        raise ValueError("eta0_frac must be in (0, 1) so that eta*cbar < 1")
    eta0 = eta0_frac / cbar
    j = np.arange(1, n_low + 1)
    low = eta0[..., None] * 10.0 ** (-decades * j / n_low)
    etas = np.concatenate([eta0[..., None], low], axis=-1)
    w = np.concatenate([[w0], np.full(n_low, (1.0 - w0) / n_low if n_low else 0.0)])
    if n_low == 0 and abs(w0 - 1.0) > 1e-12:
        raise ValueError("with n_low = 0 the single grid point must carry weight w0 = 1")
    return etas, w


# --------------------------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------------------------- #
def psi(lam):
    """psi(l) = -log(1-l) - l for l in [0, 1); +inf for l >= 1 (Fan et al.). Series for small l."""
    lam = np.asarray(lam, dtype=float)
    out = np.full(lam.shape, np.inf)
    ok = lam < 1.0
    small = ok & (np.abs(lam) < 1e-4)
    big = ok & ~small
    with np.errstate(divide="ignore", invalid="ignore"):
        lb = lam[big]
        out[big] = -np.log1p(-lb) - lb
        ls = lam[small]
        out[small] = ls * ls * (0.5 + ls * (1.0 / 3.0 + ls * (0.25 + ls * 0.2)))
    return out


def logsumexp(t, axis=-1):
    t = np.asarray(t, dtype=float)
    m = np.max(t, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s = np.log(np.sum(np.exp(t - m), axis=axis, keepdims=True)) + m
    return np.squeeze(s, axis=axis)


def lcs_log_eproc(r, V, etas, logw):
    """log M^-(r) = logsumexp_k(log w_k + eta_k r - eta_k^2 V/2); r, V broadcast over leading axes."""
    r = np.asarray(r, dtype=float)[..., None]
    V = np.asarray(V, dtype=float)[..., None]
    t = logw + etas * r - 0.5 * etas * etas * V
    return logsumexp(t)


def ucs_log_eproc(r, W, etas, logw):
    """log M^+(r) = logsumexp_k(log w_k + eta_k r - W_k); W has the grid as its last axis."""
    r = np.asarray(r, dtype=float)[..., None]
    t = logw + etas * r - np.asarray(W, dtype=float)
    return logsumexp(t)


def _bisect(fun, hi, iters):
    """Smallest r in [0, hi] with fun(r) True (fun monotone), returned conservatively (upper end)."""
    lo = np.zeros_like(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = fun(mid)
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


def lcs_radius(V, etas, logw, log_thr, iters: int = 60):
    """r^- = inf{r >= 0: M^-(r) >= exp(log_thr)} by vectorised bisection (monotone inversion).

    Upper bracket: the k-th term alone crosses at (log_thr - log w_k)/eta_k + eta_k V/2.
    Returned value is the upper end of the final bracket (conservative)."""
    V = np.asarray(V, dtype=float)
    lt = np.asarray(log_thr, dtype=float)
    hi = np.min((lt[..., None] - logw + 0.5 * etas * etas * V[..., None]) / etas, axis=-1)
    hi = np.broadcast_to(hi, np.broadcast_shapes(hi.shape, V.shape, lt.shape)).copy()
    return _bisect(lambda r: lcs_log_eproc(r, V, etas, logw) >= lt, hi, iters)


def ucs_radius(W, etas, logw, log_thr, iters: int = 60):
    """r^+ = inf{r >= 0: M^+(r) >= exp(log_thr)}; W shape (..., K). +inf where W is +inf for all k."""
    W = np.asarray(W, dtype=float)
    lt = np.asarray(log_thr, dtype=float)
    with np.errstate(invalid="ignore"):
        hi = np.min((lt[..., None] - logw + W) / etas, axis=-1)
    bad = ~np.isfinite(hi)
    hi = np.where(bad, 1.0, hi)
    r = _bisect(lambda r: ucs_log_eproc(r, W, etas, logw) >= lt, hi, iters)
    return np.where(bad, np.inf, r)


def ucs_increment(X, c, etas):
    """sum over the last unit axis of psi(eta_k c_i) X_i^2/c_i^2 -> shape X.shape[:-1] + (K,).

    Only entries with X > 0 contribute (X = 0 contributes exactly 0). ``etas`` (K,) or
    X.shape[:-1] + (K,)."""
    X = np.asarray(X, dtype=float)
    c = np.asarray(c, dtype=float)
    pos = X > 0
    csafe = np.where(pos, c, 1.0)
    ratio2 = np.where(pos, (X / csafe) ** 2, 0.0)          # = (f/b)^2 for a retained hit
    lam = np.asarray(etas)[..., None, :] * csafe[..., None]  # (..., n, K)
    ps = psi(lam)
    with np.errstate(invalid="ignore"):
        term = np.where(pos[..., None], ps * ratio2[..., None], 0.0)
    return term.sum(axis=-2)


# --------------------------------------------------------------------------------------------- #
# stateful CS objects (vectorised over a leading `shape`)
# --------------------------------------------------------------------------------------------- #
def _out(x):
    x = np.asarray(x)
    return x.item() if x.ndim == 0 else x


class LowerCS:
    """Prop. 1(b) / Prop. 2(a) lower confidence sequence (constant-eta mixture).

    update(X_epoch) with X already zeroed on W (unit axis last). ``f_true_epoch_sum`` (optional)
    tracks the truth and the time-uniform miss flag ``miss`` (e-process at the true Q)."""

    def __init__(self, etas, weights, alpha_side: float, shape=()):
        self.etas = np.asarray(etas, dtype=float)
        self.logw = np.log(np.asarray(weights, dtype=float))
        if self.etas.shape[-1] != self.logw.shape[-1]:
            raise ValueError("etas and weights must share the grid axis")
        if np.any(self.etas <= 0):
            raise ValueError("etas must be > 0")
        self.alpha_side = float(alpha_side)
        self.log_thr = math.log(1.0 / alpha_side)
        self.shape = tuple(shape)
        self.Qhat = np.zeros(self.shape)
        self.V = np.zeros(self.shape)
        self.Lstar = np.full(self.shape, -np.inf)
        self.Q = np.zeros(self.shape)
        self.miss = np.zeros(self.shape, dtype=bool)
        self.n_updates = 0

    def update(self, X_epoch, f_true_epoch_sum=None) -> None:
        X = np.asarray(X_epoch, dtype=float)
        self.update_sums(X.sum(axis=-1), (X * X).sum(axis=-1), f_true_epoch_sum)

    def update_sums(self, dQhat, dV, f_true_epoch_sum=None) -> None:
        self.Qhat = self.Qhat + dQhat
        self.V = self.V + dV
        if f_true_epoch_sum is not None:
            self.Q = self.Q + f_true_epoch_sum
            self.miss = self.miss | self.miss_at(self.Q)
        self.n_updates += 1

    def log_eproc_at(self, r):
        return lcs_log_eproc(r, self.V, self.etas, self.logw)

    def eproc_at(self, r):
        """Sum_k w_k exp(eta_k r - eta_k^2 V/2)."""
        with np.errstate(over="ignore"):
            return _out(np.exp(self.log_eproc_at(r)))

    def miss_at(self, Q):
        """True where the LCB exceeds Q, i.e. M^-(Qhat - Q) >= 1/alpha_side."""
        return self.log_eproc_at(self.Qhat - np.asarray(Q, dtype=float)) >= self.log_thr

    def radius(self, iters: int = 60):
        return lcs_radius(self.V, self.etas, self.logw, self.log_thr, iters)

    def current(self, iters: int = 60):
        """L_e = Qhat_e - r^-_e (no running max)."""
        return self.Qhat - self.radius(iters)

    def bound(self, iters: int = 60):
        """Running max L*_e over the epochs at which bound() was called."""
        self.Lstar = np.maximum(self.Lstar, self.current(iters))
        return _out(self.Lstar)


class UpperCS:
    """Prop. 1(c) upper confidence sequence; requires every eta < 1/cbar.

    update(X_epoch, c_epoch): c_i = b_i/pi_i for the epoch's eligible units (X = 0 for unretained).
    Any eligible unit with c_i > cbar voids the statement (``void``; e.g. flood-guard capped
    units); a void UCS never issues and cannot miss."""

    def __init__(self, etas, weights, alpha_side: float, cbar, shape=()):
        self.etas = np.asarray(etas, dtype=float)
        self.cbar = np.asarray(cbar, dtype=float)
        if np.any(self.etas * self.cbar[..., None] >= 1.0):
            raise ValueError("UCS grid requires eta < 1/cbar for every grid point")
        if np.any(self.etas <= 0):
            raise ValueError("etas must be > 0")
        self.logw = np.log(np.asarray(weights, dtype=float))
        self.alpha_side = float(alpha_side)
        self.log_thr = math.log(1.0 / alpha_side)
        self.shape = tuple(shape)
        K = self.etas.shape[-1]
        self.Qhat = np.zeros(self.shape)
        self.W = np.zeros(self.shape + (K,))
        self.Q = np.zeros(self.shape)
        self.miss = np.zeros(self.shape, dtype=bool)
        self.void = np.zeros(self.shape, dtype=bool)
        self.n_updates = 0

    def update(self, X_epoch, c_epoch, f_true_epoch_sum=None) -> None:
        X = np.asarray(X_epoch, dtype=float)
        c = np.asarray(c_epoch, dtype=float)
        viol = (c > self.cbar[..., None] * (1 + 1e-12)).any(axis=-1)
        dW = ucs_increment(X, np.minimum(c, np.broadcast_to(self.cbar[..., None], c.shape)), self.etas)
        self.update_sums(X.sum(axis=-1), dW, f_true_epoch_sum, viol)

    def update_sums(self, dQhat, dW, f_true_epoch_sum=None, void=None) -> None:
        self.Qhat = self.Qhat + dQhat
        self.W = self.W + dW
        if void is not None:
            self.void = self.void | np.asarray(void, dtype=bool)
        if f_true_epoch_sum is not None:
            self.Q = self.Q + f_true_epoch_sum
            self.miss = self.miss | (self.miss_at(self.Q) & ~self.void)
        self.n_updates += 1

    def log_eproc_at(self, r):
        return ucs_log_eproc(r, self.W, self.etas, self.logw)

    def eproc_at(self, r):
        """Sum_k w_k exp(eta_k r - W(eta_k))."""
        with np.errstate(over="ignore"):
            return _out(np.exp(self.log_eproc_at(r)))

    def miss_at(self, Q):
        return self.log_eproc_at(np.asarray(Q, dtype=float) - self.Qhat) >= self.log_thr

    def radius(self, iters: int = 60):
        return ucs_radius(self.W, self.etas, self.logw, self.log_thr, iters)

    def bound(self, Dcap=None, iters: int = 60):
        """U+ = min(Qhat + r^+, Dcap); NaN where void (statement withheld)."""
        U = self.Qhat + self.radius(iters)
        if Dcap is not None:
            U = np.minimum(U, Dcap)
        return _out(np.where(self.void, np.nan, U))


class NormalMixtureLowerCS(LowerCS):
    """[SHOULD] one-sided normal-mixture LCS (Howard et al. 2021, method of mixtures).

    Mixing density over eta >= 0: half-normal with precision rho, which gives the closed form
        M(s, v) = 2 sqrt(rho/(v+rho)) exp(s^2/(2(v+rho))) Phi(s/sqrt(v+rho)),   s = Qhat - Q, v = V.
    Each eta-component is the Prop. 1(b) supermartingale, so the mixture is valid (Fubini).
    rho is tuned for intrinsic time V_ref with Howard et al.'s two-sided rule evaluated at alpha_side:
        rho = V_ref / (2 log(1/alpha_side) + log(1 + 2 log(1/alpha_side))).
    """

    def __init__(self, V_ref, alpha_side: float, shape=(), rho=None):
        L = math.log(1.0 / alpha_side)
        V_ref = np.asarray(V_ref, dtype=float)
        self.rho = np.asarray(rho if rho is not None else V_ref / (2 * L + math.log(1 + 2 * L)), dtype=float)
        super().__init__(np.ones(1), np.ones(1), alpha_side, shape)

    def log_eproc_at(self, r):
        return nm_log_eproc(r, self.V, self.rho)

    def radius(self, iters: int = 80):
        return nm_radius(self.V, self.rho, self.log_thr, iters)


def nm_log_eproc(s, V, rho):
    """log of the one-sided normal mixture  2 sqrt(rho/a) exp(s^2/(2a)) Phi(s/sqrt(a)),  a = V + rho."""
    s = np.asarray(s, dtype=float)
    a = np.asarray(V, dtype=float) + rho
    return math.log(2.0) + 0.5 * np.log(rho / a) + s * s / (2 * a) + log_ndtr(s / np.sqrt(a))


def nm_radius(V, rho, log_thr, iters: int = 80):
    """inf{s >= 0: nm_log_eproc(s) >= log_thr} by bisection.  Phi <= 1 gives root >= sqrt(2a need);
    at s = sqrt(2a(need+1)) + 4 sqrt(a) the mixture exceeds the threshold (log Phi(4) > -1)."""
    a = np.asarray(V, dtype=float) + rho
    need = np.maximum(log_thr - math.log(2.0) - 0.5 * np.log(rho / a), 0.0)
    hi = np.array(np.sqrt(2 * a * (need + 1.0)) + 4.0 * np.sqrt(a), dtype=float)
    return _bisect(lambda r: nm_log_eproc(r, V, rho) >= log_thr, hi, iters)


# --------------------------------------------------------------------------------------------- #
# sure bounds, retrospective tightening, null-regime UCS radius
# --------------------------------------------------------------------------------------------- #
def sure_bounds(Z, f, b, W_mask=None):
    """(S, Dcap): S = sum_{Z=1, not in W} f ;  Dcap = sum [Z f 1(not W) + (1 - Z 1(not W)) b].

    Both hold surely (Prop. 1(d), Prop. 2(b)); sums over the last axis."""
    Z = np.asarray(Z, dtype=bool)
    f = np.asarray(f, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = Z if W_mask is None else Z & ~np.asarray(W_mask, dtype=bool)
    S = np.where(keep, f, 0.0).sum(axis=-1)
    Dcap = np.where(keep, f, b).sum(axis=-1)
    return _out(S), _out(Dcap)


def retro_tighten(U_seq, axis=-1):
    """Retrospective tightening [SHOULD, E8]: for a window that ended at e, min_{e''>=e} U+_{e''}
    is also a valid UCB for Q_e (Q is non-decreasing on the time-uniform good event).
    Input: U+ sequence along ``axis`` (NaN = not issued, ignored); output: reverse cumulative min."""
    U = np.asarray(U_seq, dtype=float)
    U = np.where(np.isnan(U), np.inf, U)
    rev = np.flip(U, axis=axis)
    out = np.flip(np.minimum.accumulate(rev, axis=axis), axis=axis)
    return np.where(np.isinf(out), np.nan, out)


def null_ucs_radius(cbar: float, alpha_side: float = 0.025, **grid_kw) -> float:
    """r^+ with no retained hit (W = 0): the UCS null slack of Cor. 1 (<= cbar log(1/(alpha w0))/0.999)."""
    etas, w = ucs_grid(cbar, **grid_kw)
    return float(ucs_radius(np.zeros(etas.shape[-1]), etas, np.log(w), math.log(1.0 / alpha_side)))


# --------------------------------------------------------------------------------------------- #
# fixed-time (final-epoch) comparators -- rev1, review R1-M3 / fix list M2
# --------------------------------------------------------------------------------------------- #
# Valid at ONE pre-specified epoch (the end of the estimand's window), not uniformly over time.
# l := log(1/alpha_side) throughout.
#
# (1) Fixed-time Chernoff LCB (range-free; the Prop. 1(b) supermartingale with a SINGLE eta, evaluated once):
#         M_E(eta) = exp(eta (Qhat_E - Q_E) - eta^2 V_E / 2),   E[M_E] <= 1   (Prop. 1(b); Prop. 2(a) under X := 0 on W)
#     Markov (instead of Ville): P(M_E >= 1/alpha_side) <= alpha_side, and
#         M_E >= 1/alpha_side  <=>  Q_E <= Qhat_E - l/eta - eta V_E/2 =: L_ft.
#     eta is fixed in advance: eta* = sqrt(2 l / V_ref), the geometric centre of the LCS grid (lcs_grid).
#     Equals lcs_radius / LowerCS with the one-point grid (K = 1, w = 1): no mixture (no log K) and no running max.
# (2) Fixed-time Fan UCB (single eta < 1/cbar; the Prop. 1(c) supermartingale, evaluated once):
#         U_ft = Qhat_E + (l + W_E(eta)) / eta,   W_E(eta) = sum psi(eta c_i) X_i^2 / c_i^2   (as in the UCS)
#     pre-registered eta = min(eta0_frac / cbar, eta*)  (ft_eta_upper; reasons in its docstring).
#     Issued under the UCS rules (indicators; no capped or withheld eligible unit in the window).
# (3) Bernstein with the known design-variance bound (floor phi, f_i <= bbar), for populations fixed in advance:
#     Y_i = X_i - f_i is a martingale difference with Y_i <= f_i (1 - pi_i)/pi_i <= c := bbar (1-phi)/phi and
#     Var(Y_i | past) = f_i^2 (1 - pi_i)/pi_i <= kappa f_i,  kappa := bbar (1 - phi)/phi   (pi_i >= phi by (F));
#     Bennett's lemma + (e^u - 1 - u) <= u^2 / (2 (1 - u/3)) make sum Y sub-gamma(v = kappa Q, c/3) for EVERY
#     predictable design, so P(Qhat - Q >= sqrt(2 kappa Q l) + c l/3) <= alpha_side (Boucheron-Lugosi-Massart
#     Thm 2.3); upper side: f_i - X_i <= f_i <= bbar, same variance.  Inverting in Q gives closed forms
#         L_B = (sqrt(kappa l/2 + Qhat - c l/3) - sqrt(kappa l/2))^2   (0 if Qhat <= c l/3)
#         U_B = (sqrt(kappa l/2) + sqrt(kappa l/2 + Qhat + bbar l/3))^2.
#     The variance proxy is kappa Q (every hit at the floor), the only one valid without knowing f; an empirical
#     Bernstein bound (Maurer-Pontil / Audibert et al.) needs i.i.d. terms, which a finite population under a
#     data-adaptive design does not provide.  Not valid for populations that react to the selections (Q random).
# (4) Exact worst-case Poisson-binomial bounds (indicator totals, static design, population fixed in advance):
#     see exact_pb_bounds.


def ft_eta_lower(V_ref, alpha_side: float = 0.025):
    """eta* = sqrt(2 log(1/alpha_side) / V_ref) -- the geometric centre of the LCS grid (``lcs_grid``)."""
    V_ref = np.asarray(V_ref, dtype=float)
    if np.any(~(V_ref > 0)):
        raise ValueError("V_ref must be > 0")
    return _out(np.sqrt(2.0 * np.log(1.0 / np.asarray(alpha_side, dtype=float)) / V_ref))


def ft_eta_upper(V_ref, cbar, alpha_side: float = 0.025, cap_frac: float = 0.999):
    """Pre-registered single eta of the fixed-time Fan UCB: min(cap_frac / cbar, eta*(V_ref)).

    Why: for a sparse window psi(eta c_i) ~ (eta c_i)^2/2, so W(eta) ~ eta^2 V / 2 and the radius
    (l + W)/eta ~ l/eta + eta V/2 is minimised near eta*(V) -- the same centring as the lower side (V_ref).
    Fan's supermartingale needs eta < 1/cbar and psi(eta cbar) diverges as eta cbar -> 1, so eta is capped at
    the top point of the UCS grid, cap_frac/cbar (= eta0, the null-evidence optimum: with no retained hit the
    radius l/eta is smallest for the largest admissible eta).  Both inputs are fixed before the test stream."""
    cbar = np.asarray(cbar, dtype=float)
    if np.any(~(cbar > 0)):
        raise ValueError("cbar must be > 0")
    if not (0 < cap_frac < 1):
        raise ValueError("cap_frac must be in (0, 1)")
    return _out(np.minimum(cap_frac / cbar, np.asarray(ft_eta_lower(V_ref, alpha_side), dtype=float)))


def ft_lower(Qhat, V, eta, alpha_side: float = 0.025):
    """Fixed-time Chernoff LCB  L = Qhat - log(1/alpha_side)/eta - eta V/2  (closed form; = Qhat - lcs_radius
    with the one-point grid).  Valid at the single pre-specified time it is evaluated (Markov)."""
    Qhat = np.asarray(Qhat, dtype=float)
    V = np.asarray(V, dtype=float)
    eta = np.asarray(eta, dtype=float)
    return _out(Qhat - np.log(1.0 / np.asarray(alpha_side, dtype=float)) / eta - 0.5 * eta * V)


def ft_upper(Qhat, W, eta, alpha_side: float = 0.025):
    """Fixed-time Fan UCB  U = Qhat + (log(1/alpha_side) + W(eta))/eta  (= Qhat + ucs_radius, one-point grid)."""
    Qhat = np.asarray(Qhat, dtype=float)
    W = np.asarray(W, dtype=float)
    eta = np.asarray(eta, dtype=float)
    return _out(Qhat + (np.log(1.0 / np.asarray(alpha_side, dtype=float)) + W) / eta)


def ft_lower_miss(Qhat, V, Q, eta, alpha_side: float = 0.025):
    """Miss flag of ft_lower at the truth: eta (Qhat - Q) - eta^2 V/2 >= log(1/alpha_side)  (<=> Q <= L_ft)."""
    Qhat, V, Q, eta = (np.asarray(x, dtype=float) for x in (Qhat, V, Q, eta))
    return _out(eta * (Qhat - Q) - 0.5 * eta * eta * V >= np.log(1.0 / np.asarray(alpha_side, dtype=float)))


def ft_upper_miss(Qhat, W, Q, eta, alpha_side: float = 0.025):
    """Miss flag of ft_upper at the truth: eta (Q - Qhat) - W(eta) >= log(1/alpha_side)  (<=> Q >= U_ft)."""
    Qhat, W, Q, eta = (np.asarray(x, dtype=float) for x in (Qhat, W, Q, eta))
    return _out(eta * (Q - Qhat) - W >= np.log(1.0 / np.asarray(alpha_side, dtype=float)))


def bernstein_floor_kappa(bbar, phi):
    """kappa = bbar (1 - phi)/phi: Var(X_i | past) <= kappa f_i and X_i - f_i <= kappa for every pi_i >= phi."""
    bbar = np.asarray(bbar, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if np.any(~((phi > 0) & (phi <= 1))):
        raise ValueError("phi must be in (0, 1]")
    return _out(bbar * (1.0 - phi) / phi)


def bernstein_lower(Qhat, kappa, c, alpha_side: float = 0.025):
    """Fixed-time Bernstein LCB with variance proxy kappa Q and increment bound c (see the section comment):
    L_B = inf{q >= 0: Qhat - q < sqrt(2 kappa q l) + c l/3} = (sqrt(kappa l/2 + Qhat - c l/3) - sqrt(kappa l/2))^2,
    0 when Qhat <= c l/3.  Always >= 0 (q = 0 is rejected only when Qhat > c l/3 > 0, which implies Q > 0)."""
    Qhat, kappa, c = (np.asarray(x, dtype=float) for x in (Qhat, kappa, c))
    lt = np.log(1.0 / np.asarray(alpha_side, dtype=float))
    a = kappa * lt / 2.0
    d = Qhat - c * lt / 3.0
    with np.errstate(invalid="ignore"):
        L = (np.sqrt(a + np.maximum(d, 0.0)) - np.sqrt(a)) ** 2
    return _out(np.where(d > 0, L, 0.0))


def bernstein_upper(Qhat, kappa, bbar, alpha_side: float = 0.025):
    """Fixed-time Bernstein UCB: sup{q: q - Qhat < sqrt(2 kappa q l) + bbar l/3}
    = (sqrt(kappa l/2) + sqrt(kappa l/2 + Qhat + bbar l/3))^2."""
    Qhat, kappa, bbar = (np.asarray(x, dtype=float) for x in (Qhat, kappa, bbar))
    lt = np.log(1.0 / np.asarray(alpha_side, dtype=float))
    a = kappa * lt / 2.0
    return _out((np.sqrt(a) + np.sqrt(a + Qhat + bbar * lt / 3.0)) ** 2)


def bernstein_lower_miss(Qhat, Q, kappa, c, alpha_side: float = 0.025):
    """Qhat - Q >= sqrt(2 kappa Q l) + c l/3   (<=> Q <= L_B; probability <= alpha_side)."""
    Qhat, Q, kappa, c = (np.asarray(x, dtype=float) for x in (Qhat, Q, kappa, c))
    lt = np.log(1.0 / np.asarray(alpha_side, dtype=float))
    return _out(Qhat - Q >= np.sqrt(2.0 * kappa * np.maximum(Q, 0.0) * lt) + c * lt / 3.0)


def bernstein_upper_miss(Qhat, Q, kappa, bbar, alpha_side: float = 0.025):
    """Q - Qhat >= sqrt(2 kappa Q l) + bbar l/3   (<=> Q >= U_B; probability <= alpha_side)."""
    Qhat, Q, kappa, bbar = (np.asarray(x, dtype=float) for x in (Qhat, Q, kappa, bbar))
    lt = np.log(1.0 / np.asarray(alpha_side, dtype=float))
    return _out(Q - Qhat >= np.sqrt(2.0 * kappa * np.maximum(Q, 0.0) * lt) + bbar * lt / 3.0)


def exact_pb_bounds(pi_elig, kmax: int, alpha_side: float = 0.025):
    """Exact worst-case Poisson-binomial fixed-time bounds for an INDICATOR total under a STATIC design.

    Setting: Z_i ~ Bernoulli(pi_i) independent with pi fixed in advance (non-adaptive design) and a population
    fixed in advance; the statistic is K = number of retained hits (= S for an indicator).  Which eligible units
    are the Q hits is unknown, so the p-values take the least favourable placement (stochastic monotonicity of
    the Poisson-binomial in each pi_i):
        lower:  p_lo(q, k) = max_{|H| = q} P_H(K >= k) = P(PB(q LARGEST eligible pi) >= k)
                L(k) = min{q >= 0 : p_lo(q, k) > alpha_side}      (n_elig + 1 if none: empty set -> a miss)
        upper:  p_hi(q, k) = max_{|H| = q} P_H(K <= k) = P(PB(q SMALLEST eligible pi) <= k)
                U(k) = max{q <= n_elig : p_hi(q, k) > alpha_side}
    For the true H the p-values are super-uniform, so P(L(K) > Q) <= alpha_side and P(U(K) < Q) <= alpha_side.
    U(0) is the hiding capacity of Thm. 1 up to integer rounding (prod_{q smallest}(1 - pi) > alpha_side).
    L and U are non-decreasing in k, so withholding (K' <= K) keeps L(K') valid; U needs the UCS issuance rules.
    Exact DP over units (truncated at kmax, which is exact for all k <= kmax).
    Returns (L, U): int64 arrays indexed by k = 0..kmax."""
    p = np.sort(np.clip(np.asarray(pi_elig, dtype=float).ravel(), 0.0, 1.0))
    n = len(p)
    kmax = int(kmax)
    if kmax < 0:
        raise ValueError("kmax must be >= 0")
    a = float(alpha_side)
    L = np.full(kmax + 1, n + 1, dtype=np.int64)
    U = np.full(kmax + 1, n, dtype=np.int64)
    # ---- lower: add units in descending pi; tail(k) = P(K >= k) = 1 - cdf(k - 1) ----
    L[0] = 0
    pmf = np.zeros(kmax + 1)
    pmf[0] = 1.0
    k = 1
    for q, pq in enumerate(p[::-1], start=1):
        if k > kmax:
            break
        new = pmf * (1.0 - pq)
        new[1:] += pmf[:-1] * pq
        pmf = new
        cdf = np.cumsum(pmf)
        while k <= kmax and 1.0 - cdf[k - 1] > a:
            L[k] = q
            k += 1
    # ---- upper: add units in ascending pi; U(k) = (first q with P(K <= k) <= alpha) - 1 ----
    pmf = np.zeros(kmax + 1)
    pmf[0] = 1.0
    k = 0
    for q, pq in enumerate(p, start=1):
        new = pmf * (1.0 - pq)
        new[1:] += pmf[:-1] * pq
        pmf = new
        cdf = np.cumsum(pmf)
        while k <= kmax and cdf[k] <= a:
            U[k] = q - 1
            k += 1
        if k > kmax:
            break
    return L, U
