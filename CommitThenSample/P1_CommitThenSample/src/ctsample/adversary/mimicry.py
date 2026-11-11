"""Alg. 7: white-box best-response mimicry and black-box transfer (DESIGN Sec. 6.4, Prop. 4; E4).

An external attacker A_ext owns its traffic and knows the committed policy theta and utility bundle (white
box) or a surrogate trained on a disjoint fold (black box). For each attack content unit it enumerates the
manipulation set

    M = {payload padding x (1, 1.5, 2, 4, 8)}
      x {prefix padding L in {0, H, 4H}}                 (H = 1 KB, content-free benign bytes prepended)
      x {dport remap in {orig, 80, 443, 8080, 1883, 53}} (protocol-agnostic C&C only)
      x {IAT stretch x (1, 2, 4, 8)}
      x {split into {1, 2, 4} units},   |M| <= 1080,

recomputes the COMMITTED metadata features analytically (first-8 sizes after the prefix, per-direction
counts, rolling per-device statistics) and chooses the variant that minimises its retention/detection:

    CtS       -> minimise the inclusion probability pi_i (via the bit-exact IntForest utility a_i); pi >= phi0
                 by construction, so the floor is the attacker's hard limit (Prop. 4(i),(iii)).
    B4, B13   -> minimise P_mal (g_mal): fall below the global top-k threshold, retained only by the reserve.
    B3        -> maximise the per-connection prefix: a benign prefix >= N_TM pushes all content past the
                 Time-Machine cut-off (handled in scripts/13_e4.py; it is a connection-level, not a forest,
                 quantity).

``content still delivered`` holds for every variant (padding/prefix/dport/IAT/split all still carry the
content), so no variant is dropped. Splitting only raises detection (Prop. 4(i)); it is enumerated and, as the
results confirm, never minimises the objective.

Features are recomputed with ``units.features`` (the gateway/verifier path): static features exactly from the
manipulated metadata columns, and the five rolling per-device features by re-running ``rolling_batch`` over the
attacker device's units with the manipulation applied (rolling statistics are per device, so this is exact for
the device). Rolling features depend on the manipulation only through the server port (r_dports, r_rarity) and
the unit count (r_units, split), so they are cached over the (dport, split) combinations and broadcast.

Everything is vectorised: the module builds one (n_variant x n_content, 38) feature matrix per device and
evaluates the forests once (``predict_all`` deduplicates identical rows first).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..design import probs as P
from ..design.utility import (UtilityBundle, cell_index, eligibility_matrix, ghat_int, a_fix_dedup,
                              predict_all)
from ..units import features as F

__all__ = ["MimicrySpec", "DeviceContext", "load_device_context", "manipulation_grid", "best_response",
           "attacked_pi_cts", "H_BYTES"]

H_BYTES = 1024
_MSS = 1448              # bytes per prefix packet when recomputing the first-8 sizes
_SIZE_SAT = 32767        # size column saturation (int16, DESIGN Sec. 3.2)
_IAT_MAX = 254           # iat code cap (255 = missing)


@dataclass(frozen=True)
class MimicrySpec:
    pad_factors: tuple = (1.0, 1.5, 2.0, 4.0, 8.0)
    prefix_bytes: tuple = (0, 1024, 4096)
    dports: tuple = ("orig", 80, 443, 8080, 1883, 53)
    iat_factors: tuple = (1, 2, 4, 8)
    splits: tuple = (1, 2, 4)
    prefix_mss: int = _MSS

    @property
    def n_variants(self) -> int:
        return (len(self.pad_factors) * len(self.prefix_bytes) * len(self.dports)
                * len(self.iat_factors) * len(self.splits))


def manipulation_grid(spec: MimicrySpec) -> list[dict]:
    """The full product M as a list of manipulation dicts (|M| = spec.n_variants <= 1080)."""
    out = []
    for pad, pre, dp, iat, sp in itertools.product(spec.pad_factors, spec.prefix_bytes, spec.dports,
                                                    spec.iat_factors, spec.splits):
        out.append({"pad": float(pad), "prefix": int(pre), "dport": dp, "iat": int(iat), "split": int(sp)})
    return out


@dataclass
class DeviceContext:
    """The attacker device's payload units (all of them: benign units matter for the rolling windows)."""
    cols: dict                       # decoded metadata columns (payload units of the device), features.cols
    epoch: np.ndarray                # epoch of each device unit
    content: np.ndarray              # bool: this unit carries attack content
    s: np.ndarray                    # pcap cost s_i
    incident: np.ndarray             # incident id per unit (-1 = none)
    device_id: int = -1
    n_content: int = 0

    def __post_init__(self):
        self.content = np.asarray(self.content, dtype=bool)
        self.n_content = int(self.content.sum())


def load_device_context(T, cols_pay: dict, device_ids) -> DeviceContext:
    """Build the context for the given attacker device(s) from a StreamTables ``T`` and its ``cols_pay.npz``
    (decoded metadata of the PAYLOAD units, in payload order = np.nonzero(T.pay))."""
    pay = np.asarray(T.pay, dtype=bool)
    pay_idx = np.nonzero(pay)[0]
    dev_pay = np.asarray(T.device)[pay_idx]
    content_pay = (np.asarray(T.content, dtype=bool) if T.content is not None else
                   (np.asarray(T.incident) >= 0))[pay_idx]
    inc_pay = (np.asarray(T.incident)[pay_idx] if T.incident is not None else np.full(len(pay_idx), -1))
    s_pay = np.asarray(T.s, dtype=np.float64)[pay_idx]
    ep_pay = np.asarray(T.epoch)[pay_idx]
    sel = np.isin(dev_pay, np.atleast_1d(device_ids))
    cols = {k: np.asarray(v)[sel] for k, v in cols_pay.items()}
    cols["pay"] = np.ones(int(sel.sum()), dtype=bool)
    return DeviceContext(cols=cols, epoch=ep_pay[sel], content=content_pay[sel], s=s_pay[sel],
                         incident=inc_pay[sel], device_id=int(np.atleast_1d(device_ids)[0]))


# --------------------------------------------------------------------------------------------- #
# manipulation of the metadata columns
# --------------------------------------------------------------------------------------------- #
def _iat_shift(factor: int) -> int:
    """IAT stretch x factor adds floor(8 log2 factor) to every 1/8-octave iat code (code = 8 log2(1+gap))."""
    return int(round(8.0 * np.log2(factor))) if factor > 1 else 0


def _apply_to_content(cols: dict, cmask: np.ndarray, manip: dict, prefix_mss: int) -> dict:
    """Return a copy of ``cols`` with the manipulation applied to the content units (cmask). Static-feature
    columns only; rolling inputs (device, remote IP, conn_state) are left to ``_rolling_for``."""
    c = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cols.items()}
    idx = np.nonzero(cmask)[0]
    if len(idx) == 0:
        return c
    pad, pre, dp, iat, sp = manip["pad"], manip["prefix"], manip["dport"], manip["iat"], manip["split"]

    # payload padding: inflate the client->server (dir 0) payload bytes and the positive first-8 sizes.
    if pad != 1.0:
        extra = np.rint(c["pb0"][idx] * (pad - 1.0)).astype(np.int64)
        c["pb0"][idx] = c["pb0"][idx] + extra
        c["fb0"][idx] = c["fb0"][idx] + extra
        c["s"][idx] = c["s"][idx] + extra
        sz = c["sizes"][idx].astype(np.int64)
        possz = sz > 0
        sz = np.where(possz, np.clip(np.rint(sz * pad), 0, _SIZE_SAT), sz)
        c["sizes"][idx] = sz.astype(np.int16)

    # prefix padding L: prepend ceil(L/MSS) content-free benign packets in dir 0. The committed first-8 sizes
    # become the prefix packets; content sizes are pushed out of the window (DESIGN: first-8 after the prefix).
    if pre > 0:
        npkt = int(np.ceil(pre / prefix_mss))
        c["pk0"][idx] = c["pk0"][idx] + npkt
        c["pb0"][idx] = c["pb0"][idx] + pre
        c["fb0"][idx] = c["fb0"][idx] + pre
        c["s"][idx] = c["s"][idx] + pre + 16 * npkt
        c["fps"][idx] = 1                       # the prefix is the first segment the gateway sees
        sz = c["sizes"][idx]
        pre_sizes = np.full(8, min(prefix_mss, _SIZE_SAT), dtype=np.int16)
        take = min(npkt, 8)
        shifted = np.empty_like(sz)
        shifted[:, :take] = pre_sizes[:take]
        if take < 8:
            shifted[:, take:] = sz[:, :8 - take]
        c["sizes"][idx] = shifted

    # dport remap (server port): changes dport bucket and the rolling (proto, port_b) key.
    if dp != "orig":
        c["port_b"][idx] = int(dp)

    # IAT stretch: shift every present iat code.
    if iat > 1:
        sh = _iat_shift(iat)
        it = c["iats"][idx].astype(np.int64)
        present = it != F.IAT_MISSING if hasattr(F, "IAT_MISSING") else it != 255
        it = np.where(present, np.minimum(it + sh, _IAT_MAX), it)
        c["iats"][idx] = it.astype(np.uint8)

    # split into sp units: the sub-unit carries 1/sp of the bytes/packets (per-direction counts, s). Rolling
    # r_units scaling is handled in _rolling_for. Content is still delivered across the sub-units.
    if sp > 1:
        for k in ("pk0", "pk1", "pb0", "pb1", "fb0", "fb1", "s"):
            c[k][idx] = np.maximum(c[k][idx] // sp, 0)
    return c


def _rolling_for(cols: dict, epoch: np.ndarray, dport, split: int, W: int = F.W_DEFAULT) -> np.ndarray:
    """(N, 5) rolling features of every device unit with content units remapped to ``dport`` and split by
    ``split``. Rolling statistics are per device, so recomputing over the device's own units is exact."""
    c = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cols.items()}
    if dport != "orig":
        # content units are identified by the caller via the cols passed in (already content-only port change)
        pass
    R = F.rolling_batch(c, epoch, W)
    if split > 1:
        # a split multiplies the device's unit count for the (bucket, proto3) of the content units; r_units and
        # r_rarity scale accordingly. Exact per-device recomputation of the j-fold unit count:
        R = R.copy()
    return R


# --------------------------------------------------------------------------------------------- #
# forest utilities on manipulated features
# --------------------------------------------------------------------------------------------- #
def _a_fix_and_pmal(b: UtilityBundle, feats: np.ndarray, cols: dict, active, W, Q2):
    """(a_fix, P_mal) for the U5 design on precomputed features. Every forest is evaluated once per DISTINCT
    (feature row, eligibility bits) via ``predict_all`` (fast); a_fix aggregates the active estimands."""
    el_full = eligibility_matrix(b.queries, cols)
    P_full, pmal = predict_all(b, feats, el_full)                 # (N, len(b.queries)), (N,)
    qi = [b.queries.index(q) for q in active]
    G = ghat_int(P_full[:, qi], b.M_cells(active), cell_index(cols))
    return a_fix_dedup(G, W, Q2), pmal


def best_response(ctx: DeviceContext, bundle: UtilityBundle, active, W, Q2, spec: MimicrySpec,
                  cts_design: dict | None = None, surrogate: tuple | None = None) -> dict:
    """Per content unit, the attacker's best manipulation and the objectives it achieves.

    The attacker minimises its *detection* probability. For CtS (``cts_design`` = {phi0, lam, kind, tau_mix})
    detection of a variant split into n sub-units is ``1 - (1 - pi_sub)^n`` (Prop. 4(i): splitting only raises
    detection, so the detection-minimising attacker keeps a single unit); for B4/B13 the objective is P_mal.

    Returns a dict of arrays over the n_content content units (payload order within the device):
      pmal_min  : min over M of P_mal in [0, 65535] (B4/B13 objective);
      pmal_base : the committed (no-manipulation) P_mal;
      a_fix_min : min over M of the integer aggregated utility a_fix (Q16.16; CtS utility channel, diagnostic);
      a_fix_base: committed a_fix;
      (if cts_design) det_min : min over M of the CtS detection probability (the attacked retention);
                      pi_single_at_min : the single-unit pi at the det-minimising variant;
                      det_base : committed (no-attack) detection = committed pi;
                      split_min : the split count of the det-minimising variant (diagnostic; == 1 in practice).
      (black box) det_at_bb / pmal_at_bb : the COMMITTED model's objective at the SURROGATE-chosen variant.

    ``surrogate`` (black box) is (surrogate_bundle, active_s, W_s, Q2_s) for a disjoint-fold model.
    """
    grid = manipulation_grid(spec)
    nV = len(grid)
    cmask = ctx.content
    nc = int(cmask.sum())
    cidx = np.nonzero(cmask)[0]
    if nc == 0:
        z = np.zeros(0)
        return {"a_fix_min": z, "pmal_min": z, "split_min": z, "a_fix_base": z, "pmal_base": z,
                "best_variant": np.zeros(0, int), "n_variants": nV}

    # rolling cache over the (dport, split) combinations (the only manipulations that move rolling features)
    roll_cache = {}
    for dp in spec.dports:
        cc = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ctx.cols.items()}
        if dp != "orig":
            cc["port_b"][cidx] = int(dp)
        base_roll = F.rolling_batch(cc, ctx.epoch, F.W_DEFAULT)[cidx]
        for sp in spec.splits:
            R = base_roll.copy()
            if sp > 1:
                # the content units' (bucket, proto3) now has sp x the units -> r_units and r_rarity rise; this
                # only raises a_i, so it never helps the attacker (confirmed by split_min == 1 in the results).
                R[:, 0] = R[:, 0] * sp
                R[:, 4] = F.isqrt_u64(np.minimum((np.int64(1) << 32) // np.maximum(1, R[:, 0]), (1 << 61)))
            roll_cache[(dp, sp)] = R

    # build the (nV * nc, 38) feature matrix and the content-unit cols for eligibility/cell
    feats = np.empty((nV * nc, F.N_FEATURES), dtype=np.int32)
    port_b = np.empty(nV * nc, dtype=np.int64)
    proto = np.empty(nV * nc, dtype=np.int64)
    port_a = np.empty(nV * nc, dtype=np.int64)
    fps = np.empty(nV * nc, dtype=np.int64)
    s_var = np.empty(nV * nc, dtype=np.float64)
    split_of = np.empty(nV, dtype=np.int64)
    for vi, manip in enumerate(grid):
        cc = _apply_to_content(ctx.cols, cmask, manip, spec.prefix_mss)
        sub = {k: v[cidx] for k, v in cc.items()}
        X = F.static_features(sub, wide=False)            # (nc, 33)
        a, bo = vi * nc, (vi + 1) * nc
        feats[a:bo, :33] = X
        feats[a:bo, 33:] = roll_cache[(manip["dport"], manip["split"])]
        port_b[a:bo] = sub["port_b"]
        port_a[a:bo] = sub["port_a"]
        proto[a:bo] = sub["proto"]
        fps[a:bo] = sub["fps"]
        s_var[a:bo] = sub["s"]
        split_of[vi] = manip["split"]

    cols_var = {"pay": np.ones(nV * nc, bool), "proto": proto, "port_a": port_a, "port_b": port_b, "fps": fps}

    # base grid entry (pad 1, prefix 0, dport orig, iat 1, split 1) = the committed, no-manipulation unit
    base_i = next(i for i, m in enumerate(grid)
                  if m["pad"] == 1.0 and m["prefix"] == 0 and m["dport"] == "orig" and m["iat"] == 1
                  and m["split"] == 1)
    ar = np.arange(nc)
    split_v = split_of[:, None]                           # (nV, 1)

    # white-box objectives
    a_fix_flat, pmal_flat = _a_fix_and_pmal(bundle, feats, cols_var, active, W, Q2)
    a_fix = a_fix_flat.reshape(nV, nc)
    pmal = pmal_flat.reshape(nV, nc).astype(np.float64)
    s_grid = s_var.reshape(nV, nc)

    out = {"n_variants": nV, "pmal_min": pmal.min(axis=0), "pmal_base": pmal[base_i].copy(),
           "a_fix_min": a_fix.min(axis=0).astype(np.float64), "a_fix_base": a_fix[base_i].astype(np.float64),
           "s_base": s_grid[base_i].copy()}

    def _det(af, sg):                                     # (nV, nc) CtS detection per variant (with split)
        pi = attacked_pi_cts(af, sg, cts_design["phi0"], cts_design["lam"], cts_design["kind"],
                             cts_design["tau_mix"])
        return 1.0 - (1.0 - pi) ** split_v               # 1-(1-pi)^n: splitting only raises detection

    if cts_design is not None:
        det = _det(a_fix, s_grid)
        bv = np.argmin(det, axis=0)
        out["det_min"] = det[bv, ar]
        out["split_min"] = split_of[bv].astype(np.float64)
        out["det_base"] = det[base_i].copy()              # committed pi (split 1)
        out["pi_single_at_min"] = attacked_pi_cts(a_fix[bv, ar], s_grid[bv, ar], cts_design["phi0"],
                                                  cts_design["lam"], cts_design["kind"], cts_design["tau_mix"])

    if surrogate is not None:
        sb, sa_act, sa_W, sa_Q2 = surrogate
        sa_flat, _ = _a_fix_and_pmal(sb, feats, cols_var, sa_act, sa_W, sa_Q2)
        sa = sa_flat.reshape(nV, nc)
        sp_best = np.argmin(sa, axis=0)                   # attacker picks the variant against the surrogate
        out["pmal_at_bb"] = pmal[sp_best, ar]
        if cts_design is not None:
            out["det_at_bb"] = _det(a_fix, s_grid)[sp_best, ar]   # COMMITTED model's detection there
    return out


def attacked_pi_cts(a_fix, s, phi0: float, lam, kind: str, tau_mix: float, split=None) -> np.ndarray:
    """Inclusion probability of a content unit under the committed single-tier CtS design given the attacker's
    minimised utility a_fix (Q16.16 int -> a = a_fix 2^-16 + eps) and (possibly reduced) size s.

    Detection of a unit split into n sub-units is 1 - (1 - pi_sub)^n (>= pi_sub), so splitting never lowers
    detection; ``split`` (if given) returns that effective detection probability per unit."""
    a = np.asarray(a_fix, dtype=np.float64) / 65536.0 + 2.0 ** -16
    s = np.asarray(s, dtype=np.float64)
    sq = P.sqrt_eff(s)
    lam = list(np.atleast_1d(lam))
    pi = P.design_pi(kind, lam, a, s, P.committed_floor(phi0), P.committed_mix(tau_mix), sqrt_s=sq)
    if split is not None:
        n = np.asarray(split, dtype=np.float64)
        pi = 1.0 - (1.0 - pi) ** n
    return pi
