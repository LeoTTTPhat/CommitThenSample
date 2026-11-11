"""Epoch header Hdr_e with the hiding state commitment C^st_e (paper §IV-B eq. (cst); Thm. 2(d)).

Public header (published with σ_e and ω_e):

    Hdr_e = (ver="CTS/v1", gw_id, e, t_e, T, n_e, n_e⁺, Σ_{U⁺_e} s_i, Σ_{U_e} s_i, R_e, H(metadata store_e),
             H(θ), H(policy), τ^φ, C^st_e, r*_e, H(Hdr_{e−1}))
    H(Hdr) = BLAKE3("CTS/v1/hdr" ‖ enc(Hdr));   σ_e = Ed25519(sk_G, H(Hdr))

Everything public is pre-β and a function of traffic volume or pre-registered (τ^φ, H(θ), H(policy)).
rev1: H(store_e) also commits the per-unit selection thresholds τ_i (protocol/store.py: H(store_e) =
BLAKE3("CTS/v1/store2" ‖ BLAKE3(meta_e) ‖ BLAKE3(tau_e))), so the header binds them before β; the layout and the
272-byte size are unchanged.
The design state, which under the adaptive arm (realised-cost controller, follow-the-evidence boost)
depends on earlier SELECTIONS, is hidden in a salted commitment:

    C^st_e = BLAKE3("CTS/v1/st" ‖ salt_e ‖ u64 Λ^A_e ‖ u64 Λ^S_e ‖ i64 D_{e−1} ‖ u8 fm_e ‖ enc(B_e))
    enc(B_e) = u32 |B_e| ‖ |B_e| × u32 device_id (sorted, unique) ‖ H_boost (32 B)
    salt_e   = PRF_{k_st}(e) = BLAKE3-keyed(k_st, "CTS/v1 state salt" ‖ u64le e)        (32 B)

H_boost is the policy's digest of the boost set and its justification records
(``policy.boost_digest``), so the justification records are bound too. Binding: BLAKE3 collision
resistance; hiding: salt_e is a PRF output under the gateway's long-term state key k_st (assumption
(K)); it is derived deterministically, so the gateway can re-open any epoch at disclosure time
without storing salts. A disclosure carries the opening (salt_e, state_e) of the requested epochs; V4
checks it. ``OpenedHeader`` = public header + opening, exposing the former header attribute names
(Lam_A, Lam_S, D_prev, floor_mode, H_boost, boost_set) for ThresholdPolicy.state_from_commit /
compliance.

enc(Hdr): fixed width, little-endian, 272 bytes, in this order:
    ver 8s | gw_id 16s | e u32 | t_e i64 (µs) | T u32 (µs) | n_e u32 | n_e⁺ u32 | Σ_{U⁺}s u64 | Σ_U s u64 |
    R_e 32s | H_store 32s | H_theta 32s | H_policy 32s | tau_phi u64 | C_st 32s | r_star u64 | H_prev 32s
H_prev of the first epoch of a chain is 32 zero bytes (``GENESIS_PREV``).
"""
from __future__ import annotations

import struct
from dataclasses import asdict, dataclass, field, replace

from blake3 import blake3

from ..crypto.signing import SigningKey, verify as ed_verify

DOM_HDR = b"CTS/v1/hdr"
DOM_ST = b"CTS/v1/st"
SALT_INFO = b"CTS/v1 state salt"
VER = b"CTS/v1"
GENESIS_PREV = bytes(32)
_FMT = struct.Struct("<8s16sIqIIIQQ32s32s32s32sQ32sQ32s")
HDR_LEN = _FMT.size                      # 272
_ST = struct.Struct("<QQqB")

__all__ = ["Header", "StateOpening", "OpenedHeader", "state_salt", "DOM_HDR", "DOM_ST", "VER", "GENESIS_PREV",
           "HDR_LEN", "sign_header", "verify_header", "gw_id_from_pk"]


def gw_id_from_pk(pk_G: bytes) -> bytes:
    """A 16-byte gateway id derived from its public key."""
    return blake3(b"CTS/v1/gwid" + pk_G).digest()[:16]


def state_salt(k_st: bytes, e: int) -> bytes:
    """salt_e = PRF_{k_st}(e) = BLAKE3-keyed(k_st, "CTS/v1 state salt" ‖ u64le e)."""
    if len(k_st) != 32:
        raise ValueError("state key k_st must be 32 bytes")
    return blake3(SALT_INFO + struct.pack("<Q", e), key=k_st).digest()


@dataclass(frozen=True)
class StateOpening:
    """Opening (salt_e, state_e) of C^st_e."""
    salt: bytes
    Lam_A: int
    Lam_S: int
    D_prev: int
    floor_mode: int
    boost_set: tuple[int, ...] = ()
    H_boost: bytes = bytes(32)

    def __post_init__(self):
        object.__setattr__(self, "boost_set", tuple(sorted(set(int(d) for d in self.boost_set))))

    def enc(self) -> bytes:
        """enc(state_e) = u64 Λ^A ‖ u64 Λ^S ‖ i64 D_{e−1} ‖ u8 fm ‖ enc(B_e)."""
        if len(self.H_boost) != 32:
            raise ValueError("H_boost must be 32 bytes")
        devs = self.boost_set
        return (_ST.pack(self.Lam_A, self.Lam_S, self.D_prev, self.floor_mode) + struct.pack("<I", len(devs))
                + b"".join(struct.pack("<I", d) for d in devs) + self.H_boost)

    def commitment(self) -> bytes:
        """C^st_e = BLAKE3("CTS/v1/st" ‖ salt_e ‖ enc(state_e))."""
        if len(self.salt) != 32:
            raise ValueError("salt must be 32 bytes")
        return blake3(DOM_ST + self.salt + self.enc()).digest()

    def replace(self, **kw) -> "StateOpening":
        return replace(self, **kw)

    @property
    def size_bytes(self) -> int:
        return 32 + _ST.size + 4 + 4 * len(self.boost_set) + 32


@dataclass(frozen=True)
class Header:
    gw_id: bytes
    e: int
    t_e_us: int
    T_us: int
    n: int
    n_pay: int
    sum_s_pay: int
    sum_s: int
    R: bytes
    H_store: bytes
    H_theta: bytes
    H_policy: bytes
    tau_phi: int
    C_st: bytes
    r_star: int
    H_prev: bytes
    ver: bytes = VER

    def encode(self) -> bytes:
        if len(self.gw_id) != 16:
            raise ValueError("gw_id must be 16 bytes")
        for name in ("R", "H_store", "H_theta", "H_policy", "C_st", "H_prev"):
            if len(getattr(self, name)) != 32:
                raise ValueError(f"{name} must be 32 bytes")
        if not 0 <= self.tau_phi < (1 << 64):
            raise ValueError("tau_phi must fit u64 (phi0 < 1)")
        return _FMT.pack(self.ver.ljust(8, b"\x00"), self.gw_id, self.e, self.t_e_us, self.T_us, self.n,
                         self.n_pay, self.sum_s_pay, self.sum_s, self.R, self.H_store, self.H_theta,
                         self.H_policy, self.tau_phi, self.C_st, self.r_star, self.H_prev)

    @classmethod
    def decode(cls, b: bytes) -> "Header":
        if len(b) != HDR_LEN:
            raise ValueError(f"header must be {HDR_LEN} bytes")
        (ver, gw, e, te, T, n, npay, ssp, ss, R, Hs, Ht, Hp, tphi, Cst, rs, Hprev) = _FMT.unpack(b)
        return cls(gw_id=gw, e=e, t_e_us=te, T_us=T, n=n, n_pay=npay, sum_s_pay=ssp, sum_s=ss, R=R, H_store=Hs,
                   H_theta=Ht, H_policy=Hp, tau_phi=tphi, C_st=Cst, r_star=rs, H_prev=Hprev,
                   ver=ver.rstrip(b"\x00"))

    def digest(self) -> bytes:
        """H(Hdr) = BLAKE3("CTS/v1/hdr" ‖ enc(Hdr))."""
        return blake3(DOM_HDR + self.encode()).digest()

    def replace(self, **kw) -> "Header":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.hex() if isinstance(v, bytes) else v) for k, v in d.items()}


_STATE_ATTRS = ("Lam_A", "Lam_S", "D_prev", "floor_mode", "boost_set", "H_boost", "salt")


@dataclass(frozen=True)
class OpenedHeader:
    """Public header + verified state opening; exposes every header attribute and the opened state
    under the former header names (Lam_A, Lam_S, D_prev, floor_mode, H_boost, boost_set)."""
    header: Header
    opening: StateOpening = field(repr=False)

    def __getattr__(self, name: str):
        if name in _STATE_ATTRS:
            return getattr(object.__getattribute__(self, "opening"), name)
        return getattr(object.__getattribute__(self, "header"), name)

    def digest(self) -> bytes:
        return self.header.digest()

    def encode(self) -> bytes:
        return self.header.encode()


def sign_header(sk_G: SigningKey, hdr: Header) -> bytes:
    """σ_e = Ed25519(sk_G, H(Hdr))."""
    return sk_G.sign(hdr.digest())


def verify_header(pk_G: bytes, hdr: Header, sigma: bytes) -> bool:
    try:
        msg = hdr.digest()
    except (ValueError, struct.error):
        return False
    return ed_verify(pk_G, sigma, msg)
