"""ECVRF-EDWARDS25519-SHA512-TAI (RFC 9381, suite_string 0x03), pure Python.

Parameters (RFC 9381 §5.5): suite_string = 0x03; cLen = 16; ptLen = qLen = 32; cofactor 8;
Hash = SHA-512; integers little-endian; secret scalar x and PK as in RFC 8032 §5.1.5;
encode_to_curve = try_and_increment (§5.4.1.1) with encode_to_curve_salt = PK_string;
nonce per §5.4.2.2 (RFC 8032 style); challenge per §5.4.3; decode_proof per §5.4.4;
validate_key per §5.4.5.

API (DESIGN §7.1):
    prove(sk, alpha) -> pi (80 B)
    proof_to_hash(pi) -> beta (64 B)          (raises ValueError if pi does not decode)
    verify(pk, alpha, pi, validate_key=True) -> (ok, beta | None)

``tests/test_ecvrf_vectors.py`` checks PK, x, H, k, U, V, pi and beta against RFC 9381
App. B.3 (Examples 16–18) parsed from the verbatim RFC text.
"""
from __future__ import annotations

import hashlib

from . import ed25519_point as ed

SUITE = b"\x03"
C_LEN = 16
PT_LEN = 32
Q_LEN = 32
PI_LEN = PT_LEN + C_LEN + Q_LEN          # 80
BETA_LEN = 64

__all__ = ["prove", "proof_to_hash", "verify", "verify_detailed", "public_key", "validate_key",
           "encode_to_curve", "decode_proof", "prove_trace", "PI_LEN", "BETA_LEN"]


def _sha512(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()


def public_key(sk: bytes) -> bytes:
    """PK_string = point_to_string(x·B) (identical to the Ed25519 public key of sk)."""
    return ed.public_key(sk)


def encode_to_curve(salt: bytes, alpha: bytes) -> tuple[ed.Point, int]:
    """try_and_increment (RFC 9381 §5.4.1.1); returns (H, ctr)."""
    front, back = b"\x01", b"\x00"
    for ctr in range(256):
        hs = _sha512(SUITE + front + salt + alpha + bytes([ctr]) + back)
        H = ed.decode(hs[:PT_LEN])           # interpret_hash_value_as_a_point
        if H is not None:
            H = ed.mul_by_cofactor(H)
            if not ed.is_identity(H):
                return H, ctr
    raise ValueError("encode_to_curve: no valid point for ctr < 256")


def _nonce(prefix: bytes, h_string: bytes) -> tuple[int, bytes]:
    k_string = _sha512(prefix + h_string)
    return int.from_bytes(k_string, "little") % ed.q, k_string


def _challenge(*point_strings: bytes) -> int:
    c_string = _sha512(SUITE + b"\x02" + b"".join(point_strings) + b"\x00")
    return int.from_bytes(c_string[:C_LEN], "little")


def decode_proof(pi: bytes) -> tuple[ed.Point, int, int] | None:
    """(Gamma, c, s) or None (RFC 9381 §5.4.4: Gamma must decode, s < q)."""
    if len(pi) != PI_LEN:
        return None
    Gamma = ed.decode(pi[:PT_LEN])
    if Gamma is None:
        return None
    c = int.from_bytes(pi[PT_LEN:PT_LEN + C_LEN], "little")
    s = int.from_bytes(pi[PT_LEN + C_LEN:], "little")
    if s >= ed.q:
        return None
    return Gamma, c, s


def prove_trace(sk: bytes, alpha: bytes) -> dict:
    """prove() with every intermediate value (for the RFC vector test)."""
    x, prefix = ed.expand_secret(sk)
    Y = ed.base_mult(x)
    pk = ed.encode(Y)
    H, ctr = encode_to_curve(pk, alpha)
    h_string = ed.encode(H)
    Gamma = ed.scalar_mult(x, H)
    k, k_string = _nonce(prefix, h_string)
    U = ed.base_mult(k)
    V = ed.scalar_mult(k, H)
    g_string, u_string, v_string = ed.encode(Gamma), ed.encode(U), ed.encode(V)
    c = _challenge(pk, h_string, g_string, u_string, v_string)
    s = (k + c * x) % ed.q
    pi = g_string + c.to_bytes(C_LEN, "little") + s.to_bytes(Q_LEN, "little")
    return {"x": x, "PK": pk, "H": h_string, "ctr": ctr, "k_string": k_string, "k": k,
            "U": u_string, "V": v_string, "Gamma": g_string, "c": c, "s": s, "pi": pi}


def prove(sk: bytes, alpha: bytes) -> bytes:
    """ECVRF_prove (RFC 9381 §5.1): 80-byte proof pi."""
    return prove_trace(sk, alpha)["pi"]


def proof_to_hash(pi: bytes) -> bytes:
    """ECVRF_proof_to_hash (RFC 9381 §5.2): beta = SHA512(0x03 ‖ 0x03 ‖ enc(8·Gamma) ‖ 0x00)."""
    D = decode_proof(pi)
    if D is None:
        raise ValueError("invalid proof")
    return _gamma_to_hash(D[0])


def _gamma_to_hash(Gamma: ed.Point) -> bytes:
    return _sha512(SUITE + b"\x03" + ed.encode(ed.mul_by_cofactor(Gamma)) + b"\x00")


def validate_key(pk: bytes) -> ed.Point | None:
    """ECVRF_validate_key (§5.4.5): Y if it decodes and 8·Y is not the identity, else None."""
    Y = ed.decode(pk)
    if Y is None or ed.is_identity(ed.mul_by_cofactor(Y)):
        return None
    return Y


def verify_detailed(pk: bytes, alpha: bytes, pi: bytes,
                    validate_key: bool = True) -> tuple[bool, bytes | None, str]:
    """(ok, beta | None, reason)."""
    Y = ed.decode(pk) if len(pk) == PT_LEN else None
    if Y is None:
        return False, None, "public key does not decode"
    if validate_key and ed.is_identity(ed.mul_by_cofactor(Y)):
        return False, None, "small-order public key"
    D = decode_proof(pi)
    if D is None:
        return False, None, "proof does not decode (bad length, invalid Gamma or s >= q)"
    Gamma, c, s = D
    H, _ = encode_to_curve(pk, alpha)
    U = ed.double_scalar_mult(s, ed.B, c, ed.neg(Y))          # s·B − c·Y
    V = ed.double_scalar_mult(s, H, c, ed.neg(Gamma))         # s·H − c·Gamma
    c2 = _challenge(pk, ed.encode(H), ed.encode(Gamma), ed.encode(U), ed.encode(V))
    if c2 != c:
        return False, None, "challenge mismatch"
    return True, _gamma_to_hash(Gamma), "ok"


def verify(pk: bytes, alpha: bytes, pi: bytes, validate_key: bool = True) -> tuple[bool, bytes | None]:
    """ECVRF_verify (RFC 9381 §5.3): (True, beta) if VALID else (False, None)."""
    ok, beta, _ = verify_detailed(pk, alpha, pi, validate_key)
    return ok, beta
