"""edwards25519 group arithmetic from RFC 8032 §5.1 (pure Python ints, extended coordinates).

Used only by the ECVRF (crypto/ecvrf.py); Ed25519 *signatures* use ``cryptography``.

* Points are tuples (X, Y, Z, T) with x = X/Z, y = Y/Z, x·y = T/Z (RFC 8032 §5.1.4).
* ``add``/``double`` are the RFC 8032 §5.1.4 formulas (complete on edwards25519).
* ``encode`` is §5.1.2; ``decode`` is §5.1.3 and returns None for any invalid string, including
  a non-canonical y ≥ p and x = 0 with the sign bit set.
* ``expand_secret`` is §5.1.5: h = SHA-512(SK); clamp h[0:32] → scalar x; prefix = h[32:64];
  PK = encode(x·B).

Constant time is NOT a goal (research prototype; the gateway key never leaves the process).
"""
from __future__ import annotations

import hashlib

p = 2 ** 255 - 19
q = 2 ** 252 + 27742317777372353535851937790883648493          # prime order L of B
d = (-121665 * pow(121666, p - 2, p)) % p
_d2 = (2 * d) % p
SQRT_M1 = pow(2, (p - 1) // 4, p)
COFACTOR = 8

Point = tuple[int, int, int, int]
IDENTITY: Point = (0, 1, 1, 0)

__all__ = ["p", "q", "d", "COFACTOR", "IDENTITY", "B", "add", "double", "neg", "sub",
           "scalar_mult", "base_mult", "double_scalar_mult", "mul_by_cofactor", "is_identity",
           "point_equal", "encode", "decode", "recover_x", "expand_secret", "public_key",
           "on_curve", "to_affine"]


def recover_x(y: int, sign: int) -> int | None:
    """x from y and the sign bit (RFC 8032 §5.1.3 steps 2–4); None if no valid x."""
    if y >= p:
        return None
    y2 = y * y % p
    u = (y2 - 1) % p
    v = (d * y2 + 1) % p
    v3 = v * v % p * v % p
    v7 = v3 * v3 % p * v % p
    x = u * v3 % p * pow(u * v7 % p, (p - 5) // 8, p) % p
    vx2 = v * x % p * x % p
    if vx2 == u:
        pass
    elif vx2 == (p - u) % p:
        x = x * SQRT_M1 % p
    else:
        return None
    if x == 0 and sign == 1:
        return None
    if (x & 1) != sign:
        x = p - x
    return x


_By = 4 * pow(5, p - 2, p) % p
_Bx = recover_x(_By, 0)
B: Point = (_Bx, _By, 1, _Bx * _By % p)


def add(P: Point, Q: Point) -> Point:
    X1, Y1, Z1, T1 = P
    X2, Y2, Z2, T2 = Q
    A = (Y1 - X1) * (Y2 - X2) % p
    Bv = (Y1 + X1) * (Y2 + X2) % p
    C = T1 * _d2 % p * T2 % p
    D = Z1 * 2 * Z2 % p
    E = Bv - A
    F = D - C
    G = D + C
    H = Bv + A
    return (E * F % p, G * H % p, F * G % p, E * H % p)


def double(P: Point) -> Point:
    X1, Y1, Z1, _ = P
    A = X1 * X1 % p
    Bv = Y1 * Y1 % p
    C = 2 * Z1 * Z1 % p
    H = A + Bv
    s = X1 + Y1
    E = H - s * s % p
    G = A - Bv
    F = C + G
    return (E * F % p, G * H % p, F * G % p, E * H % p)


def neg(P: Point) -> Point:
    X, Y, Z, T = P
    return ((-X) % p, Y, Z, (-T) % p)


def sub(P: Point, Q: Point) -> Point:
    return add(P, neg(Q))


def _window_table(P: Point, w: int = 4) -> list[Point]:
    tbl = [IDENTITY, P]
    for _ in range(2, 1 << w):
        tbl.append(add(tbl[-1], P))
    return tbl


def scalar_mult(k: int, P: Point) -> Point:
    """k·P for any non-negative integer k (not reduced mod q: P may lie outside <B>)."""
    if k < 0:
        return scalar_mult(-k, neg(P))
    if k == 0:
        return IDENTITY
    tbl = _window_table(P)
    nib = (k.bit_length() + 3) // 4
    R = IDENTITY
    for i in range(nib - 1, -1, -1):
        R = double(double(double(double(R))))
        v = (k >> (4 * i)) & 15
        if v:
            R = add(R, tbl[v])
    return R


_BASE_TABLE: list[list[Point]] | None = None


def _base_table() -> list[list[Point]]:
    """table[i][j] = j·16^i·B for i < 64, j < 16 (fixed-base comb, built once)."""
    global _BASE_TABLE
    if _BASE_TABLE is None:
        tbl = []
        P = B
        for _ in range(64):
            row = _window_table(P)
            tbl.append(row)
            P = double(double(double(double(P))))
        _BASE_TABLE = tbl
    return _BASE_TABLE


def base_mult(k: int) -> Point:
    """k·B. Uses the fixed-base table for 0 ≤ k < 2^256, generic ladder otherwise."""
    if not 0 <= k < (1 << 256):
        return scalar_mult(k, B)
    tbl = _base_table()
    R = IDENTITY
    i = 0
    while k:
        v = k & 15
        if v:
            R = add(R, tbl[i][v])
        k >>= 4
        i += 1
    return R


def double_scalar_mult(a: int, P: Point, b: int, Q: Point) -> Point:
    """a·P + b·Q (Straus, 4-bit windows); a, b ≥ 0."""
    if a < 0:
        a, P = -a, neg(P)
    if b < 0:
        b, Q = -b, neg(Q)
    tp, tq = _window_table(P), _window_table(Q)
    nib = (max(a.bit_length(), b.bit_length()) + 3) // 4
    R = IDENTITY
    for i in range(nib - 1, -1, -1):
        R = double(double(double(double(R))))
        va = (a >> (4 * i)) & 15
        vb = (b >> (4 * i)) & 15
        if va:
            R = add(R, tp[va])
        if vb:
            R = add(R, tq[vb])
    return R


def mul_by_cofactor(P: Point) -> Point:
    return double(double(double(P)))


def point_equal(P: Point, Q: Point) -> bool:
    X1, Y1, Z1, _ = P
    X2, Y2, Z2, _ = Q
    return (X1 * Z2 - X2 * Z1) % p == 0 and (Y1 * Z2 - Y2 * Z1) % p == 0


def is_identity(P: Point) -> bool:
    X, Y, Z, _ = P
    return X % p == 0 and (Y - Z) % p == 0


def to_affine(P: Point) -> tuple[int, int]:
    X, Y, Z, _ = P
    zi = pow(Z, p - 2, p)
    return X * zi % p, Y * zi % p


def on_curve(P: Point) -> bool:
    x, y = to_affine(P)
    return (-x * x + y * y - 1 - d * x * x % p * y * y) % p == 0


def encode(P: Point) -> bytes:
    """RFC 8032 §5.1.2: 32-byte little-endian y with the sign of x in bit 255."""
    x, y = to_affine(P)
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def decode(s: bytes) -> Point | None:
    """RFC 8032 §5.1.3; None if invalid (wrong length, y ≥ p, non-square, x = 0 with sign 1)."""
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    if y >= p:
        return None
    x = recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % p)


def expand_secret(sk: bytes) -> tuple[int, bytes]:
    """RFC 8032 §5.1.5: (clamped scalar x, prefix = SHA-512(SK)[32:64])."""
    if len(sk) != 32:
        raise ValueError("secret key must be 32 bytes")
    h = hashlib.sha512(sk).digest()
    a = bytearray(h[:32])
    a[0] &= 248
    a[31] &= 127
    a[31] |= 64
    return int.from_bytes(a, "little"), h[32:]


def public_key(sk: bytes) -> bytes:
    x, _ = expand_secret(sk)
    return encode(base_mult(x))
