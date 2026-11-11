"""RFC 9381 App. B.3 (ECVRF-EDWARDS25519-SHA512-TAI, Examples 16–18), parsed from the verbatim RFC
text in tests/data/rfc9381_B3.txt (never retyped), plus rejection tests and the RFC 8032 key
derivation cross-check against ``cryptography``."""
from __future__ import annotations

import hashlib
import os
import random
import re
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ctsample.crypto import ecvrf
from ctsample.crypto import ed25519_point as ed

VEC = Path(__file__).parent / "data" / "rfc9381_B3.txt"
VEC_SHA256 = "b4b9f6f04e4fea409892cfbfc8846327fba642769797e32cbb70f4caa87a48e4"

_FIELDS = ("SK", "PK", "alpha", "x", "H", "k_string", "k", "U", "V", "pi", "beta")
_FIELD_RE = re.compile(r"^(SK|PK|alpha|x|H|k_string|k|U|V|pi|beta)\s*=(.*)$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")


def parse_vectors(text: str) -> list[dict]:
    """Parse 'Example NN:' blocks; hex values that wrap across lines are joined."""
    examples: list[dict] = []
    cur: dict | None = None
    field: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"^Example (\d+):$", line)
        if m:
            cur = {"example": int(m.group(1))}
            examples.append(cur)
            field = None
            continue
        if cur is None:
            continue
        if not line:
            field = None
            continue
        m = re.match(r"^try_and_increment succeeded on ctr = (\d+)$", line)
        if m:
            cur["ctr"] = int(m.group(1))
            field = None
            continue
        m = _FIELD_RE.match(line)
        if m:
            field = m.group(1)
            rest = m.group(2)
            if "=" in rest:                       # "U = k*B =" → value after the last '='
                rest = rest.rsplit("=", 1)[1]
            rest = rest.strip()
            if field == "alpha":
                tok = rest.split("(")[0].strip()  # "72 (1 byte)", " (the empty string)"
                cur["alpha"] = tok
                field = None
            else:
                cur[field] = rest
            continue
        if field is not None and _HEX_RE.match(line):
            cur[field] += line
            continue
        raise AssertionError(f"unparsed line: {raw!r}")
    for ex in examples:
        for f in _FIELDS:
            assert f in ex, (ex.get("example"), f)
            assert _HEX_RE.match(ex[f]) or (f == "alpha" and ex[f] == ""), (f, ex[f])
    return examples


@pytest.fixture(scope="module")
def vectors():
    raw = VEC.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == VEC_SHA256, "vector file is not the verbatim copy"
    ex = parse_vectors(raw.decode("ascii"))
    assert [e["example"] for e in ex] == [16, 17, 18]
    return ex


def test_vector_lengths(vectors):
    for ex in vectors:
        assert len(ex["SK"]) == 64 and len(ex["PK"]) == 64
        assert len(ex["x"]) == 64 and len(ex["H"]) == 64 and len(ex["k"]) == 64
        assert len(ex["k_string"]) == 128
        assert len(ex["U"]) == 64 and len(ex["V"]) == 64
        assert len(ex["pi"]) == 160 and len(ex["beta"]) == 128


def test_rfc9381_b3_all_intermediates(vectors):
    for ex in vectors:
        sk = bytes.fromhex(ex["SK"])
        alpha = bytes.fromhex(ex["alpha"])
        tr = ecvrf.prove_trace(sk, alpha)
        assert tr["PK"].hex() == ex["PK"], ex["example"]
        assert ecvrf.public_key(sk).hex() == ex["PK"]
        # x is printed as a 32-byte little-endian string
        assert tr["x"].to_bytes(32, "little").hex() == ex["x"]
        assert tr["ctr"] == ex["ctr"]
        assert tr["H"].hex() == ex["H"]
        assert tr["k_string"].hex() == ex["k_string"]
        assert tr["k"].to_bytes(32, "little").hex() == ex["k"]
        assert tr["U"].hex() == ex["U"]
        assert tr["V"].hex() == ex["V"]
        assert tr["pi"].hex() == ex["pi"]
        pi = bytes.fromhex(ex["pi"])
        assert ecvrf.prove(sk, alpha) == pi
        assert ecvrf.proof_to_hash(pi).hex() == ex["beta"]
        ok, beta = ecvrf.verify(bytes.fromhex(ex["PK"]), alpha, pi, validate_key=True)
        assert ok and beta.hex() == ex["beta"]


def test_rejections(vectors):
    ex = vectors[1]
    pk, alpha, pi = bytes.fromhex(ex["PK"]), bytes.fromhex(ex["alpha"]), bytes.fromhex(ex["pi"])
    assert ecvrf.verify(pk, alpha, pi)[0]
    # every single-byte mutation of the proof is rejected (or fails to decode)
    rng = random.Random(9)
    for pos in list(range(0, 80, 1)):
        b = bytearray(pi)
        b[pos] ^= 1 << rng.randrange(8)
        ok, beta = ecvrf.verify(pk, alpha, bytes(b))
        assert not ok and beta is None, pos
    # wrong alpha
    assert ecvrf.verify(pk, alpha + b"\x00", pi) == (False, None)
    assert ecvrf.verify(pk, b"", pi) == (False, None)
    # proof of another key
    pk_other = bytes.fromhex(vectors[0]["PK"])
    assert ecvrf.verify(pk_other, alpha, pi) == (False, None)
    # wrong lengths
    assert ecvrf.verify(pk, alpha, pi[:-1]) == (False, None)
    assert ecvrf.verify(pk[:31], alpha, pi) == (False, None)


def test_s_ge_q_rejected(vectors):
    ex = vectors[0]
    pk, alpha, pi = bytes.fromhex(ex["PK"]), bytes.fromhex(ex["alpha"]), bytes.fromhex(ex["pi"])
    s = int.from_bytes(pi[48:], "little")
    for s_bad in (s + ed.q, ed.q, (1 << 256) - 1):
        if s_bad >= 1 << 256:
            continue
        bad = pi[:48] + s_bad.to_bytes(32, "little")
        ok, beta, reason = ecvrf.verify_detailed(pk, alpha, bad)
        assert not ok and beta is None and "s >= q" in reason
        assert ecvrf.decode_proof(bad) is None
        with pytest.raises(ValueError):
            ecvrf.proof_to_hash(bad)


def test_invalid_gamma_rejected(vectors):
    ex = vectors[2]
    pk, alpha, pi = bytes.fromhex(ex["PK"]), bytes.fromhex(ex["alpha"]), bytes.fromhex(ex["pi"])
    # y = p (non-canonical) and a y with no square root
    bad_ys = [ed.p.to_bytes(32, "little")]
    for y in range(2, 200):
        if ed.recover_x(y, 0) is None:
            bad_ys.append(y.to_bytes(32, "little"))
            break
    for g in bad_ys:
        assert ed.decode(g) is None
        ok, beta, reason = ecvrf.verify_detailed(pk, alpha, g + pi[32:])
        assert not ok and "Gamma" in reason


def _small_order_encodings() -> list[bytes]:
    """Encodings of the 8 small-order points (orders 1, 2, 4, 8)."""
    out = []
    T8 = None
    for y in range(2, 10 ** 6):                  # find any point P, then 8-torsion via q·P
        x = ed.recover_x(y, 0)
        if x is None:
            continue
        P = (x, y, 1, x * y % ed.p)
        T = ed.scalar_mult(ed.q, P)
        if not ed.is_identity(T) and ed.is_identity(ed.mul_by_cofactor(T)):
            # T is in the torsion subgroup; find a generator of order 8
            if not ed.is_identity(ed.scalar_mult(4, T)):
                T8 = T
                break
    assert T8 is not None
    R = ed.IDENTITY
    for _ in range(8):
        out.append(ed.encode(R))
        R = ed.add(R, T8)
    assert len(set(out)) == 8
    return out


def test_small_order_pk_rejected_with_validate_key(vectors):
    ex = vectors[0]
    alpha, pi = bytes.fromhex(ex["alpha"]), bytes.fromhex(ex["pi"])
    for enc in _small_order_encodings():
        assert ecvrf.validate_key(enc) is None
        ok, beta, reason = ecvrf.verify_detailed(enc, alpha, pi, validate_key=True)
        assert not ok and beta is None and reason == "small-order public key"
        # without validate_key the key is accepted for decoding but this proof still fails
        ok2, _, reason2 = ecvrf.verify_detailed(enc, alpha, pi, validate_key=False)
        assert not ok2 and reason2 != "small-order public key"


def test_prove_verify_random_keys_and_uniqueness():
    rng = random.Random(11)
    for _ in range(5):
        sk = rng.randbytes(32)
        pk = ecvrf.public_key(sk)
        alpha = rng.randbytes(rng.randrange(0, 100))
        pi = ecvrf.prove(sk, alpha)
        assert len(pi) == 80
        ok, beta = ecvrf.verify(pk, alpha, pi)
        assert ok and beta == ecvrf.proof_to_hash(pi) and len(beta) == 64
        assert ecvrf.prove(sk, alpha) == pi          # deterministic


def test_ed25519_pk_derivation_matches_cryptography_1000_seeds():
    rng = random.Random(12345)
    for _ in range(1000):
        seed = rng.randbytes(32)
        want = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
        assert ed.public_key(seed) == want


def test_point_arithmetic_sanity():
    assert ed.on_curve(ed.B)
    assert ed.is_identity(ed.base_mult(ed.q))
    assert ed.is_identity(ed.scalar_mult(ed.q, ed.B))
    rng = random.Random(3)
    for _ in range(20):
        a, b = rng.randrange(1 << 255), rng.randrange(1 << 128)
        P = ed.base_mult(rng.randrange(1, ed.q))
        assert ed.point_equal(ed.base_mult(a), ed.scalar_mult(a, ed.B))
        lhs = ed.double_scalar_mult(a, ed.B, b, P)
        rhs = ed.add(ed.scalar_mult(a, ed.B), ed.scalar_mult(b, P))
        assert ed.point_equal(lhs, rhs)
        assert ed.decode(ed.encode(P)) is not None
        assert ed.point_equal(ed.decode(ed.encode(P)), P)
        assert ed.point_equal(ed.double(P), ed.add(P, P))
    # non-canonical identity encoding y = p + 1 must be rejected; x = 0 with sign bit set too
    assert ed.decode((ed.p + 1).to_bytes(32, "little")) is None
    assert ed.decode((1 | (1 << 255)).to_bytes(32, "little")) is None
    assert ed.decode(os.urandom(31)) is None
