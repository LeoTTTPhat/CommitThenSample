"""DESIGN §7.4 test_merkle: roots over 1..1,025 leaves; proofs; tamper detection; odd-node
promotion; domain separation; leaf0 recomputation."""
from __future__ import annotations

import os
import random
import struct

import pytest
from blake3 import blake3

from ctsample.crypto import hashing as hs


def _H(b: bytes) -> bytes:
    return blake3(b).digest()


def _mth_rfc6962(leaves: list[bytes]) -> bytes:
    """Recursive RFC 6962 §2.1 MTH with BLAKE3 (reference implementation)."""
    n = len(leaves)
    if n == 0:
        return _H(b"CTS/v1/empty")
    if n == 1:
        return _H(b"\x00" + leaves[0])
    k = 1
    while k * 2 < n:
        k *= 2
    return _H(b"\x01" + _mth_rfc6962(leaves[:k]) + _mth_rfc6962(leaves[k:]))


@pytest.fixture(scope="module")
def leaves1025():
    rng = random.Random(1)
    return [rng.randbytes(32) for _ in range(1025)]


def test_empty_root():
    assert hs.merkle_root([]) == _H(b"CTS/v1/empty") == hs.EMPTY_ROOT


def test_roots_match_rfc6962_reference_1_to_1025(leaves1025):
    for n in range(1, 1026):
        assert hs.merkle_root(leaves1025[:n]) == _mth_rfc6962(leaves1025[:n]), n


def test_small_trees_explicit():
    a, b, c = (bytes([i]) * 32 for i in (1, 2, 3))
    la, lb, lc = (_H(b"\x00" + x) for x in (a, b, c))
    assert hs.merkle_root([a]) == la
    assert hs.merkle_root([a, b]) == _H(b"\x01" + la + lb)
    # odd promotion: the 3rd leaf node is promoted unchanged to level 1
    assert hs.merkle_root([a, b, c]) == _H(b"\x01" + _H(b"\x01" + la + lb) + lc)
    levels = hs.merkle_levels([a, b, c])
    assert levels[1][1] == lc            # promoted, not re-hashed
    assert [len(lv) for lv in levels] == [3, 2, 1]


def test_proofs_all_indices_up_to_67_and_samples_to_1025(leaves1025):
    rng = random.Random(2)
    for n in list(range(1, 68)) + [127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025]:
        L = leaves1025[:n]
        root = hs.merkle_root(L)
        idxs = range(n) if n <= 67 else rng.sample(range(n), 25) + [0, n - 1]
        proofs = hs.merkle_proofs(L, idxs)
        depth = max(1, (n - 1).bit_length())
        for i in idxs:
            p = proofs[i]
            assert p == hs.merkle_proof(L, i)
            assert len(p) <= depth                     # ⌈log₂ n⌉ siblings at most
            assert hs.verify_proof(L[i], i, n, p, root), (n, i)


def test_proof_tamper_detection(leaves1025):
    n = 300
    L = leaves1025[:n]
    root = hs.merkle_root(L)
    rng = random.Random(3)
    for i in rng.sample(range(n), 30):
        p = hs.merkle_proof(L, i)
        # wrong leaf
        bad_leaf = bytearray(L[i]); bad_leaf[rng.randrange(32)] ^= 1
        assert not hs.verify_proof(bytes(bad_leaf), i, n, p, root)
        # wrong index (n itself is authentic: it is the committed n_e of the signed header)
        assert not hs.verify_proof(L[i], (i + 1) % n, n, p, root)
        # flipped sibling bit
        if p:
            k = rng.randrange(len(p))
            q = list(p); b = bytearray(q[k]); b[0] ^= 0x80; q[k] = bytes(b)
            assert not hs.verify_proof(L[i], i, n, q, root)
            # truncated / extended proof
            assert not hs.verify_proof(L[i], i, n, p[:-1], root)
        assert not hs.verify_proof(L[i], i, n, p + [b"\x00" * 32], root)
        # wrong root
        assert not hs.verify_proof(L[i], i, n, p, _H(root))
    # out of range
    assert not hs.verify_proof(L[0], -1, n, [], root)
    assert not hs.verify_proof(L[0], n, n, [], root)
    with pytest.raises(IndexError):
        hs.merkle_proof(L, n)


def test_root_changes_on_any_leaf_change_reorder_or_append(leaves1025):
    L = leaves1025[:101]
    root = hs.merkle_root(L)
    for i in (0, 50, 99, 100):
        M = list(L); b = bytearray(M[i]); b[7] ^= 4; M[i] = bytes(b)
        assert hs.merkle_root(M) != root
    M = list(L); M[3], M[4] = M[4], M[3]
    assert hs.merkle_root(M) != root
    assert hs.merkle_root(L + [L[-1]]) != root          # duplicating the last leaf changes the root
    assert hs.merkle_root(L[:-1]) != root


def test_odd_promotion_is_not_duplication():
    # Bitcoin-style duplication of the odd node would make [a,b,c] and [a,b,c,c] collide.
    a, b, c = (bytes([i]) * 32 for i in (7, 8, 9))
    assert hs.merkle_root([a, b, c]) != hs.merkle_root([a, b, c, c])


def test_frame_leaf_layout_and_batch():
    frame = os.urandom(200)
    ts, d, cl = 1_672_531_200_123_456, 1, 200
    want = _H(b"CTS/v1/frm" + struct.pack("<Q", ts) + bytes([d]) + struct.pack("<I", cl) + _H(frame))
    assert hs.frame_leaf(ts, d, cl, frame) == want
    assert hs.frame_leaf_from_hash(ts, d, cl, _H(frame)) == want
    frames = [os.urandom(60 + i) for i in range(5)]
    tss = [ts + 10 * i for i in range(5)]
    dirs = [i & 1 for i in range(5)]
    cls = [len(f) for f in frames]
    batch = hs.frame_leaves(tss, dirs, cls, frames)
    assert batch == [hs.frame_leaf(t, x, c, f) for t, x, c, f in zip(tss, dirs, cls, frames)]
    root, lv = hs.packet_root(tss, dirs, cls, frames)
    assert lv == batch and root == hs.merkle_root(batch)


def test_unit_leaf_leaf0_flow_uid_layouts():
    meta = os.urandom(128)
    root_pkt = os.urandom(32)
    assert hs.unit_leaf(meta, root_pkt) == _H(b"CTS/v1/leaf" + meta + root_pkt)
    assert hs.leaf0(meta[:56]) == _H(b"CTS/v1/leaf0" + meta[:56])
    key = os.urandom(37)
    assert hs.flow_uid(key, 123) == _H(b"CTS/v1/flow" + key + struct.pack("<Q", 123))[:16]
    assert len(hs.flow_uid(key, 123)) == 16
    with pytest.raises(ValueError):
        hs.unit_leaf(meta, root_pkt[:31])


def test_domain_separation():
    x = os.urandom(32)
    outs = {
        "plain": _H(x),
        "leaf_node": hs.leaf_node(x),
        "inner_node": hs.inner_node(x[:16], x[16:]),
        "leaf0": hs.leaf0(x),
        "unit_leaf": hs.unit_leaf(b"", x),
        "flow": hs.flow_uid(x, 0) + b"\x00" * 16,
        "empty": hs.EMPTY_ROOT,
    }
    assert len(set(outs.values())) == len(outs)
    # the same (meta, root) bytes give different payload vs zero-payload leaves
    meta = os.urandom(96)
    r = os.urandom(32)
    assert hs.unit_leaf(meta, r) != hs.leaf0(meta + r)
    # leaf-node vs inner-node prefixes: a 64-byte leaf cannot pose as an inner node
    l, rr = os.urandom(32), os.urandom(32)
    assert hs.leaf_node(l + rr) != hs.inner_node(l, rr)
    # a one-leaf tree's root is not the leaf itself (second-preimage hardening)
    assert hs.merkle_root([x]) != x


def test_leaf0_recomputation_in_mixed_tree():
    rng = random.Random(5)
    metas, pays, roots, leaves = [], [], [], []
    for i in range(257):
        pay = rng.random() < 0.6
        m = rng.randbytes(128 if pay else 56)
        r = rng.randbytes(32) if pay else None
        metas.append(m); pays.append(pay); roots.append(r)
        leaves.append(hs.unit_leaf(m, r) if pay else hs.leaf0(m))
    R = hs.merkle_root(leaves)
    # the store keeps only meta (+root_pkt for payload units); zero-payload leaves are recomputed
    recomputed = [hs.unit_leaf(m, r) if p else hs.leaf0(m) for m, p, r in zip(metas, pays, roots)]
    assert hs.merkle_root(recomputed) == R
    # presenting a payload unit as zero-payload (or vice versa) changes the root
    j = pays.index(True)
    alt = list(recomputed); alt[j] = hs.leaf0(metas[j])
    assert hs.merkle_root(alt) != R
    k = pays.index(False)
    alt = list(recomputed); alt[k] = hs.unit_leaf(metas[k], b"\x00" * 32)
    assert hs.merkle_root(alt) != R


def test_cross_check_with_units_hash_shim():
    """units/_hash_shim.py (P1-B's local copy used by the flow table) must equal crypto/hashing.py."""
    shim = pytest.importorskip("ctsample.units._hash_shim")
    rng = random.Random(9)
    for n in list(range(0, 40)) + [255, 256, 257]:
        L = [rng.randbytes(32) for _ in range(n)]
        assert shim.merkle_root(L) == hs.merkle_root(L)
    fr = rng.randbytes(300)
    assert shim.frame_leaf(123456789, 1, 300, fr) == hs.frame_leaf(123456789, 1, 300, fr)
    m, r = rng.randbytes(138), rng.randbytes(32)
    assert shim.unit_leaf(m, r) == hs.unit_leaf(m, r)
    assert shim.unit_leaf0(m[:68]) == hs.leaf0(m[:68]) == hs.unit_leaf0(m[:68])
    k = rng.randbytes(37)
    assert shim.flow_uid(k, 42) == hs.flow_uid(k, 42)
    assert shim.EMPTY_ROOT == hs.EMPTY_ROOT
