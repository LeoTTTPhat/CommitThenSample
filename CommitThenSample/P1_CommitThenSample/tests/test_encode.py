"""Metadata record layouts (units/encode.py) and the BLAKE3 commitment formulas (units/_hash_shim.py)."""
import os
import random
import struct

import pytest
from blake3 import blake3

from ctsample.units import _hash_shim as hs
from ctsample.units.encode import (CONN_STATES, CS_CODE, IAT_MISSING, PAY_LEN, ZERO_LEN, dec_metadata, enc_metadata,
                                   iat_code, make_key, split_key)

V4 = b"\x00" * 10 + b"\xff\xff"


def _rec(pay=True, **kw):
    m = {"key": make_key(6, V4 + bytes([192, 168, 1, 2]), 40000, V4 + bytes([8, 8, 8, 8]), 443),
         "device_id": 7, "first_ts_us": 1_545_403_816_870_900, "last_ts_us": 1_545_403_817_000_000,
         "pkts": (5, 4), "payload_bytes": (40, 320), "frame_bytes": (370, 584), "n_syn": 1, "n_synack": 1,
         "n_fin": 2, "n_rst": 0, "n_psh": 3, "conn_state": CS_CODE["SF"], "sizes": [0, 0, 0, 40, -320],
         "iats": [10, 20, 30, 40], "s": 1098, "flow_uid": bytes(range(16)), "continuation": False, "fps": True,
         "pay": pay}
    m.update(kw)
    return m


def test_key_layout():
    k = make_key(17, V4 + b"\x0a\x00\x00\x01", 53, V4 + b"\x0a\x00\x00\x02", 5353)
    assert len(k) == 37
    assert k[0] == 17 and k[1:17] == V4 + b"\x0a\x00\x00\x01"
    assert struct.unpack("<H", k[17:19])[0] == 53 and struct.unpack("<H", k[35:37])[0] == 5353
    assert split_key(k) == (17, V4 + b"\x0a\x00\x00\x01", 53, V4 + b"\x0a\x00\x00\x02", 5353)


def test_payload_roundtrip():
    m = _rec()
    b = enc_metadata(m)
    assert len(b) == PAY_LEN == 138
    d = dec_metadata(b)
    assert d["pay"] and d["key"] == m["key"] and d["device_id"] == 7
    assert d["first_ts_us"] == m["first_ts_us"] and d["last_ts_us"] == m["last_ts_us"]
    assert d["pkts"] == (5, 4) and d["payload_bytes"] == (40, 320) and d["frame_bytes"] == (370, 584)
    assert (d["n_syn"], d["n_synack"], d["n_fin"], d["n_rst"], d["n_psh"]) == (1, 1, 2, 0, 3)
    assert CONN_STATES[d["conn_state"]] == "SF"
    assert d["sizes"] == (0, 0, 0, 40, -320, 0, 0, 0)
    assert d["iats"] == (10, 20, 30, 40, IAT_MISSING, IAT_MISSING, IAT_MISSING, IAT_MISSING)
    assert d["s"] == 1098 and d["flow_uid"] == bytes(range(16)) and d["fps"] and not d["continuation"]
    assert enc_metadata({**m, **{k: v for k, v in d.items() if k in m}}) == b


def test_zero_roundtrip_and_saturation():
    m = _rec(pay=False, pkts=(2 ** 33, 1), n_syn=70000)
    b = enc_metadata(m)
    assert len(b) == ZERO_LEN == 68
    d = dec_metadata(b)
    assert not d["pay"] and d["pkts"] == (2 ** 32 - 1, 1) and d["n_syn"] == 65535
    assert d["conn_state"] == CS_CODE["SF"] and d["first_ts_us"] == m["first_ts_us"]
    big = enc_metadata(_rec(sizes=[70000, -70000]))
    assert dec_metadata(big)["sizes"][:2] == (32767, -32767)
    with pytest.raises(ValueError):
        dec_metadata(b"\x00" * 50)


def test_iat_code():
    assert iat_code(0) == 0 and iat_code(1) == 8 and iat_code(3) == 16
    assert iat_code(1_000_000) == int(8 * 19.931569)
    assert iat_code(10 ** 12) == 254
    codes = [iat_code(x) for x in (1, 10, 100, 10 ** 3, 10 ** 6, 3 * 10 ** 7)]
    assert codes == sorted(codes)


def test_hash_formulas_known_answers():
    f = b"\x01\x02frame"
    exp = blake3(b"CTS/v1/frm" + struct.pack("<QBI", 123, 1, len(f)) + blake3(f).digest()).digest()
    assert hs.frame_leaf(123, 1, len(f), f) == exp
    assert hs.merkle_root([]) == blake3(b"CTS/v1/empty").digest()
    a, b, c = (blake3(bytes([i])).digest() for i in range(3))
    la, lb, lc = (blake3(b"\x00" + x).digest() for x in (a, b, c))
    assert hs.merkle_root([a]) == la
    assert hs.merkle_root([a, b]) == blake3(b"\x01" + la + lb).digest()
    assert hs.merkle_root([a, b, c]) == blake3(b"\x01" + blake3(b"\x01" + la + lb).digest() + lc).digest()
    meta = b"m" * 68
    assert hs.unit_leaf0(meta) == blake3(b"CTS/v1/leaf0" + meta).digest()
    assert hs.unit_leaf(meta, a) == blake3(b"CTS/v1/leaf" + meta + a).digest()
    key = b"k" * 37
    assert hs.flow_uid(key, 5) == blake3(b"CTS/v1/flow" + key + struct.pack("<Q", 5)).digest()[:16]
    sel = sum(hs.sub10_selected(g) for g in range(20000))
    assert 1800 < sel < 2200


def test_shim_matches_crypto_hashing():
    hashing = pytest.importorskip("ctsample.crypto.hashing")
    rng = random.Random(1)
    for n in (0, 1, 2, 3, 5, 8, 13, 64, 100):
        leaves = [os.urandom(32) for _ in range(n)]
        assert hs.merkle_root(leaves) == hashing.merkle_root(leaves)
    for _ in range(50):
        f = os.urandom(rng.randint(14, 1600))
        ts, d = rng.randint(0, 2 ** 62), rng.randint(0, 1)
        assert hs.frame_leaf(ts, d, len(f), f) == hashing.frame_leaf(ts, d, len(f), f)
        meta = os.urandom(rng.choice((PAY_LEN, ZERO_LEN)))
        r = os.urandom(32)
        assert hs.unit_leaf(meta, r) == hashing.unit_leaf(meta, r)
        assert hs.unit_leaf0(meta) == hashing.leaf0(meta)
        key = os.urandom(37)
        assert hs.flow_uid(key, ts) == hashing.flow_uid(key, ts)
