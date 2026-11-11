"""units/features.py: features come from committed metadata only, every path agrees bit-exactly.

* vectorised enc(m_i) decode == dec_metadata reference == S1 Parquet columns (real units if present);
* static features == an independent per-unit Python transcription of the DESIGN §3.4 list;
* rolling statistics: online RollingDeviceStats == vectorised rolling_batch == brute force;
* encode_cols is the inverse of cols_from_meta.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ctsample.units import encode as enc
from ctsample.units import features as F
from ctsample.units.flowtable import default_is_lan

RES = Path(__file__).resolve().parents[1] / "results" / "units"


# ------------------------------------------------------------------------------------------ #
# synthetic metadata records
# ------------------------------------------------------------------------------------------ #
def _rand_ip(rng):
    r = rng.random()
    if r < 0.4:
        return b"\x00" * 10 + b"\xff\xff" + bytes([192, 168, int(rng.integers(0, 3)), int(rng.integers(1, 30))])
    if r < 0.8:
        return b"\x00" * 10 + b"\xff\xff" + bytes(rng.integers(0, 256, 4).tolist())
    if r < 0.9:
        return bytes([0xFE, 0x80]) + bytes(rng.integers(0, 256, 14).tolist())
    return bytes(rng.integers(0, 256, 16).tolist())


def synth_records(n, n_dev=7, n_epochs=400, seed=0):
    """(epochs sorted, list of enc(m_i)) with a realistic mix of protocols, ports and devices."""
    rng = np.random.default_rng(seed)
    ports = [23, 2323, 22, 80, 8080, 443, 53, 123, 1883, 502, 5683, 6667, 25, 587, 7, 1000, 5000, 50000, 60000]
    ips = [_rand_ip(rng) for _ in range(40)]
    epochs = np.sort(rng.choice(n_epochs, size=n))
    out = []
    for i in range(n):
        proto = int(rng.choice([6, 17, 1, 58, 0, 2], p=[0.45, 0.3, 0.08, 0.05, 0.07, 0.05]))
        pay = proto in (6, 17) and rng.random() < 0.7
        pa_ = int(rng.choice(ports)) if rng.random() < 0.3 else int(rng.integers(0, 65536))
        pb_ = int(rng.choice(ports)) if rng.random() < 0.6 else int(rng.integers(0, 65536))
        if proto == 0:
            ipa, ipb = bytes(rng.integers(0, 256, 6).tolist()) + b"\x00" * 10, b"\x00" * 16
        else:
            ipa, ipb = ips[int(rng.integers(0, len(ips)))], ips[int(rng.integers(0, len(ips)))]
        key = enc.make_key(proto, ipa, pa_, ipb, pb_)
        t0 = 1_600_000_000_000_000 + int(epochs[i]) * 30_000_000 + int(rng.integers(0, 29_000_000))
        m = {"key": key, "device_id": int(rng.integers(0, n_dev)), "first_ts_us": t0,
             "last_ts_us": t0 + int(rng.integers(0, 900_000)),
             "pkts": (int(rng.integers(0, 300)), int(rng.integers(0, 300))),
             "payload_bytes": (int(rng.integers(0, 10 ** 6)) if pay else 0, int(rng.integers(0, 10 ** 5)) if pay else 0),
             "frame_bytes": (int(rng.integers(0, 10 ** 6)), int(rng.integers(0, 10 ** 6))),
             "n_syn": int(rng.integers(0, 4)), "n_synack": int(rng.integers(0, 3)), "n_fin": int(rng.integers(0, 3)),
             "n_rst": int(rng.integers(0, 3)), "n_psh": int(rng.integers(0, 50)),
             "conn_state": int(rng.integers(0, len(enc.CONN_STATES))),
             "sizes": [int(x) for x in rng.integers(-1500, 1500, int(rng.integers(0, 9)))],
             "iats": [int(x) for x in rng.integers(0, 255, int(rng.integers(0, 8)))],
             "s": int(rng.integers(60, 10 ** 6)), "flow_uid": bytes(rng.integers(0, 256, 16).tolist()),
             "continuation": bool(rng.random() < 0.3), "fps": bool(rng.random() < 0.4), "pay": pay}
        out.append(enc.enc_metadata(m))
    return epochs, out


# ------------------------------------------------------------------------------------------ #
# independent reference transcription of the static features
# ------------------------------------------------------------------------------------------ #
def _q8_ref(x: int) -> int:
    return ((1 + x) ** 8).bit_length() - 1          # max k with 2^k <= (1+x)^8  <=>  floor(8 log2(1+x))


_NAMED = {23: 0, 2323: 0, 22: 1, 2222: 1, 80: 2, 8080: 2, 8000: 2, 8081: 2, 443: 3, 8443: 3, 53: 4, 123: 5,
          1883: 6, 8883: 6, 502: 7, 5683: 8, 25: 10, 587: 10, **{p: 9 for p in range(6660, 6670)}}


def _rng_b(p):
    return 11 if p < 1024 else (12 if p < 49152 else 13)


def _bucket_ref(proto, pa_, pb_):
    if proto == 0:
        return 15
    if proto in (1, 58):
        return 14
    if proto in (6, 17):
        if pb_ in _NAMED:
            return _NAMED[pb_]
        if pa_ in _NAMED:
            return _NAMED[pa_]
        return _rng_b(pa_ if (pb_ >= 1024 and pa_ < 1024) else pb_)
    return _rng_b(pb_)


def _static_ref(m: dict) -> list[int]:
    proto = m["proto"]
    if proto == 0:
        la = lb = True
    else:
        la, lb = default_is_lan(m["ip_a"]), default_is_lan(m["ip_b"])
    dc = 2 if (la and lb) else (0 if la else (1 if lb else 3))
    pay = m["pay"]
    dur = (m["last_ts_us"] - m["first_ts_us"]) if pay else 0
    pb = m["payload_bytes"] if pay else (0, 0)
    fb = m["frame_bytes"] if pay else (0, 0)
    row = [0 if proto == 6 else (1 if proto == 17 else 2), _bucket_ref(proto, m["port_a"], m["port_b"]), dc,
           int(m.get("continuation", False)) if pay else 0, int(m.get("fps", False)) if pay else 0, _q8_ref(max(dur, 0)),
           _q8_ref(pb[0]), _q8_ref(pb[1]), _q8_ref(m["pkts"][0]), _q8_ref(m["pkts"][1]), _q8_ref(fb[0]), _q8_ref(fb[1]),
           m["conn_state"], m["n_syn"], m["n_fin"], m["n_rst"], m["n_psh"]]
    row += list(m["sizes"]) if pay else [0] * 8
    row += list(m["iats"]) if pay else [enc.IAT_MISSING] * 8
    return row


def _assert_cols_equal(a, b, keys=None):
    for k in keys or F.COL_NAMES:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


# ------------------------------------------------------------------------------------------ #
def test_decode_paths_and_static_reference():
    ep, metas = synth_records(3000, seed=1)
    c_vec = F.cols_from_meta(metas)
    dicts = [enc.dec_metadata(m) for m in metas]
    c_ref = F.cols_from_dicts(dicts)
    _assert_cols_equal(c_vec, c_ref)
    X = F.static_features(c_vec)
    Xr = np.array([_static_ref(d) for d in dicts], dtype=np.int64)
    assert np.array_equal(X, Xr)
    assert F.encode_cols(c_vec) == metas                         # exact inverse


def test_pyarrow_binary_path_and_q8_isqrt():
    import pyarrow as pa
    _, metas = synth_records(500, seed=2)
    _assert_cols_equal(F.cols_from_meta(pa.array(metas, type=pa.binary())), F.cols_from_meta(metas))
    _assert_cols_equal(F.cols_from_meta(pa.chunked_array([pa.array(metas[:200]), pa.array(metas[200:])])),
                       F.cols_from_meta(metas))
    xs = np.array(list(range(0, 5000)) + [2 ** k + d for k in range(1, 40) for d in (-1, 0, 1)], dtype=np.int64)
    assert np.array_equal(F.q8(xs), [_q8_ref(int(x)) for x in xs])
    vs = np.array(list(range(0, 3000)) + [k * k + d for k in (2 ** 16, 2 ** 20, 3 ** 13) for d in (-1, 0, 1)]
                  + [2 ** 32, 2 ** 32 - 1], dtype=np.int64)
    assert np.array_equal(F.isqrt_u64(vs), [math.isqrt(int(v)) for v in vs])


def test_lan_predicate_matches_flowtable():
    rng = np.random.default_rng(3)
    ips = [_rand_ip(rng) for _ in range(3000)]
    special = ["10.1.2.3", "172.16.0.1", "172.32.0.1", "192.168.137.240", "169.254.1.1", "100.64.0.1", "100.128.0.1",
               "127.0.0.1", "0.0.0.0", "224.0.0.251", "255.255.255.255", "8.8.8.8", "192.169.0.1"]
    ips += [b"\x00" * 10 + b"\xff\xff" + bytes(int(x) for x in s.split(".")) for s in special]
    ips += [b"\x00" * 16, b"\x00" * 15 + b"\x01", b"\x00" * 15 + b"\x02", b"\xff\x02" + b"\x00" * 14,
            b"\xfd\x00" + b"\x01" * 14, b"\xfe\xc0" + b"\x00" * 14, b"\x20\x01" + b"\x00" * 14]
    arr = np.frombuffer(b"".join(ips), dtype=np.uint8).reshape(-1, 16)
    assert F.is_lan_v(arr).tolist() == [default_is_lan(x) for x in ips]


def _brute_rolling(epochs, dicts, W):
    out = []
    for i, (e, m) in enumerate(zip(epochs, dicts)):
        win = [(ee, mm) for ee, mm in zip(epochs, dicts) if e - W < ee <= e and mm["device_id"] == m["device_id"]]
        n_u = len(win)
        rd, dp, sr, bp = set(), set(), 0, 0
        for _, mm in win:
            if mm["proto"] != 0:
                la, lb = default_is_lan(mm["ip_a"]), default_is_lan(mm["ip_b"])
                rd.add(mm["ip_a"] if (lb and not la) else mm["ip_b"])
            if mm["proto"] in (6, 17):
                dp.add((mm["proto"], mm["port_b"]))
            sr += mm["conn_state"] in (enc.CS_CODE["S0"], enc.CS_CODE["REJ"])
            same = (_bucket_ref(mm["proto"], mm["port_a"], mm["port_b"]) == _bucket_ref(m["proto"], m["port_a"], m["port_b"])
                    and min(2, (0 if mm["proto"] == 6 else 1 if mm["proto"] == 17 else 2))
                    == min(2, (0 if m["proto"] == 6 else 1 if m["proto"] == 17 else 2)))
            bp += same
        out.append([n_u, len(rd), len(dp), (sr << 16) // n_u, math.isqrt((1 << 32) // (1 + bp))])
    return np.array(out, dtype=np.int64)


def _online(epochs, c, W, cls=None):
    st = (cls or F.RollingDeviceStats)(W)
    R = np.zeros((len(epochs), 5), dtype=np.int64)
    for e in np.unique(epochs):
        idx = np.nonzero(epochs == e)[0]
        R[idx] = st.step(int(e), F.take_cols(c, idx))
    return R


@pytest.mark.parametrize("W", [1, 7, 120])
def test_rolling_online_batch_bruteforce(W):
    ep, metas = synth_records(700, n_dev=5, n_epochs=300, seed=10 + W)
    c = F.cols_from_meta(metas)
    Rb = F.rolling_batch(c, ep, W)
    Ro = _online(ep, c, W)
    assert np.array_equal(Rb, Ro)
    Rr = _brute_rolling(ep, [enc.dec_metadata(m) for m in metas], W)
    assert np.array_equal(Rb, Rr)
    # rev1: the incremental twin gives the same integers
    Ri = _online(ep, c, W, F.IncrementalRollingStats)
    assert np.array_equal(Rb, Ri)


@pytest.mark.parametrize("W", [1, 3, 120])
def test_incremental_rolling_gaps_prime_and_expiry(W):
    """Epoch gaps, priming and expiry: incremental == re-sorting implementation, and its multisets drain to empty."""
    ep, metas = synth_records(900, n_dev=6, n_epochs=400, seed=40 + W)
    keep = (ep % 7 != 3)                                   # gaps: epochs without units
    ep, metas = ep[keep], [m for m, k in zip(metas, keep) if k]
    c = F.cols_from_meta(metas)
    es = np.unique(ep)
    a, b = F.RollingDeviceStats(W), F.IncrementalRollingStats(W)
    half = es[: len(es) // 2]
    hist = [(int(e), F.take_cols(c, np.nonzero(ep == e)[0])) for e in half]
    a.prime(hist)
    b.prime(hist)
    for e in es[len(es) // 2:]:
        idx = np.nonzero(ep == e)[0]
        assert np.array_equal(a.step(int(e), F.take_cols(c, idx)), b.step(int(e), F.take_cols(c, idx)))
    assert b.n_rows == sum(len(w[1]) for w in a.win)
    # far in the future with an empty epoch: everything expires
    b.step(int(es[-1]) + W + 5, F.take_cols(c, np.zeros(0, dtype=np.int64)))
    assert not b.cnt and not b.rd and not b.dp and not b.bpc and not b.nrd and not b.ndp and not b.csr


def test_segment_features_api():
    ep, metas = synth_records(400, n_dev=4, n_epochs=50, seed=5)
    c = F.cols_from_meta(metas)
    Xb = F.features_batch(c, ep)
    assert Xb.dtype == np.int32 and Xb.shape == (len(metas), F.N_FEATURES)
    st = F.RollingDeviceStats()
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]
        Xo = F.segment_features([metas[i] for i in idx], st, epoch=int(e))
        assert np.array_equal(Xo, Xb[idx])


# ------------------------------------------------------------------------------------------ #
# real S1 units (skipped when the extraction outputs are absent)
# ------------------------------------------------------------------------------------------ #
def _real_sources():
    out = []
    for ds, src in (("iot23", "CTU-IoT-Malware-Capture-34-1"), ("ciciot2023_benign", "BenignTraffic1"),
                    ("iot23", "CTU-IoT-Malware-Capture-3-1")):
        d = RES / ds / src
        if (d / "DONE").exists():
            out.append(d)
    return out


@pytest.mark.skipif(not _real_sources(), reason="S1 units not extracted")
def test_real_units_all_paths_identical():
    import pyarrow.parquet as pq
    for d in _real_sources():
        parts = sorted(d.glob("part-*.parquet"))[:2]
        t = pq.read_table(parts[0]) if len(parts) == 1 else __import__("pyarrow").concat_tables(
            [pq.read_table(p) for p in parts])
        t = t.slice(0, min(t.num_rows, 6000))
        c_vec = F.cols_from_meta(t.column("meta"))
        metas = t.column("meta").to_pylist()
        c_ref = F.cols_from_dicts([enc.dec_metadata(m) for m in metas])
        c_pq = F.cols_from_parquet(t)
        _assert_cols_equal(c_vec, c_ref)
        _assert_cols_equal(c_vec, c_pq)
        # the flow table's committed direction class equals the recomputed one
        _, dc = F.side_and_dir(c_vec)
        assert np.array_equal(dc, t.column("dir_class").to_numpy())
        ep = t.column("epoch").to_numpy().astype(np.int64)
        ep = ep - ep.min()
        Xa = F.features_batch(c_vec, ep)
        Xb = F.features_batch(c_pq, ep)
        assert np.array_equal(Xa, Xb)
        Ro = _online(ep, c_vec, F.W_DEFAULT)
        assert np.array_equal(Xa[:, 33:], np.clip(Ro, -2 ** 31, 2 ** 31 - 1))
        assert np.array_equal(Ro, _online(ep, c_vec, F.W_DEFAULT, F.IncrementalRollingStats))
