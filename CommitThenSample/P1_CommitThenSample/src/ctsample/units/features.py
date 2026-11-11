"""Utility-model features from COMMITTED METADATA ONLY (DESIGN §3.4; leakage rules §5.7).

Every feature is an integer computed from enc(m_i) (units/encode.py layouts) plus per-device
rolling statistics over the committed metadata of the last W = 120 epochs, so a verifier that holds
only the metadata store recomputes them bit-exactly. Excluded: IP addresses, MACs, absolute times,
dataset identifiers, service strings (the device id is used only to GROUP the rolling statistics).

Canonical columns ("cols"): a dict of numpy arrays decoded from enc(m_i). Three equivalent producers:
  * ``cols_from_meta(metas)``    vectorised decode of the fixed-width records (production path);
  * ``cols_from_dicts(dicts)``   from ``units.encode.dec_metadata`` dicts (reference path);
  * ``cols_from_parquet(table)`` from the S1 Parquet columns (tests prove the three agree).
Fields absent from the compact zero-payload layout are filled with neutral values
(last_ts = first_ts, payload/frame bytes 0, sizes 0, iats 255, s 0, continuation 0, fps 0).

Static features (per unit; FEATURE_NAMES[0:33]):
  proto3 (TCP 0, UDP 1, other 2); dport bucket (16, see DPORT_BUCKETS; the service port is port_b if
  it is a named port, else port_a if named, else the range bucket of port_b — or of port_a when only
  port_a is < 1024, i.e. the server started the capture mid-flow); direction class (0 LAN→WAN,
  1 WAN→LAN, 2 LAN↔LAN incl. L2, 3 WAN↔WAN; LAN = flowtable.default_is_lan); continuation; fps;
  duration within the epoch (last_ts − first_ts, µs) and per-direction payload bytes, packets and frame
  bytes, each q8(x) = ⌊8·log2(1 + x)⌋ (exact integer thresholds, no float log); conn_state code;
  SYN, FIN, RST, PSH counts; first-8 signed payload sizes; first-8 IAT codes.
Rolling features (per unit, device d = its device_id, window = epochs (e − W, e], i.e. the current
epoch's committed units are included; FEATURE_NAMES[33:38]):
  r_units  = #units of d;  r_rdst = #distinct remote IPs of d (IP units: the non-device endpoint);
  r_dports = #distinct (proto, port_b) of d's TCP/UDP units;
  r_s0rej  = ⌊2^16 · #(conn_state ∈ {S0, REJ}) / r_units⌋;
  r_rarity = ⌊2^16 / √(1 + #units of d with the same (dport bucket, proto3))⌋  (= isqrt(2^32 // (1+c))).
``RollingDeviceStats`` computes them online in epoch order (gateway/verifier); ``rolling_batch`` is the
vectorised whole-source equivalent (training, streams). Tests check that both agree exactly.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np

from .encode import CS_CODE, IAT_MISSING, PAY_LEN, ZERO_LEN

__all__ = ["FEATURE_NAMES", "STATIC_NAMES", "ROLLING_NAMES", "N_FEATURES", "DPORT_BUCKETS", "PAY_DT", "ZERO_DT",
           "cols_from_meta", "cols_from_dicts", "cols_from_parquet", "cols_from_units", "concat_cols",
           "take_cols", "is_lan_v", "side_and_dir", "dport_bucket", "proto3", "q8", "isqrt_u64",
           "static_features", "RollingDeviceStats", "IncrementalRollingStats", "rolling_batch", "segment_features",
           "features_batch",
           "encode_cols", "W_DEFAULT", "FEATURE_VERSION"]

FEATURE_VERSION = "ctsample.features/v1"
W_DEFAULT = 120

STATIC_NAMES = (["proto3", "dport_bucket", "dir_class", "continuation", "fps", "dur_q8",
                 "pay0_q8", "pay1_q8", "pk0_q8", "pk1_q8", "fb0_q8", "fb1_q8", "conn_state",
                 "n_syn", "n_fin", "n_rst", "n_psh"]
                + [f"size{j}" for j in range(8)] + [f"iat{j}" for j in range(8)])
ROLLING_NAMES = ["r_units", "r_rdst", "r_dports", "r_s0rej_q16", "r_rarity_q16"]
FEATURE_NAMES = STATIC_NAMES + ROLLING_NAMES
N_FEATURES = len(FEATURE_NAMES)
assert len(STATIC_NAMES) == 33 and N_FEATURES == 38

# ------------------------------------------------------------------------------------------ #
# fixed-width record layouts (must mirror units/encode.py PAY / ZERO structs exactly)
# ------------------------------------------------------------------------------------------ #
_COMMON = [("proto", "u1", 0), ("ip_a", ("u1", 16), 1), ("port_a", "<u2", 17), ("ip_b", ("u1", 16), 19),
           ("port_b", "<u2", 35), ("device_id", "<u4", 37), ("first_ts", "<u8", 41)]
PAY_DT = np.dtype({
    "names": [n for n, _, _ in _COMMON] + ["last_ts", "pk0", "pk1", "pb0", "pb1", "fb0", "fb1", "syn", "synack",
                                           "fin", "rst", "psh", "cs", "sizes", "iats", "s", "flow_uid", "cont",
                                           "fps"],
    "formats": [f for _, f, _ in _COMMON] + ["<u8", "<u4", "<u4", "<u4", "<u4", "<u4", "<u4", "<u2", "<u2", "<u2",
                                             "<u2", "<u2", "u1", ("<i2", 8), ("u1", 8), "<u4", ("u1", 16), "u1",
                                             "u1"],
    "offsets": [o for _, _, o in _COMMON] + [49, 57, 61, 65, 69, 73, 77, 81, 83, 85, 87, 89, 91, 92, 108, 116, 120,
                                             136, 137],
    "itemsize": PAY_LEN})
ZERO_DT = np.dtype({
    "names": [n for n, _, _ in _COMMON] + ["pk0", "pk1", "syn", "synack", "fin", "rst", "psh", "cs"],
    "formats": [f for _, f, _ in _COMMON] + ["<u4", "<u4", "<u2", "<u2", "<u2", "<u2", "<u2", "u1"],
    "offsets": [o for _, _, o in _COMMON] + [49, 53, 57, 59, 61, 63, 65, 67],
    "itemsize": ZERO_LEN})

_INT_COLS = ("proto", "port_a", "port_b", "device_id", "first_ts", "last_ts", "pk0", "pk1", "pb0", "pb1", "fb0",
             "fb1", "syn", "synack", "fin", "rst", "psh", "cs", "s", "cont", "fps")
COL_NAMES = ("pay",) + _INT_COLS + ("ip_a", "ip_b", "sizes", "iats", "flow_uid")


def _empty_cols(n: int) -> dict:
    c = {k: np.zeros(n, dtype=np.int64) for k in _INT_COLS}
    c["pay"] = np.zeros(n, dtype=bool)
    c["ip_a"] = np.zeros((n, 16), dtype=np.uint8)
    c["ip_b"] = np.zeros((n, 16), dtype=np.uint8)
    c["sizes"] = np.zeros((n, 8), dtype=np.int16)
    c["iats"] = np.full((n, 8), IAT_MISSING, dtype=np.uint8)
    c["flow_uid"] = np.zeros((n, 16), dtype=np.uint8)
    return c


def _fill(c: dict, rows: np.ndarray, rec: np.ndarray, pay: bool) -> None:
    for k in ("proto", "port_a", "port_b", "device_id", "first_ts", "pk0", "pk1", "syn", "synack", "fin", "rst",
              "psh", "cs"):
        c[k][rows] = rec[k].astype(np.int64)
    c["ip_a"][rows] = rec["ip_a"]
    c["ip_b"][rows] = rec["ip_b"]
    c["pay"][rows] = pay
    if pay:
        for k in ("last_ts", "pb0", "pb1", "fb0", "fb1", "s", "cont", "fps"):
            c[k][rows] = rec[k].astype(np.int64)
        c["sizes"][rows] = rec["sizes"]
        c["iats"][rows] = rec["iats"]
        c["flow_uid"][rows] = rec["flow_uid"]
    else:
        c["last_ts"][rows] = rec["first_ts"].astype(np.int64)


def _binary_arrays(metas):
    """Yield (offsets int64 (n+1,), data uint8) for a pyarrow (chunked) binary array, or None."""
    try:
        import pyarrow as pa
    except ImportError:  # pragma: no cover
        return None
    if isinstance(metas, pa.ChunkedArray):
        chunks = metas.chunks
    elif isinstance(metas, pa.Array):
        chunks = [metas]
    else:
        return None
    out = []
    for ch in chunks:
        if ch.null_count:
            raise ValueError("null metadata record")
        if pa.types.is_large_binary(ch.type):
            ow = np.int64
        elif pa.types.is_binary(ch.type):
            ow = np.int32
        else:
            raise TypeError(f"metadata column must be binary, got {ch.type}")
        bufs = ch.buffers()
        offs = np.frombuffer(bufs[1], dtype=ow)[ch.offset: ch.offset + len(ch) + 1].astype(np.int64)
        data = np.frombuffer(bufs[2], dtype=np.uint8) if bufs[2] is not None else np.zeros(0, np.uint8)
        out.append((offs, data))
    return out


def cols_from_meta(metas) -> dict:
    """Vectorised decode of enc(m_i) records (list of bytes, numpy object array or a pyarrow binary
    (chunked) array). The record length selects the layout (138 B payload, 68 B compact)."""
    arrs = _binary_arrays(metas)
    if arrs is None:
        metas = list(metas)
        n = len(metas)
        lens = np.fromiter((len(m) for m in metas), dtype=np.int64, count=n)
        c = _empty_cols(n)
        for L, dt, pay in ((PAY_LEN, PAY_DT, True), (ZERO_LEN, ZERO_DT, False)):
            rows = np.nonzero(lens == L)[0]
            if len(rows):
                buf = b"".join(metas[i] for i in rows)
                _fill(c, rows, np.frombuffer(buf, dtype=dt), pay)
        bad = ~((lens == PAY_LEN) | (lens == ZERO_LEN))
        if bad.any():
            raise ValueError(f"metadata record of unexpected length {int(lens[bad][0])}")
        return c
    parts = []
    for offs, data in arrs:
        n = len(offs) - 1
        lens = np.diff(offs)
        c = _empty_cols(n)
        bad = ~((lens == PAY_LEN) | (lens == ZERO_LEN))
        if bad.any():
            raise ValueError(f"metadata record of unexpected length {int(lens[bad][0])}")
        for L, dt, pay in ((PAY_LEN, PAY_DT, True), (ZERO_LEN, ZERO_DT, False)):
            rows = np.nonzero(lens == L)[0]
            if len(rows):
                idx = offs[rows][:, None] + np.arange(L, dtype=np.int64)[None, :]
                raw = np.ascontiguousarray(data[idx])
                _fill(c, rows, raw.view(dt).reshape(len(rows)), pay)
        parts.append(c)
    return concat_cols(parts) if len(parts) != 1 else parts[0]


def cols_from_dicts(dicts: Sequence[dict]) -> dict:
    """Reference path: columns from ``units.encode.dec_metadata`` dicts (slow, per unit)."""
    n = len(dicts)
    c = _empty_cols(n)
    for i, m in enumerate(dicts):
        c["pay"][i] = bool(m["pay"])
        c["proto"][i] = m["proto"]
        c["ip_a"][i] = np.frombuffer(m["ip_a"], dtype=np.uint8)
        c["ip_b"][i] = np.frombuffer(m["ip_b"], dtype=np.uint8)
        c["port_a"][i], c["port_b"][i] = m["port_a"], m["port_b"]
        c["device_id"][i] = m["device_id"]
        c["first_ts"][i] = m["first_ts_us"]
        c["pk0"][i], c["pk1"][i] = m["pkts"]
        c["syn"][i], c["synack"][i], c["fin"][i] = m["n_syn"], m["n_synack"], m["n_fin"]
        c["rst"][i], c["psh"][i], c["cs"][i] = m["n_rst"], m["n_psh"], m["conn_state"]
        if m["pay"]:
            c["last_ts"][i] = m["last_ts_us"]
            c["pb0"][i], c["pb1"][i] = m["payload_bytes"]
            c["fb0"][i], c["fb1"][i] = m["frame_bytes"]
            c["sizes"][i] = m["sizes"]
            c["iats"][i] = m["iats"]
            c["s"][i] = m["s"]
            c["flow_uid"][i] = np.frombuffer(m["flow_uid"], dtype=np.uint8)
            c["cont"][i] = int(m["continuation"])
            c["fps"][i] = int(m["fps"])
        else:
            c["last_ts"][i] = m["first_ts_us"]
    return c


def _ip16_from_text(txt: str, proto: int) -> bytes:
    import ipaddress
    if proto == 0:                                  # L2 key: MAC text in ip_a, "" in ip_b
        if not txt:
            return b"\x00" * 16
        return bytes(int(x, 16) for x in txt.split(":")) + b"\x00" * 10
    a = ipaddress.ip_address(txt)
    if a.version == 4:
        return b"\x00" * 10 + b"\xff\xff" + a.packed
    return a.packed


def cols_from_parquet(table) -> dict:
    """Columns from the S1 Parquet columns (pyarrow Table / pandas / polars DataFrame); used by tests to
    prove that the Parquet columns equal the committed metadata (saturations as in enc_metadata)."""
    if hasattr(table, "to_pandas") and not hasattr(table, "iloc"):
        df = table.to_pandas()
    else:
        df = table
    n = len(df)
    c = _empty_cols(n)
    pay = df["pay"].to_numpy(dtype=bool)
    proto = df["proto"].to_numpy(dtype=np.int64)
    sat32 = lambda x: np.minimum(np.asarray(x, dtype=np.int64), 0xFFFFFFFF)  # noqa: E731
    sat16 = lambda x: np.minimum(np.asarray(x, dtype=np.int64), 0xFFFF)  # noqa: E731
    c["pay"] = pay
    c["proto"] = proto
    c["port_a"] = df["port_a"].to_numpy(dtype=np.int64) & 0xFFFF
    c["port_b"] = df["port_b"].to_numpy(dtype=np.int64) & 0xFFFF
    c["device_id"] = df["device_id"].to_numpy(dtype=np.int64)
    c["first_ts"] = df["first_ts_us"].to_numpy(dtype=np.int64)
    c["last_ts"] = np.where(pay, df["last_ts_us"].to_numpy(dtype=np.int64), c["first_ts"])
    c["pk0"] = sat32(df["pkts0"])
    c["pk1"] = sat32(df["pkts1"])
    z = np.zeros(n, np.int64)
    c["pb0"] = np.where(pay, sat32(df["payload0"]), z)
    c["pb1"] = np.where(pay, sat32(df["payload1"]), z)
    c["fb0"] = np.where(pay, sat32(df["frame0"]), z)
    c["fb1"] = np.where(pay, sat32(df["frame1"]), z)
    for k, col in (("syn", "n_syn"), ("synack", "n_synack"), ("fin", "n_fin"), ("rst", "n_rst"), ("psh", "n_psh")):
        c[k] = sat16(df[col])
    c["cs"] = df["conn_state"].to_numpy(dtype=np.int64)
    c["s"] = np.where(pay, sat32(df["s"]), z)
    c["cont"] = np.where(pay, df["continuation"].to_numpy(dtype=np.int64), z)
    c["fps"] = np.where(pay, df["fps"].to_numpy(dtype=np.int64), z)
    sizes = np.stack([df[f"size{j}"].to_numpy(dtype=np.int64) for j in range(8)], axis=1)
    iats = np.stack([df[f"iat{j}"].to_numpy(dtype=np.int64) for j in range(8)], axis=1)
    c["sizes"] = np.where(pay[:, None], np.clip(sizes, -32767, 32767), 0).astype(np.int16)
    c["iats"] = np.where(pay[:, None], iats, IAT_MISSING).astype(np.uint8)
    ipa = df["ip_a"].tolist()
    ipb = df["ip_b"].tolist()
    c["ip_a"] = np.frombuffer(b"".join(_ip16_from_text(x, int(p)) for x, p in zip(ipa, proto)),
                              dtype=np.uint8).reshape(n, 16).copy()
    c["ip_b"] = np.frombuffer(b"".join(_ip16_from_text(x, int(p)) for x, p in zip(ipb, proto)),
                              dtype=np.uint8).reshape(n, 16).copy()
    fu = df["flow_uid"].tolist()
    fuz = np.frombuffer(b"".join(bytes(x) for x in fu), dtype=np.uint8).reshape(n, 16)
    c["flow_uid"] = np.where(pay[:, None], fuz, 0).astype(np.uint8)
    return c


def cols_from_units(units) -> dict:
    """Columns of ``units.batch.Unit`` objects (protocol path: decodes ``u.meta``)."""
    return cols_from_meta([u.meta for u in units])


def concat_cols(parts: Sequence[dict]) -> dict:
    parts = [p for p in parts if p is not None]
    if not parts:
        return _empty_cols(0)
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def take_cols(c: dict, idx) -> dict:
    return {k: v[idx] for k, v in c.items()}


def encode_cols(c: dict) -> list[bytes]:
    """Re-encode columns into enc(m_i) records (inverse of cols_from_meta; used by the LP stream builder
    after address remapping and time offsets)."""
    n = len(c["pay"])
    out = [b""] * n
    for L, dt, pay in ((PAY_LEN, PAY_DT, True), (ZERO_LEN, ZERO_DT, False)):
        rows = np.nonzero(c["pay"] == pay)[0]
        if not len(rows):
            continue
        rec = np.zeros(len(rows), dtype=dt)
        for k in ("proto", "port_a", "port_b", "device_id", "first_ts", "pk0", "pk1", "syn", "synack", "fin",
                  "rst", "psh", "cs"):
            rec[k] = c[k][rows]
        rec["ip_a"] = c["ip_a"][rows]
        rec["ip_b"] = c["ip_b"][rows]
        if pay:
            for k in ("last_ts", "pb0", "pb1", "fb0", "fb1", "s", "cont", "fps"):
                rec[k] = c[k][rows]
            rec["sizes"] = c["sizes"][rows]
            rec["iats"] = c["iats"][rows]
            rec["flow_uid"] = c["flow_uid"][rows]
        raw = rec.tobytes()
        for j, i in enumerate(rows):
            out[i] = raw[j * L:(j + 1) * L]
    return out


# ------------------------------------------------------------------------------------------ #
# elementary integer feature maps
# ------------------------------------------------------------------------------------------ #
_V4P = np.array([0] * 10 + [0xFF, 0xFF], dtype=np.uint8)


def is_lan_v(ip: np.ndarray) -> np.ndarray:
    """Vectorised ``units.flowtable.default_is_lan`` over (N, 16) uint8 addresses."""
    ip = np.asarray(ip, dtype=np.uint8).reshape(-1, 16)
    v4 = np.all(ip[:, :12] == _V4P, axis=1)
    a = ip[:, 12].astype(np.int64)
    b = ip[:, 13].astype(np.int64)
    lan4 = ((a == 10) | ((a == 172) & (b >= 16) & (b <= 31)) | ((a == 192) & (b == 168)) | ((a == 169) & (b == 254))
            | ((a == 100) & (b >= 64) & (b <= 127)) | (a == 127) | (a == 0) | (a >= 224))
    b0 = ip[:, 0].astype(np.int64)
    b1 = ip[:, 1].astype(np.int64)
    zero15 = np.all(ip[:, :15] == 0, axis=1)
    lan6 = (b0 == 0xFF) | ((b0 & 0xFE) == 0xFC) | ((b0 == 0xFE) & ((b1 & 0xC0) == 0x80)) | (zero15 & (ip[:, 15] <= 1))
    return np.where(v4, lan4, lan6)


def side_and_dir(c: dict) -> tuple[np.ndarray, np.ndarray]:
    """(device side 0/1, direction class) exactly as flowtable._device (LAN = default_is_lan; L2 → (0, 2))."""
    la = is_lan_v(c["ip_a"])
    lb = is_lan_v(c["ip_b"])
    l2 = c["proto"] == 0
    la = la | l2
    lb = lb | l2
    dc = np.where(la & lb, 2, np.where(la, 0, np.where(lb, 1, 3))).astype(np.int64)
    side = np.where(~la & lb, 1, 0).astype(np.int64)
    return side, dc


# 16 dport buckets (DESIGN §3.4)
DPORT_BUCKETS = ("telnet", "ssh", "http", "https", "dns", "ntp", "mqtt", "modbus", "coap", "irc", "smtp",
                 "other<1024", "1024-49151", ">=49152", "icmp", "l2")
_NAMED = {23: 0, 2323: 0, 22: 1, 2222: 1, 80: 2, 8080: 2, 8000: 2, 8081: 2, 443: 3, 8443: 3, 53: 4, 123: 5,
          1883: 6, 8883: 6, 502: 7, 5683: 8, 25: 10, 587: 10}
for _p in range(6660, 6670):
    _NAMED[_p] = 9
_NAMED_LUT = np.full(65536, -1, dtype=np.int64)
for _p, _b in _NAMED.items():
    _NAMED_LUT[_p] = _b


def _range_bucket(p: np.ndarray) -> np.ndarray:
    return np.where(p < 1024, 11, np.where(p < 49152, 12, 13))


def dport_bucket(proto, port_a, port_b) -> np.ndarray:
    proto = np.asarray(proto, dtype=np.int64)
    pa_ = np.asarray(port_a, dtype=np.int64) & 0xFFFF
    pb_ = np.asarray(port_b, dtype=np.int64) & 0xFFFF
    na = _NAMED_LUT[pa_]
    nb = _NAMED_LUT[pb_]
    srv = np.where((pb_ >= 1024) & (pa_ < 1024), pa_, pb_)
    tcpudp = np.where(nb >= 0, nb, np.where(na >= 0, na, _range_bucket(srv)))
    other = _range_bucket(pb_)                      # other IP protocols: ports are 0 -> "other<1024"
    out = np.where((proto == 6) | (proto == 17), tcpudp, other)
    out = np.where((proto == 1) | (proto == 58), 14, out)
    out = np.where(proto == 0, 15, out)
    return out.astype(np.int64)


def proto3(proto) -> np.ndarray:
    proto = np.asarray(proto, dtype=np.int64)
    return np.where(proto == 6, 0, np.where(proto == 17, 1, 2)).astype(np.int64)


def _q8_table(kmax: int = 8 * 64) -> np.ndarray:
    """t_k = ⌈2^(k/8)⌉ (exact integer 8th roots); q8(x) = max{k : t_k ≤ 1 + x}."""
    t = []
    for k in range(kmax + 1):
        n = 1 << k
        r = math.isqrt(math.isqrt(math.isqrt(n)))            # ⌊n^(1/8)⌋ (nested floors are exact)
        while (r + 1) ** 8 <= n:                             # defensive fix-ups (never taken in practice)
            r += 1
        while r ** 8 > n:
            r -= 1
        t.append(r if r ** 8 == n else r + 1)
    return np.array([min(v, (1 << 63) - 1) for v in t], dtype=np.int64)


_Q8 = _q8_table()


def q8(x) -> np.ndarray:
    """⌊8·log2(1 + x)⌋ for integers x ≥ 0 (exact)."""
    x = np.asarray(x, dtype=np.int64)
    if x.size and x.min() < 0:
        raise ValueError("q8 needs x >= 0")
    return (np.searchsorted(_Q8, x + 1, side="right") - 1).astype(np.int64)


def isqrt_u64(v) -> np.ndarray:
    """Exact ⌊√v⌋ for integer arrays 0 ≤ v < 2^62 (float estimate + integer fix-up)."""
    v = np.asarray(v, dtype=np.int64)
    if v.size and (v.min() < 0 or v.max() >= (1 << 62)):
        raise ValueError("isqrt_u64 range")
    r = np.floor(np.sqrt(v.astype(np.float64))).astype(np.int64)
    for _ in range(4):
        r = np.where(r * r > v, r - 1, r)
        r = np.where((r + 1) * (r + 1) <= v, r + 1, r)
    return r


def static_features(c: dict, wide: bool = True) -> np.ndarray:
    """(N, 33) per-unit features (FEATURE_NAMES[:33]); int64 (wide) or int32 (values clipped to int32)."""
    n = len(c["pay"])
    side, dc = side_and_dir(c)
    dur = np.maximum(c["last_ts"] - c["first_ts"], 0)
    cols = [proto3(c["proto"]), dport_bucket(c["proto"], c["port_a"], c["port_b"]), dc,
            c["cont"].astype(np.int64), c["fps"].astype(np.int64), q8(dur),
            q8(c["pb0"]), q8(c["pb1"]), q8(c["pk0"]), q8(c["pk1"]), q8(c["fb0"]), q8(c["fb1"]),
            c["cs"].astype(np.int64), c["syn"], c["fin"], c["rst"], c["psh"]]
    X = np.empty((n, 33), dtype=np.int64 if wide else np.int32)
    for j, v in enumerate(cols):
        X[:, j] = np.clip(v, _I32.min, _I32.max) if not wide else v
    X[:, 17:25] = c["sizes"]
    X[:, 25:33] = c["iats"]
    return X


# ------------------------------------------------------------------------------------------ #
# rolling per-device statistics
# ------------------------------------------------------------------------------------------ #
_S0, _REJ = CS_CODE["S0"], CS_CODE["REJ"]
_TWO32 = 1 << 32


def _roll_inputs(c: dict):
    """Per-unit inputs of the rolling statistics: device, remote-IP bytes (N,16), valid-IP mask,
    dport key (proto<<16 | port_b) with TCP/UDP mask, S0/REJ flag, (bucket, proto3) key."""
    side, _ = side_and_dir(c)
    remote = np.where(side[:, None] == 1, c["ip_a"], c["ip_b"])
    ipm = c["proto"] != 0
    tu = (c["proto"] == 6) | (c["proto"] == 17)
    dpk = (c["proto"] << 16) | (c["port_b"] & 0xFFFF)
    sr = (c["cs"] == _S0) | (c["cs"] == _REJ)
    bp = dport_bucket(c["proto"], c["port_a"], c["port_b"]) * 3 + proto3(c["proto"])
    return c["device_id"].astype(np.int64), remote, ipm, dpk, tu, sr, bp


def _ip_to_u64pair(ip16: np.ndarray) -> np.ndarray:
    ip16 = np.ascontiguousarray(np.asarray(ip16, dtype=np.uint8).reshape(-1, 16))
    return ip16.view(np.uint64).reshape(-1, 2)


def _finish_rolling(n_units, n_rdst, n_dp, n_sr, n_bp) -> np.ndarray:
    n_units = np.asarray(n_units, dtype=np.int64)
    out = np.empty((len(n_units), 5), dtype=np.int64)
    out[:, 0] = n_units
    out[:, 1] = n_rdst
    out[:, 2] = n_dp
    out[:, 3] = np.where(n_units > 0, (np.asarray(n_sr, dtype=np.int64) << 16) // np.maximum(n_units, 1), 0)
    out[:, 4] = isqrt_u64(_TWO32 // (1 + np.asarray(n_bp, dtype=np.int64)))
    return out


class RollingDeviceStats:
    """Online per-device rolling statistics over the window of epochs (e − W, e] (gateway / verifier).

    ``step(e, cols)`` registers the committed units of epoch e (epochs strictly increasing; gaps are
    epochs without units) and returns their (n, 5) rolling features. ``prime(history)`` feeds earlier
    epochs ((e, cols) pairs) without returning features. State is exact (integer counts)."""

    def __init__(self, W: int = W_DEFAULT):
        self.W = int(W)
        self.win: deque = deque()          # (e, dev, remote u64 pairs, ipm, dpk, tu, sr, bp)
        self.last_e: int | None = None

    def _push(self, e: int, c: dict) -> None:
        if self.last_e is not None and e <= self.last_e:
            raise ValueError(f"RollingDeviceStats: epochs must increase ({e} after {self.last_e})")
        self.last_e = e
        while self.win and self.win[0][0] <= e - self.W:
            self.win.popleft()
        dev, remote, ipm, dpk, tu, sr, bp = _roll_inputs(c)
        self.win.append((e, dev, _ip_to_u64pair(remote), ipm, dpk, tu, sr, bp))

    def prime(self, history) -> None:
        for e, c in history:
            self._push(int(e), c)

    def step(self, e: int, c: dict) -> np.ndarray:
        self._push(int(e), c)
        n = len(c["pay"])
        if n == 0:
            return np.zeros((0, 5), dtype=np.int64)
        dev = np.concatenate([w[1] for w in self.win])
        rem = np.concatenate([w[2] for w in self.win])
        ipm = np.concatenate([w[3] for w in self.win])
        dpk = np.concatenate([w[4] for w in self.win])
        tu = np.concatenate([w[5] for w in self.win])
        sr = np.concatenate([w[6] for w in self.win])
        bp = np.concatenate([w[7] for w in self.win])
        qdev = self.win[-1][1]
        qbp = self.win[-1][7]
        udev, inv = np.unique(dev, return_inverse=True)
        nd = len(udev)
        qd = np.searchsorted(udev, qdev)
        cnt = np.bincount(inv, minlength=nd)
        csr = np.bincount(inv, weights=sr.astype(np.float64), minlength=nd).astype(np.int64)
        # distinct remote IPs per device
        m = ipm
        if m.any():
            trip = np.stack([inv[m].astype(np.uint64), rem[m, 0], rem[m, 1]], axis=1)
            ut = np.unique(trip, axis=0)
            nrd = np.bincount(ut[:, 0].astype(np.int64), minlength=nd)
        else:
            nrd = np.zeros(nd, np.int64)
        if tu.any():
            pr = np.unique(np.stack([inv[tu], dpk[tu]], axis=1), axis=0)
            ndp = np.bincount(pr[:, 0], minlength=nd)
        else:
            ndp = np.zeros(nd, np.int64)
        key = inv * 64 + bp
        qkey = qd * 64 + qbp
        ks = np.sort(key)
        nbp = np.searchsorted(ks, qkey, side="right") - np.searchsorted(ks, qkey, side="left")
        return _finish_rolling(cnt[qd], nrd[qd], ndp[qd], csr[qd], nbp)


class IncrementalRollingStats:
    """Bit-exact incremental twin of ``RollingDeviceStats`` (rev1, flood robustness of the commit window).

    ``RollingDeviceStats.step`` re-sorts every unit of the 120-epoch window at every epoch, so its cost grows
    with the WINDOW volume: one flooded epoch slows the next 119 commits, and a scanning device that is merely
    busy (≈ 6,000 zero-payload units per epoch on LP-IoT23(49-1)) costs ≈ 1 s per commit on a performance core.
    This class keeps exact per-device multisets of the window — units, S0/REJ units, (device, remote address)
    and (device, protocol‖responder port) multiplicities with their distinct counts, (device, bucket‖protocol)
    counts — and per epoch only adds the new epoch's unique keys and subtracts those of the epoch that leaves
    the window: O(n_e + n_{e−W}) dictionary updates. Same interface (``step``, ``prime``, ``last_e``, ``win``)
    and the same integers as ``RollingDeviceStats`` (tests/test_features.py)."""

    def __init__(self, W: int = W_DEFAULT):
        self.W = int(W)
        self.win: deque = deque()          # (e, n_rows, per-epoch unique keys and counts)
        self.last_e: int | None = None
        self.cnt: dict = {}
        self.csr: dict = {}
        self.rd: dict = {}
        self.nrd: dict = {}
        self.dp: dict = {}
        self.ndp: dict = {}
        self.bpc: dict = {}

    @property
    def n_rows(self) -> int:
        return int(sum(w[1] for w in self.win))

    @staticmethod
    def _summary(c: dict):
        dev, remote, ipm, dpk, tu, sr, bp = _roll_inputs(c)
        ud, uc = np.unique(dev, return_counts=True)
        us, usc = np.unique(dev[sr], return_counts=True)
        if ipm.any():
            pr = _ip_to_u64pair(remote[ipm])
            trip = np.stack([dev[ipm].astype(np.uint64), pr[:, 0], pr[:, 1]], axis=1)
            rk, rc = np.unique(trip, axis=0, return_counts=True)
            rd = (list(zip(rk[:, 0].tolist(), rk[:, 1].tolist(), rk[:, 2].tolist())), rc.tolist())
        else:
            rd = ([], [])
        if tu.any():
            pk, pc = np.unique(np.stack([dev[tu], dpk[tu]], axis=1), axis=0, return_counts=True)
            dp = (list(zip(pk[:, 0].tolist(), pk[:, 1].tolist())), pc.tolist())
        else:
            dp = ([], [])
        bk, bc = np.unique(dev * 64 + bp, return_counts=True)
        return (len(dev), (ud.tolist(), uc.tolist()), (us.tolist(), usc.tolist()), rd, dp, (bk.tolist(), bc.tolist())), \
            (dev, bp)

    @staticmethod
    def _bump(d: dict, key, k: int) -> tuple[int, int]:
        old = d.get(key, 0)
        new = old + k
        if new:
            d[key] = new
        else:
            d.pop(key, None)
        return old, new

    def _apply(self, s, sign: int) -> None:
        _, (ud, uc), (us, usc), (rk, rc), (pk, pc), (bk, bc) = s
        bump = self._bump
        for d, k in zip(ud, uc):
            bump(self.cnt, d, sign * k)
        for d, k in zip(us, usc):
            bump(self.csr, d, sign * k)
        for key, k in zip(rk, rc):
            old, new = bump(self.rd, key, sign * k)
            if old == 0 and new > 0:
                bump(self.nrd, key[0], 1)
            elif old > 0 and new == 0:
                bump(self.nrd, key[0], -1)
        for key, k in zip(pk, pc):
            old, new = bump(self.dp, key, sign * k)
            if old == 0 and new > 0:
                bump(self.ndp, key[0], 1)
            elif old > 0 and new == 0:
                bump(self.ndp, key[0], -1)
        for b, k in zip(bk, bc):
            bump(self.bpc, b, sign * k)

    def _push(self, e: int, c: dict):
        if self.last_e is not None and e <= self.last_e:
            raise ValueError(f"IncrementalRollingStats: epochs must increase ({e} after {self.last_e})")
        self.last_e = e
        while self.win and self.win[0][0] <= e - self.W:
            old = self.win.popleft()
            self._apply(old[1:], -1)
        s, q = self._summary(c)
        self._apply(s, +1)
        self.win.append((e,) + s)
        return q

    def prime(self, history) -> None:
        for e, c in history:
            self._push(int(e), c)

    def step(self, e: int, c: dict) -> np.ndarray:
        dev, bp = self._push(int(e), c)
        n = len(c["pay"])
        if n == 0:
            return np.zeros((0, 5), dtype=np.int64)
        qd, inv = np.unique(dev, return_inverse=True)
        qdl = qd.tolist()
        cnt = np.array([self.cnt.get(d, 0) for d in qdl], dtype=np.int64)[inv]
        nrd = np.array([self.nrd.get(d, 0) for d in qdl], dtype=np.int64)[inv]
        ndp = np.array([self.ndp.get(d, 0) for d in qdl], dtype=np.int64)[inv]
        csr = np.array([self.csr.get(d, 0) for d in qdl], dtype=np.int64)[inv]
        qb, binv = np.unique(dev * 64 + bp, return_inverse=True)
        nbp = np.array([self.bpc.get(b, 0) for b in qb.tolist()], dtype=np.int64)[binv]
        return _finish_rolling(cnt, nrd, ndp, csr, nbp)


def _window_count(key, ep, W, qkey, qep, mask=None):
    """#{j : key_j = qkey, qep − W < ep_j ≤ qep} (vectorised; ep ≥ 0)."""
    key = np.asarray(key, dtype=np.int64)
    ep = np.asarray(ep, dtype=np.int64)
    if mask is not None:
        key, ep = key[mask], ep[mask]
    SH = np.int64(1 << 32)
    comb = np.sort(key * SH + (ep + W))
    qk = np.asarray(qkey, dtype=np.int64) * SH
    qe = np.asarray(qep, dtype=np.int64)
    hi = np.searchsorted(comb, qk + (qe + W), side="right")
    lo = np.searchsorted(comb, qk + qe, side="right")
    return hi - lo


def _window_distinct(dev_dense, val_ids, ep, W, n_dev, E, q_dev, q_ep):
    """#distinct val_ids of device d in (t − W, t] at the query points (interval-union difference array)."""
    if len(dev_dense) == 0:
        return np.zeros(len(q_dev), dtype=np.int64)
    trip = np.unique(np.stack([dev_dense, val_ids, ep], axis=1), axis=0)     # sorted by (dev, val, ep)
    d, v, e = trip[:, 0], trip[:, 1], trip[:, 2]
    same_next = np.zeros(len(trip), dtype=bool)
    same_next[:-1] = (d[1:] == d[:-1]) & (v[1:] == v[:-1])
    nxt = np.empty(len(trip), dtype=np.int64)
    nxt[:-1] = e[1:]
    nxt[-1] = 0
    end = np.where(same_next, np.minimum(e + W, nxt), e + W)
    L = E + W + 1
    diff = np.zeros(n_dev * L, dtype=np.int64)
    np.add.at(diff, d * L + e, 1)
    np.add.at(diff, d * L + end, -1)
    cov = np.cumsum(diff.reshape(n_dev, L), axis=1)
    return cov[q_dev, q_ep]


def rolling_batch(c: dict, epoch, W: int = W_DEFAULT) -> np.ndarray:
    """(N, 5) rolling features of ALL units of one stream/source at once (vectorised; equals the online
    ``RollingDeviceStats`` stepped through the epochs in order). ``epoch`` (N,) ≥ 0."""
    epoch = np.asarray(epoch, dtype=np.int64)
    n = len(epoch)
    if n == 0:
        return np.zeros((0, 5), dtype=np.int64)
    if epoch.min() < 0:
        raise ValueError("epochs must be >= 0")
    dev, remote, ipm, dpk, tu, sr, bp = _roll_inputs(c)
    udev, ddev = np.unique(dev, return_inverse=True)
    nd = len(udev)
    E = int(epoch.max()) + 1
    n_units = _window_count(ddev, epoch, W, ddev, epoch)
    n_sr = _window_count(ddev, epoch, W, ddev, epoch, mask=sr)
    n_bp = _window_count(ddev * 64 + bp, epoch, W, ddev * 64 + bp, epoch)
    # distinct remote IPs
    if ipm.any():
        pairs = _ip_to_u64pair(remote[ipm])
        _, rid = np.unique(pairs, axis=0, return_inverse=True)
        n_rdst = _window_distinct(ddev[ipm], rid.astype(np.int64).ravel(), epoch[ipm], W, nd, E, ddev, epoch)
    else:
        n_rdst = np.zeros(n, np.int64)
    if tu.any():
        _, pid = np.unique(dpk[tu], return_inverse=True)
        n_dp = _window_distinct(ddev[tu], pid.astype(np.int64).ravel(), epoch[tu], W, nd, E, ddev, epoch)
    else:
        n_dp = np.zeros(n, np.int64)
    return _finish_rolling(n_units, n_rdst, n_dp, n_sr, n_bp)


# ------------------------------------------------------------------------------------------ #
# public entry points
# ------------------------------------------------------------------------------------------ #
def _as_cols(units) -> dict:
    if isinstance(units, dict):
        return units
    units = list(units)
    if units and isinstance(units[0], (bytes, bytearray, memoryview)):
        return cols_from_meta([bytes(m) for m in units])
    return cols_from_units(units)


def segment_features(units, rolling_state, epoch: int | None = None) -> np.ndarray:
    """(N, 38) int32 features of ONE epoch's committed units.

    units: canonical cols dict, a list of ``Unit`` objects or a list of enc(m_i) bytes (all units of the
    epoch, in canonical order). rolling_state: a ``RollingDeviceStats`` (stepped here with ``epoch``,
    default = the units' epoch attribute) or a precomputed (N, 5) rolling matrix."""
    if not isinstance(units, dict) and epoch is None:
        units = list(units)
        if units and hasattr(units[0], "epoch"):
            epoch = int(units[0].epoch)
    c = _as_cols(units)
    X = static_features(c)
    if isinstance(rolling_state, RollingDeviceStats):
        if epoch is None:
            raise ValueError("segment_features: epoch needed to step the rolling state")
        R = rolling_state.step(int(epoch), c)
    else:
        R = np.asarray(rolling_state, dtype=np.int64)
        if R.shape != (len(X), 5):
            raise ValueError("rolling matrix must have shape (N, 5)")
    return _to_i32(np.concatenate([X, R], axis=1))


def features_batch(c: dict, epoch, W: int = W_DEFAULT) -> np.ndarray:
    """(N, 38) int32 features of a whole source/stream (units sorted by epoch, then index)."""
    n = len(c["pay"])
    out = np.empty((n, N_FEATURES), dtype=np.int32)
    out[:, :33] = static_features(c, wide=False)
    out[:, 33:] = _to_i32(rolling_batch(c, epoch, W))
    return out


_I32 = np.iinfo(np.int32)


def _to_i32(X: np.ndarray) -> np.ndarray:
    return np.clip(X, _I32.min, _I32.max).astype(np.int32)
