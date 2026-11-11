"""Fixed-width little-endian metadata records m_i (DESIGN §3.2).

Two layouts, selected by ``pay`` and distinguished on decode by length:

Payload units (138 B; DESIGN "≈128 B")::

    off  size  field
      0    37  key = proto u8 | ip_a 16 B | port_a u16 | ip_b 16 B | port_b u16
     37     4  device_id u32
     41    16  first_ts_us u64, last_ts_us u64
     57    24  pkts[0], pkts[1], payload_bytes[0], payload_bytes[1], frame_bytes[0], frame_bytes[1]  (u32)
     81    10  n_syn, n_synack, n_fin, n_rst, n_psh  (u16)
     91     1  conn_state u8 (CONN_STATES code)
     92    16  first-8 signed payload sizes (i16; + = dir 0, - = dir 1, 0 = no payload; |x| saturates at 32767)
    108     8  first-8 log2 IAT codes (u8; see iat_code; 255 = no such gap)
    116     4  s u32 (Σ caplen + 16)
    120    16  flow_uid
    136     1  continuation u8
    137     1  fps u8

Zero-payload units, compact (68 B; DESIGN "≈56 B")::

    key (37) | device_id u32 | first_ts_us u64 | pkts[0], pkts[1] u32 | 5 flag counts u16 | conn_state u8

Counters saturate instead of wrapping. Sizes differ from the DESIGN estimates only because
addresses are IPv6-mapped (2 × 16 B) and packet counters are u32.
"""
from __future__ import annotations

import math
import struct

KEY = struct.Struct("<B16sH16sH")                                   # 37 B
PAY = struct.Struct("<B16sH16sH" "I" "QQ" "6I" "5H" "B" "8h" "8B" "I" "16s" "B" "B")
ZERO = struct.Struct("<B16sH16sH" "I" "Q" "2I" "5H" "B")
assert KEY.size == 37 and PAY.size == 138 and ZERO.size == 68

PAY_LEN = PAY.size
ZERO_LEN = ZERO.size

# Zeek-like conn_state codes (u8)
CONN_STATES = ["S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "OTH"]
CS_CODE = {s: i for i, s in enumerate(CONN_STATES)}

U32 = 0xFFFFFFFF
U16 = 0xFFFF
IAT_MISSING = 255


def iat_code(gap_us: int) -> int:
    """u8 code = floor(8·log2(1 + gap_us)), i.e. 1/8-octave log2 bins of the gap in µs (254 max)."""
    if gap_us <= 0:
        return 0
    return min(254, int(8.0 * math.log2(1.0 + gap_us)))


def iat_code_lower_us(code: int) -> float:
    """Smallest gap (µs) that maps to ``code`` (for decoding/plots)."""
    return 2.0 ** (code / 8.0) - 1.0


def make_key(proto: int, ip_a: bytes, port_a: int, ip_b: bytes, port_b: int) -> bytes:
    return KEY.pack(proto, ip_a, port_a & U16, ip_b, port_b & U16)


def split_key(key: bytes) -> tuple[int, bytes, int, bytes, int]:
    return KEY.unpack(key)


def _u32(x: int) -> int:
    return x if x <= U32 else U32


def _u16(x: int) -> int:
    return x if x <= U16 else U16


def _i16(x: int) -> int:
    return 32767 if x > 32767 else (-32767 if x < -32767 else x)


def enc_metadata(m: dict) -> bytes:
    """Encode a metadata record. ``m`` holds the keys listed in the module docstring
    (key as the 37-byte key, flags as n_syn..n_psh, sizes/iats as length-8 sequences)."""
    k = KEY.unpack(m["key"])
    flags = (_u16(m["n_syn"]), _u16(m["n_synack"]), _u16(m["n_fin"]), _u16(m["n_rst"]), _u16(m["n_psh"]))
    if m["pay"]:
        sizes = [_i16(x) for x in m["sizes"]] + [0] * (8 - len(m["sizes"]))
        iats = list(m["iats"]) + [IAT_MISSING] * (8 - len(m["iats"]))
        return PAY.pack(*k, _u32(m["device_id"]), m["first_ts_us"], m["last_ts_us"],
                        _u32(m["pkts"][0]), _u32(m["pkts"][1]),
                        _u32(m["payload_bytes"][0]), _u32(m["payload_bytes"][1]),
                        _u32(m["frame_bytes"][0]), _u32(m["frame_bytes"][1]),
                        *flags, m["conn_state"], *sizes[:8], *iats[:8], _u32(m["s"]),
                        m["flow_uid"], int(bool(m["continuation"])), int(bool(m["fps"])))
    return ZERO.pack(*k, _u32(m["device_id"]), m["first_ts_us"], _u32(m["pkts"][0]), _u32(m["pkts"][1]),
                     *flags, m["conn_state"])


def dec_metadata(b: bytes) -> dict:
    if len(b) == PAY_LEN:
        v = PAY.unpack(b)
        return {"pay": True, "key": KEY.pack(*v[0:5]), "proto": v[0], "ip_a": v[1], "port_a": v[2],
                "ip_b": v[3], "port_b": v[4], "device_id": v[5], "first_ts_us": v[6], "last_ts_us": v[7],
                "pkts": (v[8], v[9]), "payload_bytes": (v[10], v[11]), "frame_bytes": (v[12], v[13]),
                "n_syn": v[14], "n_synack": v[15], "n_fin": v[16], "n_rst": v[17], "n_psh": v[18],
                "conn_state": v[19], "sizes": tuple(v[20:28]), "iats": tuple(v[28:36]), "s": v[36],
                "flow_uid": v[37], "continuation": bool(v[38]), "fps": bool(v[39])}
    if len(b) == ZERO_LEN:
        v = ZERO.unpack(b)
        return {"pay": False, "key": KEY.pack(*v[0:5]), "proto": v[0], "ip_a": v[1], "port_a": v[2],
                "ip_b": v[3], "port_b": v[4], "device_id": v[5], "first_ts_us": v[6],
                "pkts": (v[7], v[8]), "n_syn": v[9], "n_synack": v[10], "n_fin": v[11], "n_rst": v[12],
                "n_psh": v[13], "conn_state": v[14]}
    raise ValueError(f"metadata record of unexpected length {len(b)}")
