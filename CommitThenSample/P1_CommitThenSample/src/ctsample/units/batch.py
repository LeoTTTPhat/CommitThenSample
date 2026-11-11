"""Shared unit record: the contract between the flow table (units/flowtable.py) and the
protocol layer (protocol/epoch.py, verifier, counters). DESIGN §3.2.

A *unit* is all packets of one bidirectional flow key inside one epoch [t_e, t_{e+1}).
Payload units (pay=True) carry content commitments; zero-payload units are metadata only.

Byte layouts:
* ``key``  (37 B): proto u8 | ip_a 16 B | port_a u16 LE | ip_b 16 B | port_b u16 LE, where
  (ip_a, port_a) is the flow INITIATOR (source of the flow's first packet as seen by the
  gateway's flow table); IPv4 is mapped to IPv6 (::ffff:a.b.c.d). ICMP: port_a = type,
  port_b = 0. Other IP protocols: ports 0. Non-IP L2 units: proto = 0, ip_a/ip_b hold the
  src MAC (6 B, zero-padded) and port_a = ethertype.
* ``meta`` = enc(m_i) from units/encode.py (payload layout ≈128 B, compact zero-payload
  layout ≈56 B). The protocol layer treats it as opaque bytes.
* ``pkt_leaves`` = [ℓ_ij] with ℓ_ij = BLAKE3("CTS/v1/frm" ‖ u64 ts_us ‖ u8 dir ‖ u32 caplen ‖
  BLAKE3(frame_ij)), packet order; ``root_pkt`` = Merkle root of pkt_leaves.
* ``leaf`` = L_i = BLAKE3("CTS/v1/leaf" ‖ meta ‖ root_pkt) for payload units,
  BLAKE3("CTS/v1/leaf0" ‖ meta) for zero-payload units (crypto/hashing.py computes both).

Directions: dir = 0 initiator→responder, 1 responder→initiator.
Cost: s = Σ_j (caplen_ij + 16) bytes (exact pcap-record size).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Frame:
    ts_us: int
    dir: int            # 0 = initiator->responder, 1 = reverse
    caplen: int
    wirelen: int
    data: bytes         # the whole captured frame (L2..), exactly as in the pcap


@dataclass
class Unit:
    epoch: int
    index: int                      # canonical rank in epoch: order by (first_ts_us, key bytes)
    key: bytes                      # 37-byte canonical flow key (see module docstring)
    device_id: int                  # dense index of the LAN-side MAC (u32)
    first_ts_us: int
    last_ts_us: int
    pkts: tuple[int, int]           # packets per direction (dir0, dir1)
    payload_bytes: tuple[int, int]  # transport payload bytes per direction
    frame_bytes: tuple[int, int]    # captured frame bytes per direction (Σ caplen)
    s: int                          # Σ (caplen + 16)
    pay: bool                       # Σ transport payload > 0
    fps: bool                       # first payload segment of its flow_uid
    flow_uid: bytes                 # BLAKE3("CTS/v1/flow" ‖ key ‖ first_ts)[:16]
    meta: bytes                     # enc(m_i)
    leaf: bytes = b""               # L_i (filled by crypto.hashing)
    root_pkt: bytes | None = None   # payload units only
    pkt_leaves: list[bytes] | None = None
    frames: list[Frame] | None = field(default=None, repr=False)  # payload units, until selection

    @property
    def n_pkts(self) -> int:
        return self.pkts[0] + self.pkts[1]


@dataclass
class EpochBatch:
    """All committed units of one epoch, in canonical order (index = position)."""
    epoch: int
    t_start_us: int
    units: list[Unit]

    @property
    def n(self) -> int:
        return len(self.units)

    @property
    def n_pay(self) -> int:
        return sum(1 for u in self.units if u.pay)
