"""drand beacon rounds and epoch ↔ round alignment (DESIGN §2.1, §2.3, §3.6 Alg. 3, §3.8 V2).

Data: ``data/beacon`` (see its MANIFEST.md) — LoE mainnet ``default`` chain,
``pedersen-bls-chained``, period 30 s, genesis 1595431050, 3,000 consecutive rounds
2570006…2573005. Files are UNTRUSTED data: parsed with ``json`` only, every field type/length
checked.

* ``t(r) = genesis + (r − 1)·period`` (seconds).
* ``check_randomness(r)``: randomness == SHA-256(signature).
* ``check_chain(r)``: previous_signature(r) == signature(r − 1)  (None if r − 1 is not loaded).
* ``verify_bls(r)``: not available (py_ecc / blst not installed) → NotImplementedError.

Epoch alignment (§2.3): t_e = t(r0 + e) − Δ_c, so epoch e = [t_e, t_{e+1}); designated round
r*_e = r0 + e + 1 (``ahead=1``; the two-rounds-ahead option is ``ahead=2``), hence
t(r*_e) = t_{e+1} + Δ_c. With r0 = first loaded round, epochs e = 0..E−1 need rounds
r0+ahead … r0+E−1+ahead (and r*_e − 1 for the chain check), so E ≤ n_rounds − ahead:
2,999 epochs for ahead = 1 and 2,998 for ahead = 2 with the 3,000 loaded rounds.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from ..utils.io import DATA

DEFAULT_DIR = DATA / "beacon"
CHAIN_HASH_DEFAULT = "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce"

__all__ = ["BeaconRound", "DrandRounds", "EpochClock", "align_stream", "max_epochs",
           "check_round_randomness", "check_round_link", "DEFAULT_DIR"]


def _hex(v, n: int, name: str) -> bytes:
    if not isinstance(v, str):
        raise ValueError(f"{name}: expected hex string")
    b = bytes.fromhex(v)
    if len(b) != n:
        raise ValueError(f"{name}: expected {n} bytes, got {len(b)}")
    return b


@dataclass(frozen=True)
class BeaconRound:
    round: int
    randomness: bytes           # 32 B = SHA-256(signature)
    signature: bytes            # 96 B BLS (G2)
    previous_signature: bytes   # 96 B, signature of round − 1 (chained scheme)

    def to_dict(self) -> dict:
        return {"round": self.round, "randomness": self.randomness.hex(),
                "signature": self.signature.hex(), "previous_signature": self.previous_signature.hex()}

    @classmethod
    def from_dict(cls, d: dict) -> "BeaconRound":
        r = d.get("round")
        if not isinstance(r, int) or isinstance(r, bool) or r < 1:
            raise ValueError("round: expected positive int")
        return cls(r, _hex(d.get("randomness"), 32, "randomness"), _hex(d.get("signature"), 96, "signature"),
                   _hex(d.get("previous_signature"), 96, "previous_signature"))

    def chained_message(self) -> bytes:
        """BLS message of the chained scheme: SHA-256(previous_signature ‖ uint64_be(round))."""
        return hashlib.sha256(self.previous_signature + struct.pack(">Q", self.round)).digest()


def check_round_randomness(b: BeaconRound) -> bool:
    return hashlib.sha256(b.signature).digest() == b.randomness


def check_round_link(prev: BeaconRound, cur: BeaconRound) -> bool:
    return cur.round == prev.round + 1 and cur.previous_signature == prev.signature


class DrandRounds:
    """The loaded drand rounds plus chain parameters."""

    def __init__(self, data_dir: Path | str = DEFAULT_DIR, rounds: list[BeaconRound] | None = None,
                 genesis: int | None = None, period: int | None = None):
        data_dir = Path(data_dir)
        if rounds is None:
            info = json.loads((data_dir / "chain_info.json").read_text())
            self.genesis = int(info["genesis_time"])
            self.period = int(info["period"])
            self.chain_hash = str(info.get("hash", ""))
            self.scheme = str(info.get("schemeID", ""))
            self.public_key = _hex(info["public_key"], 48, "public_key")
            rounds = []
            with open(data_dir / "rounds.jsonl", "r", encoding="ascii") as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rounds.append(BeaconRound.from_dict(json.loads(line)))
                    except (ValueError, KeyError, TypeError) as e:
                        raise ValueError(f"rounds.jsonl line {ln}: {e}") from None
        else:
            if genesis is None or period is None:
                raise ValueError("genesis and period are required with explicit rounds")
            self.genesis, self.period = int(genesis), int(period)
            self.chain_hash, self.scheme, self.public_key = "", "", b""
        if genesis is not None:
            self.genesis = int(genesis)
        if period is not None:
            self.period = int(period)
        self._by_round: dict[int, BeaconRound] = {}
        for b in rounds:
            if b.round in self._by_round:
                raise ValueError(f"duplicate round {b.round}")
            self._by_round[b.round] = b
        if not self._by_round:
            raise ValueError("no beacon rounds")
        self.first_round = min(self._by_round)
        self.last_round = max(self._by_round)

    # -------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self._by_round)

    def __contains__(self, r: int) -> bool:
        return r in self._by_round

    def t(self, r: int) -> int:
        """Round release time in Unix seconds: genesis + (r − 1)·period."""
        return self.genesis + (int(r) - 1) * self.period

    def round_at(self, t_unix: float) -> int:
        """Latest round r with t(r) ≤ t_unix."""
        return int((t_unix - self.genesis) // self.period) + 1

    def get(self, r: int) -> BeaconRound:
        try:
            return self._by_round[int(r)]
        except KeyError:
            raise KeyError(f"beacon round {r} not loaded ({self.first_round}..{self.last_round})") from None

    def randomness(self, r: int) -> bytes:
        return self.get(r).randomness

    # -------------------------------------------------------------- checks
    def check_randomness(self, r: int) -> bool:
        return check_round_randomness(self.get(r))

    def check_chain(self, r: int) -> bool | None:
        """previous_signature(r) == signature(r − 1); None if r − 1 is not loaded."""
        if r - 1 not in self._by_round:
            return None
        return check_round_link(self._by_round[r - 1], self.get(r))

    def check_all(self) -> dict:
        rs = sorted(self._by_round)
        bad_rand = [r for r in rs if not self.check_randomness(r)]
        bad_link = [r for r in rs[1:] if self.check_chain(r) is False]
        gaps = [r for a, r in zip(rs, rs[1:]) if r != a + 1]
        return {"n": len(rs), "first": rs[0], "last": rs[-1], "bad_randomness": bad_rand,
                "bad_links": bad_link, "gaps": gaps,
                "ok": not bad_rand and not bad_link and not gaps}

    def verify_bls(self, r: int) -> bool:
        raise NotImplementedError("BLS12-381 verification needs py_ecc/blst, which is not installed")


@dataclass(frozen=True)
class EpochClock:
    """Epoch e ↔ drand round alignment of one gateway chain (§2.3).

    All times are Unix seconds (floats) or integer µs (``*_us``)."""
    r0: int
    genesis: int
    period: int = 30
    delta_c: float = 3.0
    ahead: int = 1

    @property
    def T(self) -> int:
        return self.period

    def t_round(self, r: int) -> int:
        return self.genesis + (int(r) - 1) * self.period

    def t_e(self, e: int) -> float:
        return self.t_round(self.r0 + e) - self.delta_c

    def t_e_us(self, e: int) -> int:
        return self.t_round(self.r0 + e) * 1_000_000 - round(self.delta_c * 1_000_000)

    def r_star(self, e: int) -> int:
        return self.r0 + e + self.ahead

    def t_rstar(self, e: int) -> int:
        return self.t_round(self.r_star(e))

    def deadline(self, e: int, delta_W: float = 0.3, eps_beta: float = 1.0) -> float:
        """Latest witness receive time τ_W allowed by TR: t(r*_e) − δ_W − ε_β."""
        return self.t_rstar(e) - delta_W - eps_beta

    def epoch_of(self, t_unix: float) -> int:
        """e with t_e ≤ t < t_{e+1}."""
        return int((t_unix + self.delta_c - self.t_round(self.r0)) // self.period)

    def to_dict(self) -> dict:
        return {"r0": self.r0, "genesis": self.genesis, "period": self.period,
                "delta_c": self.delta_c, "ahead": self.ahead}


def max_epochs(beacon: DrandRounds, ahead: int = 1, r0: int | None = None) -> int:
    """Largest E such that epochs 0..E−1 have r*_e and r*_e − 1 loaded (r0 = first round by default)."""
    r0 = beacon.first_round if r0 is None else r0
    if r0 + ahead - 1 < beacon.first_round:
        return 0
    return max(0, beacon.last_round - r0 - ahead + 1)


def align_stream(beacon: DrandRounds, n_epochs: int, ahead: int = 1, r0: int | None = None,
                 delta_c: float = 3.0) -> EpochClock:
    """Clock for a simulated stream of ``n_epochs`` epochs (e = 0..n−1) such that every
    designated round r*_e and its predecessor (chain check) are loaded."""
    if n_epochs < 1:
        raise ValueError("n_epochs must be >= 1")
    r0 = beacon.first_round if r0 is None else int(r0)
    E = max_epochs(beacon, ahead, r0)
    if n_epochs > E:
        raise ValueError(f"{n_epochs} epochs need rounds up to {r0 + n_epochs - 1 + ahead}; "
                         f"only {E} epochs fit with r0={r0} (last loaded round {beacon.last_round})")
    for e in (0, n_epochs - 1):
        r = r0 + e + ahead
        assert r in beacon and (r - 1) in beacon
    return EpochClock(r0=r0, genesis=beacon.genesis, period=beacon.period, delta_c=delta_c, ahead=ahead)
