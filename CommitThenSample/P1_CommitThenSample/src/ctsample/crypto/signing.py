"""Ed25519 signatures (RFC 8032) via ``cryptography``, with deterministic keys from 32-byte seeds.

Keys of the protocol (DESIGN §2.1):
* sk_G — gateway header signatures σ_e;
* sk_W — witness cosignatures ω_e and witness-log registrations;
* sk_I — independent counter source records;
* sk_V — the ECVRF key. It is a separate 32-byte seed used ONLY by crypto/ecvrf.py (never to sign);
* k_st — the gateway's long-term state key: salt_e = PRF_{k_st}(e) for the hiding state commitment
  C^st_e of the header (protocol/header.py). Never used for anything else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from blake3 import blake3
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import ecvrf

SIG_LEN = 64
PK_LEN = 32

__all__ = ["SigningKey", "KeySet", "sign", "verify", "derive_seed", "SIG_LEN", "PK_LEN"]


def derive_seed(master: bytes, label: str) -> bytes:
    """Deterministic 32-byte seed for ``label`` from a master secret (tests / simulations)."""
    return blake3(master + b"\x00" + label.encode(), derive_key_context="CTS/v1 deterministic key seed").digest()


@dataclass(frozen=True)
class SigningKey:
    seed: bytes = field(repr=False)
    _sk: Ed25519PrivateKey = field(init=False, repr=False, compare=False)
    pk: bytes = field(init=False, compare=False)

    def __post_init__(self):
        if len(self.seed) != 32:
            raise ValueError("Ed25519 seed must be 32 bytes")
        sk = Ed25519PrivateKey.from_private_bytes(self.seed)
        object.__setattr__(self, "_sk", sk)
        object.__setattr__(self, "pk", sk.public_key().public_bytes_raw())

    def sign(self, msg: bytes) -> bytes:
        return self._sk.sign(msg)

    def verify(self, sig: bytes, msg: bytes) -> bool:
        return verify(self.pk, sig, msg)


def sign(sk: SigningKey | bytes, msg: bytes) -> bytes:
    if isinstance(sk, (bytes, bytearray)):
        sk = SigningKey(bytes(sk))
    return sk.sign(msg)


def verify(pk: bytes, sig: bytes, msg: bytes) -> bool:
    """True iff ``sig`` is a valid Ed25519 signature of ``msg`` under ``pk`` (never raises)."""
    if len(pk) != PK_LEN or len(sig) != SIG_LEN:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(pk).verify(sig, msg)
        return True
    except (InvalidSignature, ValueError):
        return False


@dataclass(frozen=True)
class KeySet:
    """All protocol keys of one simulated deployment."""
    sk_G: SigningKey
    sk_W: SigningKey
    sk_I: SigningKey
    sk_V: bytes = field(repr=False)          # ECVRF secret (32-byte seed)
    k_st: bytes | None = field(default=None, repr=False)   # state-commitment salt key (32 B)
    pk_V: bytes = field(init=False)

    def __post_init__(self):
        if len(self.sk_V) != 32:
            raise ValueError("VRF secret must be 32 bytes")
        object.__setattr__(self, "pk_V", ecvrf.public_key(self.sk_V))
        if self.k_st is None:      # independent key via the BLAKE3 KDF (assumption (K))
            object.__setattr__(self, "k_st", blake3(self.sk_G.seed, derive_key_context="CTS/v1 state key").digest())
        if len(self.k_st) != 32:
            raise ValueError("state key must be 32 bytes")

    @classmethod
    def from_master(cls, master: bytes) -> "KeySet":
        return cls(sk_G=SigningKey(derive_seed(master, "sk_G")), sk_W=SigningKey(derive_seed(master, "sk_W")),
                   sk_I=SigningKey(derive_seed(master, "sk_I")), sk_V=derive_seed(master, "sk_V"),
                   k_st=derive_seed(master, "k_st"))

    @property
    def pk_G(self) -> bytes:
        return self.sk_G.pk

    @property
    def pk_W(self) -> bytes:
        return self.sk_W.pk

    @property
    def pk_I(self) -> bytes:
        return self.sk_I.pk
