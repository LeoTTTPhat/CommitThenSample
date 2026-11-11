"""Retention encryption (DESIGN §3.7): ChaCha20-Poly1305 (RFC 8439) via ``cryptography``.

    K_e   = HKDF-SHA256(IKM = escrow_master, salt = none, info = b"CTS/v1 epoch key" ‖ u64le(e), L = 32)
    nonce = u32le(e) ‖ u64le(i)                      (12 bytes)
    AD    = b"CTS/v1/ret" ‖ u32le(e) ‖ u64le(i)

Deviation from DESIGN §3.7 (which names XChaCha20-Poly1305): every key K_e is used for exactly one
epoch and the nonce (e, i) is unique per retained unit under that key, so a 96-bit deterministic
nonce never repeats and the extended-nonce variant is unnecessary. (Re-encrypting the SAME
plaintext for the same (e, i) is harmless; encrypting a DIFFERENT plaintext for the same (e, i)
must never happen — the store writes each unit once.)
"""
from __future__ import annotations

import struct

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

INFO_PREFIX = b"CTS/v1 epoch key"
AD_PREFIX = b"CTS/v1/ret"
TAG_LEN = 16

__all__ = ["epoch_key", "nonce", "seal", "open_sealed", "EpochCipher", "TAG_LEN"]


def epoch_key(escrow_master: bytes, e: int) -> bytes:
    if len(escrow_master) < 16:
        raise ValueError("escrow master secret too short")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=INFO_PREFIX + struct.pack("<Q", e)).derive(escrow_master)


def nonce(e: int, i: int) -> bytes:
    return struct.pack("<IQ", e, i)


def _ad(e: int, i: int) -> bytes:
    return AD_PREFIX + struct.pack("<IQ", e, i)


def seal(K_e: bytes, e: int, i: int, plaintext: bytes) -> bytes:
    return ChaCha20Poly1305(K_e).encrypt(nonce(e, i), plaintext, _ad(e, i))


def open_sealed(K_e: bytes, e: int, i: int, ciphertext: bytes) -> bytes:
    """Decrypt; raises cryptography.exceptions.InvalidTag on any tampering."""
    return ChaCha20Poly1305(K_e).decrypt(nonce(e, i), ciphertext, _ad(e, i))


class EpochCipher:
    """Caches K_e per epoch for one escrow master secret."""

    def __init__(self, escrow_master: bytes):
        self._master = escrow_master
        self._keys: dict[int, bytes] = {}

    def key(self, e: int) -> bytes:
        k = self._keys.get(e)
        if k is None:
            k = self._keys[e] = epoch_key(self._master, e)
        return k

    def seal(self, e: int, i: int, plaintext: bytes) -> bytes:
        return seal(self.key(e), e, i, plaintext)

    def open(self, e: int, i: int, ciphertext: bytes) -> bytes:
        return open_sealed(self.key(e), e, i, ciphertext)
