"""Selection PRF (DESIGN §3.1 M5, §3.6).

    K_sel = BLAKE3-derive_key(context="CTS/v1 selection key", material=β_vrf)      (32 B)
    u_i   = u64le( BLAKE3-keyed(K_sel, b"sel" ‖ u32le(e) ‖ u32le(i) ‖ L_i)[:8] )
    Z_i   = [u_i < τ_i]

β_vrf is the 64-byte ECVRF output (proof_to_hash of Π_e, i.e. ρ_e of the DESIGN).
"""
from __future__ import annotations

import struct
from collections.abc import Sequence

import numpy as np
from blake3 import blake3

SEL_CONTEXT = "CTS/v1 selection key"
_EI = struct.Struct("<II")

__all__ = ["selection_key", "u_word", "u_words", "u_words_np", "select_units", "SEL_CONTEXT"]


def selection_key(beta_vrf: bytes) -> bytes:
    """K_sel = blake3.blake3(beta_vrf, derive_key_context="CTS/v1 selection key").digest()."""
    return blake3(beta_vrf, derive_key_context=SEL_CONTEXT).digest()


def u_word(K_sel: bytes, e: int, i: int, leaf: bytes) -> int:
    return int.from_bytes(blake3(b"sel" + _EI.pack(e, i) + leaf, key=K_sel).digest()[:8], "little")


def u_words(K_sel: bytes, e: int, leaves: Sequence[bytes], indices: Sequence[int] | None = None) -> list[int]:
    """u_i for i in ``indices`` (default: all positions of ``leaves``; leaves[k] is L of index k)."""
    if indices is None:
        indices = range(len(leaves))
    pk = _EI.pack
    fb = int.from_bytes
    return [fb(blake3(b"sel" + pk(e, i) + leaves[i], key=K_sel).digest()[:8], "little") for i in indices]


def u_words_np(K_sel: bytes, e: int, leaves: Sequence[bytes], indices: Sequence[int] | None = None) -> np.ndarray:
    return np.array(u_words(K_sel, e, leaves, indices), dtype=np.uint64)


def select_units(K_sel: bytes, e: int, leaves: Sequence[bytes], taus: Sequence[int],
                 pay: Sequence[bool]) -> tuple[list[int], list[bool]]:
    """(u, Z) for all units; zero-payload units get u = 0 placeholder and Z = False."""
    u, Z = [], []
    pk = _EI.pack
    fb = int.from_bytes
    for i, (leaf, t, p) in enumerate(zip(leaves, taus, pay, strict=True)):
        if p:
            ui = fb(blake3(b"sel" + pk(e, i) + leaf, key=K_sel).digest()[:8], "little")
            u.append(ui)
            Z.append(ui < t)
        else:
            u.append(0)
            Z.append(False)
    return u, Z
