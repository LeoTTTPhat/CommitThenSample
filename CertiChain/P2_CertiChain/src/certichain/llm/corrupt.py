"""E4 corruption models applied to s_LLM arrays (DESIGN §2.2 guarantee scope, E4).

Calibration-and-test corruptions (validity must survive them, Thm 1):
  * ``gaussian_noise``  s + σ·ε, σ ∈ {1, 2, 4}                    (hallucinating LLM)
  * ``invert``          s -> −s on a fraction f ∈ {0.5, 1}         (adversarially inverted LLM)
  * ``memorise``        s + δ on IN-BAND NULLS, δ ∈ {2, 5}         (GT-dependent memorisation adversary;
                        applied identically to calibration and test incidents)
Test-only manipulation (outside the guarantee; E4(iv) measures it):
  * ``inject_top``      q nulls per test incident forced to the top score of that incident

Randomness is keyed: with ``keys`` (e.g. (incident, cand_id) pairs) every candidate's noise /
inversion draw is a fixed function of (seed, key) via BLAKE3, so it does not depend on array
order or on which other candidates are present. Without keys a seeded numpy Generator over
the array order is used. NaN (= not in the LLM band) stays NaN everywhere.
"""
from __future__ import annotations

import zlib
from typing import Any, Sequence

import blake3
import numpy as np

E4_GRID: dict[str, tuple[float, ...]] = {"noise": (1.0, 2.0, 4.0), "invert": (0.5, 1.0), "memorise": (2.0, 5.0)}
_U64 = float(2 ** 64)


def keyed_uniform(keys: Sequence[Any], seed: int = 0, stream: str = "u") -> np.ndarray:
    """Uniform(0, 1) per key, a fixed function of (seed, stream, key); never exactly 0 or 1."""
    out = np.empty(len(keys), np.float64)
    tag = f"certichain.e4|{seed}|{stream}|".encode()
    for i, k in enumerate(keys):
        h = blake3.blake3(tag + repr(k).encode()).digest(8)
        out[i] = (int.from_bytes(h, "little") + 0.5) / _U64
    return out


def _normal(n: int, seed: int, keys: Sequence[Any] | None) -> np.ndarray:
    if keys is None:
        return np.random.default_rng(seed).standard_normal(n)
    if len(keys) != n:
        raise ValueError("keys and scores differ in length")
    u1, u2 = keyed_uniform(keys, seed, "n1"), keyed_uniform(keys, seed, "n2")
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)          # Box-Muller


def _uniform(n: int, seed: int, keys: Sequence[Any] | None, stream: str) -> np.ndarray:
    if keys is None:
        return np.random.default_rng([seed, zlib.crc32(stream.encode())]).random(n)   # stable across processes
    if len(keys) != n:
        raise ValueError("keys and scores differ in length")
    return keyed_uniform(keys, seed, stream)


def gaussian_noise(s: np.ndarray, sigma: float, seed: int = 0, keys: Sequence[Any] | None = None) -> np.ndarray:
    s = np.asarray(s, np.float64)
    return s + float(sigma) * _normal(len(s), seed, keys)


def invert(s: np.ndarray, frac: float, seed: int = 0, keys: Sequence[Any] | None = None) -> np.ndarray:
    s = np.asarray(s, np.float64)
    flip = _uniform(len(s), seed, keys, "inv") < float(frac) if frac < 1 else np.ones(len(s), bool)
    return np.where(flip, -s, s)


def memorise(s: np.ndarray, y: np.ndarray, delta: float, in_band: np.ndarray | None = None) -> np.ndarray:
    """+δ on in-band nulls (y == 0). In-band defaults to ~isnan(s). AMB (y = −1) and true links unchanged."""
    s = np.asarray(s, np.float64)
    y = np.asarray(y)
    band = ~np.isnan(s) if in_band is None else np.asarray(in_band, bool)
    return np.where(band & (y == 0), s + float(delta), s)


def inject_top(s: np.ndarray, y: np.ndarray, groups: Sequence[Any], q: int, seed: int = 0,
               keys: Sequence[Any] | None = None, margin: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Test-only: in each group (incident) force q nulls (y == 0, keyed-random choice) to the top.

    Their score becomes max(finite scores of the group) + margin (ties broken by the choice order),
    so they rank above every genuine candidate. Returns (new scores, injected mask)."""
    s = np.asarray(s, np.float64).copy()
    y = np.asarray(y)
    groups = np.asarray([str(g) for g in groups])
    u = _uniform(len(s), seed, keys, "inj")
    inj = np.zeros(len(s), bool)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        fin = s[idx][np.isfinite(s[idx])]
        top = (fin.max() if fin.size else 0.0) + margin
        nulls = idx[y[idx] == 0]
        pick = nulls[np.argsort(u[nulls], kind="stable")][: int(q)]
        s[pick] = top + margin * np.arange(len(pick))[::-1] / max(1, len(pick))
        inj[pick] = True
    return s, inj


def apply(name: str, s: np.ndarray, param: float, *, y: np.ndarray | None = None, seed: int = 0,
          keys: Sequence[Any] | None = None, in_band: np.ndarray | None = None) -> np.ndarray:
    """Dispatch for the E4 grid: name ∈ {noise, invert, memorise}."""
    if name == "noise":
        return gaussian_noise(s, param, seed, keys)
    if name == "invert":
        return invert(s, param, seed, keys)
    if name == "memorise":
        if y is None:
            raise ValueError("memorise needs GT labels y (applied identically to calibration and test)")
        return memorise(s, y, param, in_band)
    raise KeyError(name)
