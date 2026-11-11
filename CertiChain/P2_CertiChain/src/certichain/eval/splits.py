"""Calibration/test partitions of an incident pool (DESIGN §6.1, §6.4, §3.10).

* random_splits: split seed s -> np.random.default_rng((s, salt)).permutation(n_pool); the first K
  positions calibrate, the rest (or the next n_test) are test incidents. Seeds 0-199 by default.
* loso_splits: leave-one-scenario-out (IoT-23 P-LOSO): test = all incidents of one group,
  calibration = all incidents of the other groups (or K of them drawn at random).

Splits hold POSITIONS into a pool array (so the incident bootstrap can pass a resampled pool
with duplicates); padded with -1 when sizes vary (LOSO).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Splits:
    seeds: np.ndarray        # [S] split seed (or group id for LOSO)
    cal: np.ndarray          # [S, Kmax] positions, -1 padded
    test: np.ndarray         # [S, Mmax] positions, -1 padded

    @property
    def S(self) -> int:
        return int(self.seeds.size)

    @property
    def cal_mask(self) -> np.ndarray:
        return self.cal >= 0

    @property
    def test_mask(self) -> np.ndarray:
        return self.test >= 0

    @property
    def K(self) -> np.ndarray:
        return self.cal_mask.sum(axis=1)

    def __iter__(self):
        for i in range(self.S):
            c, t = self.cal[i], self.test[i]
            yield int(self.seeds[i]), c[c >= 0], t[t >= 0]


def _pad(rows: Sequence[np.ndarray]) -> np.ndarray:
    m = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), m), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def random_splits(n_pool: int, K: int, seeds: Iterable[int] = range(200), n_test: int | None = None,
                  salt: int = 0) -> Splits:
    seeds = np.asarray(list(seeds), dtype=np.int64)
    if not 0 < K < n_pool:
        raise ValueError(f"need 0 < K < n_pool (K={K}, n_pool={n_pool})")
    M = n_pool - K if n_test is None else int(n_test)
    if K + M > n_pool:
        raise ValueError("K + n_test exceeds the pool")
    cal = np.empty((seeds.size, K), dtype=np.int64)
    test = np.empty((seeds.size, M), dtype=np.int64)
    for i, s in enumerate(seeds):
        perm = np.random.default_rng((int(s), int(salt))).permutation(n_pool)
        cal[i] = perm[:K]
        test[i] = perm[K: K + M]
    return Splits(seeds, cal, test)


def loso_splits(groups: Sequence, K: int | None = None, seed: int = 0) -> Splits:
    """One split per distinct group value g: test = {i : groups[i] == g}, cal = the others
    (all of them, or a random subset of size K)."""
    g = np.asarray(groups)
    vals = list(dict.fromkeys(g.tolist()))
    cal_rows, test_rows = [], []
    rng = np.random.default_rng(seed)
    for v in vals:
        t = np.flatnonzero(g == v)
        c = np.flatnonzero(g != v)
        if K is not None and K < c.size:
            c = np.sort(rng.choice(c, size=K, replace=False))
        cal_rows.append(c)
        test_rows.append(t)
    return Splits(np.arange(len(vals), dtype=np.int64), _pad(cal_rows), _pad(test_rows))
