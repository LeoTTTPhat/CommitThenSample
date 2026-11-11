"""B1 — King-style backward/forward tracking [KING05] on the candidate graph (DESIGN §6.3).

Start points: every unit with an attack-like type (not a benign-protocol / benign-generic type). The tracker
walks entity relations within W — i.e. candidate links, which are exactly the label-free entity relations
of Alg. 2 inside the window — BACKWARD (to predecessors) and FORWARD (to successors) and claims every link it
traverses. Expansion continues through attack-like units only (``expand_benign=False``, default: a benign unit
is reached and its link claimed, but it does not start further hops); ``expand_benign=True`` follows links
through any unit (King's unrestricted dependency closure; ``max_hops`` bounds it). No guarantee (uncalibrated).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import polars as pl

from certichain.units.unit_types import BENIGN_PROTOCOL_TYPES


def is_attack_type(t) -> bool:
    return t is not None and t not in BENIGN_PROTOCOL_TYPES and not str(t).startswith("BENIGN")


def king_claims(units: pl.DataFrame, cands: pl.DataFrame, *, expand_benign: bool = False,
                max_hops: int | None = None) -> np.ndarray:
    """Claim mask over ``cands`` (row order) of the King-style tracker."""
    if cands.height == 0:
        return np.zeros(0, dtype=bool)
    types = dict(zip(units["unit_id"].to_list(), units["type"].to_list()))
    u = cands["u"].to_numpy()
    v = cands["v"].to_numpy()
    inc: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(zip(u.tolist(), v.tolist())):
        inc.setdefault(a, []).append(k)
        inc.setdefault(b, []).append(k)
    start = [x for x in sorted(types) if is_attack_type(types[x])]
    hops = {x: 0 for x in start}
    q = deque(start)
    claimed = np.zeros(cands.height, dtype=bool)
    while q:
        x = q.popleft()
        h = hops[x]
        if max_hops is not None and h >= max_hops:
            continue
        for k in inc.get(x, ()):
            claimed[k] = True
            y = int(v[k]) if int(u[k]) == x else int(u[k])
            if y not in hops and (expand_benign or is_attack_type(types.get(y))):
                hops[y] = h + 1
                q.append(y)
    return claimed
