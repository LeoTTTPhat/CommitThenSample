"""Alg. 5 widest-path chains, chain losses and the chain baselines (naive, RT, composite)."""
import math

import numpy as np
import pytest

from certichain.certify.chains import (ccr, chain_family, chain_stats, composite_loss, maxchain_fdp, prefix_family,
                                       rt_chain_threshold, rt_index, rt_r_index, unit_positions, widest_chains)
from certichain.certify.crc import crc_threshold
from certichain.certify.losses import build_loss_tables
from certichain.certify.rules import LAMBDA_GRID
from certichain.eval import synth

# Hand-made DAG (unit ids = time order), candidate id = index:
#   c0: 0->2 s=5   c1: 1->2 s=7   c2: 2->3 s=6   c3: 2->4 s=4   c4: 3->5 s=9   c5: 4->5 s=8   c6: 0->6 s=3
# roots 0, 1 (best = +inf).  best[2] = max(min(inf,5), min(inf,7)) = 7 via c1
# best[3] = min(7,6) = 6 via c2;  best[4] = min(7,4) = 4 via c3
# best[5] = max(min(6,9), min(4,8)) = 6 via c4;  best[6] = min(inf,3) = 3 via c6
# sinks (in-edge, no out-edge): 5 (best 6), 6 (best 3) -> F^max = [(c1, c2, c4), (c6,)]
U = np.array([0, 1, 2, 2, 3, 4, 0])
V = np.array([2, 2, 3, 4, 5, 5, 6])
S = np.array([5, 7, 6, 4, 9, 8, 3], float)


def test_widest_path_hand_dag():
    ch = chain_family(np.arange(7), (U, V), S, J=20)
    assert ch == [(1, 2, 4), (6,)]
    assert chain_family(np.arange(7), (U, V), S, J=1) == [(1, 2, 4)]
    assert chain_family(np.array([], int), (U, V), S) == []
    # restricting S_e changes roots: without c1, unit 1 has no edge and 2 is reached from 0 only
    assert chain_family(np.array([0, 2, 3, 4, 5]), (U, V), S) == [(0, 2, 4)]


def test_tie_breaking_larger_edge_score_then_smaller_id():
    # e0: 0->1 s=6; e1: 1->3 s=10 (val = min(6,10) = 6); e2: 2->3 s=6 (val = 6): tie in val -> larger s (e1)
    u, v, s = np.array([0, 1, 2]), np.array([1, 3, 3]), np.array([6.0, 10.0, 6.0])
    assert chain_family(np.arange(3), (u, v), s) == [(0, 1)]
    # identical val and s -> smaller candidate id
    u, v, s = np.array([0, 1]), np.array([2, 2]), np.array([4.0, 4.0])
    assert chain_family(np.arange(2), (u, v), s) == [(0,)]
    assert chain_family(np.array([1, 0]), (u, v), s) == [(0,)]


def test_unit_times_order_and_backward_edges():
    # unit times reorder ids: 1 (t=1) -> 2 (t=3) -> 0 (t=5)
    ch = chain_family(np.arange(2), {"u": np.array([1, 2]), "v": np.array([2, 0]), "unit_ts": [5.0, 1.0, 3.0]},
                      np.array([1.0, 2.0]))
    assert ch == [(0, 1)]
    with pytest.raises(ValueError):
        chain_family(np.arange(1), (np.array([3]), np.array([1])), np.array([1.0]))
    np.testing.assert_array_equal(unit_positions(3, [5.0, 1.0, 3.0]), [2, 0, 1])


def test_J_keeps_the_widest_sinks():
    # 30 independent one-edge chains r_i -> s_i with score i; J = 20 keeps the 20 largest
    u = np.arange(0, 60, 2)
    v = u + 1
    s = np.arange(30, dtype=float)
    ch = chain_family(np.arange(30), (u, v), s, J=20)
    assert len(ch) == 20 and sorted(c[0] for c in ch) == list(range(10, 30))
    assert ch[0] == (29,)


def test_chain_losses_strict_lenient_prefix_ccr():
    chains = [(1, 2, 4), (6,)]
    y = np.array([1, 1, 1, 1, 1, 1, -1], np.int8)       # only c6 is AMB
    assert maxchain_fdp(chains, y) == 0.5
    assert maxchain_fdp(chains, y, strict=False) == 0.0  # AMB-only chain excluded
    y2 = y.copy(); y2[2] = 0
    assert maxchain_fdp(chains, y2) == 1.0
    assert prefix_family(chains) == {(1,), (1, 2), (1, 2, 4), (6,)}
    n, bad, bad0, pf_n, pf_bad, cov = chain_stats(chains, y2, [[1, 2, 4], [6], [3, -1]])
    assert (n, bad, bad0, pf_n, pf_bad, cov) == (2, 2, 1, 4, 3, 2)   # bad prefixes: (1,2), (1,2,4), (6,)
    assert ccr(chains, [[1, 2, 4], [6], [3, -1]]) == pytest.approx(2 / 3)
    assert math.isnan(ccr(chains, []))
    assert maxchain_fdp([], y) == 0.0


def test_rt_equivalences():
    rng = np.random.default_rng(3)
    t_grid = np.concatenate(([np.inf], np.sort(rng.normal(size=60))[::-1]))
    for _ in range(200):
        K = int(rng.integers(1, 60))
        first_bad = rng.integers(1, 62, size=K)          # 61 = never wrong
        tab = (np.arange(61)[None, :] >= first_bad[:, None]).astype(float)
        alpha = float(rng.choice([0.05, 0.1, 0.2, 0.33]))
        j = rt_index(tab, alpha)
        assert j == crc_threshold(tab, alpha)            # conformal quantile == CRC on the 0/1 loss
        r = t_grid[rt_r_index(tab)]
        assert rt_chain_threshold(r, alpha) == t_grid[j] or (j == 0 and np.isinf(rt_chain_threshold(r, alpha)))
    assert rt_index(np.zeros((5, 4)), 0.1) == 0          # alpha (K+1) < 1 -> abstain


def test_bank_chain_tables_consistent_with_direct_dp():
    cfg = synth.SynthConfig(mu=3.5, chains=True, hidden_frac=0.2, rc_miss=0.1, n_median=150)
    s0 = synth.synth_null_pool(30, 1, cfg)
    pool = synth.synth_pool(12, 2, cfg)
    bank = build_loss_tables(pool, s0)
    assert bank.has_chains
    for k in range(bank.n):
        inc = pool[k]
        order = np.lexsort((np.arange(inc.scores.size), -inc.scores))
        for j in (5, 20, 40, 60):
            R = int(bank.t("R")[k, j])
            ch = chain_family(order[:R], {"u": inc.u, "v": inc.v, "unit_ts": inc.unit_ts}, inc.scores)
            assert bank.t("chain_fdp")[k, j] == pytest.approx(maxchain_fdp(ch, inc.y))
            assert bank.t("chN")[k, j] == len(ch)
    Lc, Le, cf = bank.t("Lcmp"), bank.t("Lmono"), bank.t("chain_fdp")
    assert np.all(Lc >= Le - 1e-12) and np.all(Lc >= cf - 1e-12) and np.all(np.diff(Lc, axis=1) >= 0)
    np.testing.assert_allclose(Lc, composite_loss(bank.t("fdp_strict"), cf))
    fw = bank.t("fwer", "st")
    assert np.all(np.diff(fw, axis=1) >= 0) and set(np.unique(fw)) <= {0.0, 1.0}
    # widest_chains direct call agrees with the wrapper
    inc = pool[0]
    upos = unit_positions(len(inc.unit_ts), inc.unit_ts)
    sel = np.argsort(-inc.scores)[:30]
    assert widest_chains(inc.u[sel], inc.v[sel], inc.scores[sel], sel, upos) == \
        chain_family(sel, {"u": inc.u, "v": inc.v, "unit_ts": inc.unit_ts}, inc.scores)
    assert LAMBDA_GRID.size == bank.t("R").shape[1]
