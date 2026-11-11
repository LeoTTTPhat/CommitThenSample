"""Group-conformal e-values (Lemma A) and batch e-LOND (Alg. 6) edge cases."""
import numpy as np
import pytest

from certichain.certify import elond
from certichain.certify.evalues import ebh, ebh_k, group_evalue, group_evalues, lemma_a_evalue
from certichain.eval import synth


def test_group_evalue_and_lemma_a_formula():
    assert group_evalue([0.0, 0.5, 1.0]) == pytest.approx(4 / 2.5)
    np.testing.assert_allclose(group_evalues(np.array([[0.0, 1.0], [0.0, 1.0]])), [3.0, 1.0])
    assert lemma_a_evalue(0.5, np.zeros(9)) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        group_evalue([1.5])


def test_ebh_basic():
    # n = 4, alpha = 0.5: thresholds 8/k
    assert ebh([10, 0, 6.7, 0], 0.5).tolist() == [True, False, True, False]
    assert not ebh([7.9, 0, 0, 0], 0.5).any()
    assert ebh_k(np.array([2.0, 2.0, 2.0, 2.0]), 4, 0.5) == 4
    assert ebh([], 0.1).size == 0


def test_batch_elond_hand_example_relaxation():
    # K = 9; look 0: all Q = 0 -> E0 = 10; look 1: sum Q = 0.45 -> E1 = 10/1.45; alpha = 0.2, gamma = (0.5, 0.5)
    Q = np.zeros((9, 2)); Q[:, 1] = 0.05
    # incident A sizes (2, 3): look 0: 10*0.2*0.5*(0+2) = 2 >= 2 -> certify (D = 2)
    #                          look 1: 6.897*0.1*(2+3) = 3.45 >= 3 -> certify thanks to D (D = 5)
    # incident B sizes (0, 3): look 1 with D = 0: 6.897*0.1*3 = 2.07 < 3 -> skip
    dec = elond.batch_elond([[2, 3], [0, 3]], Q, 0.2, gamma=(0.5, 0.5))
    assert [(d.incident, d.look, d.certified, d.D_before) for d in dec] == [(0, 0, True, 0), (0, 1, True, 2), (1, 1, False, 0)]


def test_batch_elond_gateway_pooled_hand_example():
    # K = 99; E0 = 100 (all Q = 0), E1 = 100/(9+1) = 10; alpha = 0.2, gamma = (0.5, 0.5); w_n = 1/(n(n+1))
    Q = np.zeros((99, 2)); Q[:9, 1] = 1.0
    # n=1 (w=1/2): A look0: 100*0.2*0.25*2 = 10 >= 2 certify (D=2); look1: 10*0.2*0.25*(2+3) = 2.5 < 3 skip
    # n=2 (w=1/6): B look1: 10*0.2*(1/12)*(2+3) = 0.83 < 3 skip
    # n=3 (w=1/12): C look0: 100*0.2*(1/24)*(2+1) = 2.5 >= 1 certify, D = 3 (D carried over from A)
    dec = elond.batch_elond([[2, 3], [0, 3], [1, 0]], Q, 0.2, gamma=(0.5, 0.5), w="gateway")
    assert [(d.incident, d.look, d.certified, d.D_before) for d in dec] == \
        [(0, 0, True, 0), (0, 1, False, 2), (1, 1, False, 2), (2, 0, True, 2)]
    assert elond.gateway_weight(1) == 0.5 and sum(elond.gateway_weight(n) for n in range(1, 10000)) < 1


def test_precomputable_schedule_matches_rule_at_D0():
    rng = np.random.default_rng(0)
    alpha, gamma = 0.1, np.array(elond.GAMMA_LOOKS)
    for _ in range(200):
        K = int(rng.integers(5, 200))
        Q = rng.uniform(0, 0.02, size=(K, 8)) * (rng.random((K, 8)) < 0.3)
        sizes = rng.integers(1, 5, size=(1, 8))
        E = group_evalues(Q)
        # first certification with D = 0 at look l  <=>  Qbar_l <= alpha gamma_l - (1 - alpha gamma_l)/K
        sched = Q.mean(0) <= alpha * gamma - (1 - alpha * gamma) / K + 1e-12
        cert_D0 = E * alpha * gamma >= 1 - 1e-12
        np.testing.assert_array_equal(sched, cert_D0)
        p = elond.elond_paths(sizes, np.zeros_like(sizes), sizes, E, alpha, gamma)
        first = int(np.argmax(p["cert"][0])) if p["cert"][0].any() else -1
        assert first == (int(np.argmax(sched)) if sched.any() else -1)


def test_elond_paths_equals_batch_elond():
    rng = np.random.default_rng(1)
    Q = rng.uniform(0, 0.05, size=(150, 8)) * (rng.random((150, 8)) < 0.2)
    sizes = rng.integers(0, 4, size=(30, 8))
    E = group_evalues(Q)
    for pooled, w in ((False, None), (True, "gateway")):
        p = elond.elond_paths(sizes, np.zeros_like(sizes), sizes, E, 0.2, elond.GAMMA_LOOKS, pooled=pooled)
        dec = elond.batch_elond(sizes.tolist(), Q, 0.2, elond.GAMMA_LOOKS, w=w)
        cert = np.zeros_like(p["cert"])
        for d in dec:
            cert[d.incident, d.look] = d.certified
        np.testing.assert_array_equal(cert, p["cert"])
        assert np.all(np.diff(p["D"], axis=1) >= 0)


def test_look_index_boundaries():
    v = np.array([-10.0, 0.0, 300.0, 300.1, 600.0, 96 * 300.0, 96 * 300.0 + 1])
    np.testing.assert_array_equal(elond.look_index(v), [0, 0, 0, 1, 1, 7, -1])


def test_stream_bank_and_split_evaluation():
    cfg = synth.SynthConfig(mu=4.5, tau=0.3, shift_df=float("inf"), n_median=150)
    s0 = synth.synth_null_pool(60, 1, cfg)
    train = elond.build_stream_bank(synth.synth_stream_pool(80, 2, cfg), s0)
    bank = elond.build_stream_bank(synth.synth_stream_pool(120, 3, cfg), s0)
    assert bank.batch_R.shape == (120, 8, 61) and np.all(np.diff(bank.cum_R, axis=2) >= 0)
    assert np.all(bank.batch_N.sum(1) <= np.diff(bank.csr["offsets"]))
    theta = elond.choose_look_levels(train, 0.1)
    m = train.batch_fdp().mean(0)[np.arange(8), theta]
    assert np.all(m <= 0.1 * np.asarray(elond.GAMMA_LOOKS) / 3 + 1e-12)
    res = elond.evaluate_stream_split(bank, np.arange(60), np.arange(60, 120), 0.1, theta, order_seed=0)
    for k in ("elond", "elond_pooled", "crc_looks", "crc_naive_repeated", "crc_offline_end"):
        assert 0.0 <= res[k]["sup_fdp"] <= 1.0
        assert res[k]["sup_fdp"] >= res[k]["fdr_end"] - 1e-12
    u = elond.union_paths(bank, np.arange(60, 70), np.full(8, 40))
    assert np.all(np.diff(u["D"], axis=1) >= 0) and np.all(u["V"] <= u["D"])
