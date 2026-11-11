"""scoring/fusion.py (DESIGN §3.4.2): incident-grouped cross-fitting, OOF without leakage, S_0, fusion inputs."""
from __future__ import annotations

import numpy as np
import pytest

from certichain.features import stats as F
from certichain.scoring import fusion as FU
from test_features_stats import synth_incident

FAST = dict(FU.LGB_PARAMS, num_threads=2, min_child_samples=5, num_leaves=7)


@pytest.fixture(scope="module")
def incs():
    out = []
    for s in range(20):
        units, side, cands, y = synth_incident(100 + s)
        feats = F.base_features(units, cands, side)
        out.append(FU.IncidentData(inc_id=f"inc{s:02d}", feats=feats, y=y))
    return out


def test_make_folds_deterministic_and_balanced():
    ids = [f"i{k}" for k in range(23)]
    a = FU.make_folds(ids, 5, 0)
    assert a == FU.make_folds(list(reversed(ids)), 5, 0)
    counts = np.bincount(list(a.values()), minlength=5)
    assert counts.max() - counts.min() <= 1
    assert a != FU.make_folds(ids, 5, 1)


def test_cross_fit_oof_and_null_pool(incs):
    res = FU.cross_fit(incs, F.FEATURES, params=FAST, max_rounds=40, es_rounds=10, log=lambda m: None)
    assert np.isfinite(res.oof).all() and res.oof.size == sum(i.n for i in incs)
    s0 = res.null_pool()
    assert s0.size == int((res.y != 1).sum()) and np.all(np.diff(s0) >= 0)
    scores = FU.split_by_incident(res.oof, res.sizes)
    auc = FU.within_auc(scores, [i.y for i in incs], "strict")
    assert auc["mean"] > 0.8                     # the planted pair is learnable
    s = FU.score_incident(res.booster, res.priors, incs[0], F.FEATURES)
    assert s.shape == (incs[0].n,) and np.isfinite(s).all()
    assert len(res.best_iters) == 5 and res.final_rounds >= 1
    fr = res.oof_frame()
    assert fr.height == res.oof.size and set(fr["fold"].unique().to_list()) <= set(range(5))


def test_oof_of_a_fold_ignores_its_own_labels(incs):
    res = FU.cross_fit(incs, F.FEATURES, params=FAST, max_rounds=40, es_rounds=10, log=lambda m: None)
    f0 = [k for k, i in enumerate(incs) if res.folds[i.inc_id] == 0]
    incs2 = []
    for k, i in enumerate(incs):
        y = i.y.copy()
        if k in f0:
            y = np.random.default_rng(k).permutation(y)
        incs2.append(FU.IncidentData(i.inc_id, i.feats, y))
    res2 = FU.cross_fit(incs2, F.FEATURES, params=FAST, max_rounds=40, es_rounds=10, log=lambda m: None)
    rows = np.isin(res.groups, [incs[k].inc_id for k in f0])
    assert np.allclose(res.oof[rows], res2.oof[rows])
    assert not np.allclose(res.oof[~rows], res2.oof[~rows])


def test_fusion_inputs_nan_outside_band(incs):
    with_llm = []
    for i in incs:
        rng = np.random.default_rng(len(with_llm))
        band = rng.random(i.n) < 0.3
        s_llm = np.where(i.y == 1, 3.0, -3.0) + rng.normal(0, 0.5, i.n)
        with_llm.append(FU.IncidentData(i.inc_id, i.feats, i.y, s_llm=s_llm, in_band=band))
    X = FU._extra_cols(with_llm)
    band = np.concatenate([i.in_band for i in with_llm])
    assert np.isnan(X[~band, 0]).all() and np.isfinite(X[band, 0]).all()
    assert np.array_equal(X[:, 1], band.astype(np.float32))
    names = (*F.FEATURES, *FU.FUSE_EXTRA)
    res = FU.cross_fit(with_llm, names, params=FAST, max_rounds=40, es_rounds=10, log=lambda m: None)
    assert res.with_extra and np.isfinite(res.oof).all()
    s = FU.score_incident(res.booster, res.priors, with_llm[0], names)
    assert s.shape == (with_llm[0].n,)
    with pytest.raises(ValueError):
        FU.cross_fit(with_llm, ("s_llm", *F.FEATURES), params=FAST, max_rounds=5, es_rounds=2, log=lambda m: None)


# ----------------------------------------------------------------------------- baselines B1 / B2 / B4 / B5
import polars as pl  # noqa: E402

from certichain.baselines.bothunter import bothunter_claims, unit_stages  # noqa: E402
from certichain.baselines.king import is_attack_type, king_claims  # noqa: E402
from certichain.baselines.lag_llr import lag_llr_claims, lag_llr_scores, oof_lag_llr  # noqa: E402
from certichain.baselines.llm_only import coverage, llm_only_claims, llm_scores  # noqa: E402
from certichain.features.priors import PairPriors  # noqa: E402
from certichain.io.schema import CANDIDATE_SCHEMA, conform  # noqa: E402
from test_features_stats import units_frame  # noqa: E402


def _chain():
    u = units_frame([
        dict(ts=0, te=50, src="8.8.1.1", dsts=["192.168.1.5"], type="SCAN_H", dir_class="ext->int"),        # E1
        dict(ts=60, te=90, src="8.8.1.1", dsts=["192.168.1.5"], type="BRUTE_TELNET", dir_class="ext->int"),  # E2
        dict(ts=100, te=120, src="192.168.1.5", dsts=["9.9.9.9"], type="HTTP_DOWNLOAD", dir_class="int->ext"),  # E3
        dict(ts=130, te=900, src="192.168.1.5", dsts=["7.7.7.7"], type="C2_PERIODIC", dir_class="int->ext"),  # E4
        dict(ts=200, te=300, src="192.168.1.5", dsts=["5.5.5.5"], type="SCAN_H", dir_class="int->ext"),       # E5
        dict(ts=210, te=220, src="192.168.1.7", dsts=["1.1.1.1"], type="DNS_ACTIVITY", dir_class="int->ext"),
    ])
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4), (4, 5), (0, 5)]
    c = conform(pl.DataFrame({"cand_id": list(range(len(pairs))), "u": [a for a, _ in pairs],
                              "v": [b for _, b in pairs], "rel_mask": [8] * len(pairs),
                              "lag_start": [10.0] * len(pairs), "lag_gap": [5.0] * len(pairs),
                              "overlap": [False] * len(pairs)}), CANDIDATE_SCHEMA)
    return u, c, pairs


def test_bothunter_consecutive_stages_of_one_host():
    u, c, pairs = _chain()
    assert [s for s, _ in unit_stages(u.row(0, named=True))] == ["E1"]
    assert unit_stages(u.row(3, named=True)) == [("E4", "192.168.1.5")]
    m = bothunter_claims(u, c, W=1800.0)
    got = {p for p, k in zip(pairs, m) if k}
    assert got == {(0, 1), (1, 2), (2, 3), (3, 4)}                    # E1->E2->E3->E4->E5, no skips, no benign
    tight = {p for p, k in zip(pairs, bothunter_claims(u, c, W=5.0)) if k}
    assert tight == {(3, 4)}               # window is t^s_v <= t^e_u + W: only the still-active C2 unit links


def test_king_tracks_from_attack_units():
    u, c, pairs = _chain()
    assert is_attack_type("SCAN_H") and not is_attack_type("DNS_ACTIVITY") and not is_attack_type("BENIGN_GENERIC_TCP")
    m = king_claims(u, c)
    assert m.all()                                                    # every link touches an attack-like unit
    only_benign = conform(pl.DataFrame({"cand_id": [0], "u": [5], "v": [5], "rel_mask": [4], "lag_start": [1.0],
                                        "lag_gap": [1.0], "overlap": [False]}), CANDIDATE_SCHEMA)
    assert not king_claims(u, only_benign).any()
    assert king_claims(u, c, max_hops=0).sum() == 0


def test_lag_llr_baseline_and_oof():
    units, side, cands, y = synth_incident(7)
    feats = F.base_features(units, cands, side)
    pr = PairPriors.fit(feats["type_u_str"].to_numpy(), feats["type_v_str"].to_numpy(), feats["lag_gap"].to_numpy(), y)
    s = lag_llr_scores(feats, pr)
    assert s.shape == (feats.height,) and np.array_equal(lag_llr_claims(s), s > 0)
    incs = [synth_incident(30 + k) for k in range(6)]
    fl = [F.base_features(a, c, b) for a, b, c, _ in incs]
    ids = [f"i{k}" for k in range(6)]
    folds = {f"i{k}": k % 3 for k in range(6)}
    oof = oof_lag_llr(fl, [yy for *_, yy in incs], ids, folds)
    assert [o.size for o in oof] == [f.height for f in fl]
    y2 = [yy.copy() for *_, yy in incs]
    y2[0] = np.random.default_rng(1).permutation(y2[0])               # incident 0's own labels never reach it
    assert np.allclose(oof_lag_llr(fl, y2, ids, folds)[0], oof[0])


def test_llm_only_claims_and_coverage():
    s = llm_scores([0, 1, 2, 3], {0: 1.5, 1: -0.2, 3: 0.0})
    assert np.isnan(s[2]) and coverage(s) == 0.75
    assert llm_only_claims(s).tolist() == [True, False, False, False]
