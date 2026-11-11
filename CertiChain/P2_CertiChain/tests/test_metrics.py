"""Metrics, splits, runner (outputs + resume), bootstrap/Holm, prereg, gates, B7/B8 edge cases."""
import json

import numpy as np
import polars as pl
import pytest

from certichain.certify.crc import crc_threshold
from certichain.certify.losses import Incident, build_loss_tables
from certichain.eval import bootstrap, gates, metrics, prereg, runner, splits, synth
from certichain.scoring.calibrate import IsotonicB7, PlattB8, b7_R, platt_fit


def test_fdp_definitions_edge_cases():
    np.testing.assert_allclose(metrics.fdp_strict([0, 3, 2], [0, 3, 4]), [0.0, 1.0, 0.5])
    # all-AMB selection: strict 1, lenient 0 (denominator max(R - A, 1))
    np.testing.assert_allclose(metrics.fdp_lenient([3], [3], [3]), [0.0])
    np.testing.assert_allclose(metrics.fdp_lenient([2], [1], [4]), [1 / 3])


def test_realised_vs_pooled_and_padding():
    V = np.array([[1, 0, 9]]); R = np.array([[1, 9, 9]]); mask = np.array([[True, True, False]])
    fs = metrics.fdp_strict(V, R)
    assert metrics.realised_fdr(fs, mask)[0] == pytest.approx(0.5)      # (1 + 0)/2, padded entry ignored
    assert metrics.pooled_fdr(V, R, mask)[0] == pytest.approx(0.1)      # 1/10
    assert metrics.exceedance(fs, 0.2, mask)[0] == pytest.approx(0.5)
    assert np.isnan(metrics.realised_fdr(fs, np.zeros_like(mask))[0])


def test_ratio_load_wrongful():
    num, den = np.array([[1.0, 0.0, 5.0]]), np.array([[2.0, 0.0, np.nan]])
    m = np.ones((1, 3), bool)
    assert metrics.ratio_mean(num, den, m)[0] == pytest.approx(0.5)     # zero / NaN denominators skipped
    assert metrics.ratio_pooled(num[:, :2], den[:, :2], m[:, :2])[0] == pytest.approx(0.5)
    assert np.isinf(metrics.analyst_load(np.array([[3]]), np.array([[0]]), m[:, :1])[0])
    assert np.isnan(metrics.analyst_load(np.array([[0]]), np.array([[0]]), m[:, :1])[0])
    assert metrics.analyst_load(np.array([[4, 2]]), np.array([[2, 1]]), m[:, :2])[0] == pytest.approx(2.0)
    benign = np.array([[True, True, False]])
    assert metrics.wrongful_implication(np.array([[1, 0, 5]]), benign, m)[0] == pytest.approx(0.5)


def test_chain_metrics_and_cert_gap():
    ch = {k: np.array([[2, 0]]) for k in ("chN",)}
    ch.update(chBad=np.array([[1, 0]]), chBad0=np.array([[0, 0]]), pfN=np.array([[4, 0]]), pfBad=np.array([[1, 0]]),
              ccr=np.array([[1, 0]]))
    out = metrics.chain_metrics(ch, np.array([[2.0, 0.0]]), np.ones((1, 2), bool))
    assert out["maxchain_fdr"][0] == pytest.approx(0.25)    # (1/2 + 0)/2
    assert out["maxchain_fdr_lenient"][0] == pytest.approx(0.0)
    assert out["ccr_mean"][0] == pytest.approx(0.5)
    f = np.array([[[0.0, 0.1, 0.5]]]); r = np.array([[[0.0, 0.4, 0.9]]])
    assert metrics.certification_gap(f, r, np.array([1]), np.ones((1, 1), bool))[0] == pytest.approx(0.0)
    assert metrics.certification_gap(f, r, np.array([0]), np.ones((1, 1), bool))[0] == pytest.approx(0.0)
    assert metrics.certification_gap(np.array([[[0.0, 0.0, 0.5]]]), r, np.array([0]), np.ones((1, 1), bool))[0] == \
        pytest.approx(0.4)


def test_claims_to_reach_ranked_analyst_load():
    s0 = np.random.default_rng(0).normal(size=100)
    mk = lambda obs: Incident("a", np.array([5.0, 4, 3, 2, 1]), np.array([1, 0, 1, 1, 0], np.int8),
                              trans=np.array([0, -1, 1, 2, -1]), n_trans=obs, n_trans_obs=obs)
    bank = build_loss_tables([mk(4), mk(3)], s0)
    # distinct transition hits along the ranking: [0, 1, 1, 2, 3, 3]; 80% of 4 = 3.2 unreachable; of 3 = 2.4 -> k = 4
    out = metrics.claims_to_reach(bank, [0, 1])
    assert np.isnan(out[0]) and out[1] == 4


def test_splits_random_and_loso():
    sp = splits.random_splits(20, 7, seeds=range(5))
    assert sp.cal.shape == (5, 7) and sp.test.shape == (5, 13)
    for _, c, t in sp:
        assert len(set(c) | set(t)) == 20 and not set(c) & set(t)
    sp2 = splits.random_splits(20, 7, seeds=range(5))
    np.testing.assert_array_equal(sp.cal, sp2.cal)
    lo = splits.loso_splits(["a", "a", "b", "c", "c", "c"])
    assert lo.S == 3 and lo.test_mask.sum(1).tolist() == [2, 1, 3] and lo.K.tolist() == [4, 5, 3]
    with pytest.raises(ValueError):
        splits.random_splits(5, 5)


@pytest.fixture(scope="module")
def small_bank():
    cfg = synth.SynthConfig(mu=3.5, n_median=120, chains=True, hidden_frac=0.2)
    s0 = synth.synth_null_pool(40, 7, cfg)
    return build_loss_tables(synth.synth_pool(80, 8, cfg), s0)


def test_runner_crc_matches_bruteforce(small_bank):
    bank = small_bank
    sp = splits.random_splits(bank.n, 40, seeds=range(6))
    res = runner.evaluate(bank, "crc_bh", sp, 0.2)
    for i, (_, c, t) in enumerate(sp):
        j = crc_threshold(bank.t("Lmono")[c], 0.2)
        assert res["lam_idx"][i] == j
        assert res["fdr"][i] == pytest.approx(bank.t("fdp_strict")[t, j].mean())
        tp = bank.t("TP")[t, j]
        assert res["recall_pooled"][i] == pytest.approx(tp.sum() / bank.col("n_true")[t].sum())


def test_run_splits_outputs_and_resume(small_bank, tmp_path):
    out = tmp_path / "run"
    df = runner.run_splits(small_bank, ["crc_bh", "chain_composite", "bh_iid", "by_ref", "lr_a2"], K=40,
                           seeds=range(4), alpha=0.2, out_dir=out)
    assert df.height == 20 and set(df["method"].unique()) == {"crc_bh", "chain_composite", "bh_iid", "by_ref", "lr_a2"}
    for f in ("per_split.parquet", "splits.jsonl", "summary.parquet", "run_manifest.json"):
        assert (out / f).exists()
    mtime = (out / "run_manifest.json").stat().st_mtime_ns
    df2 = runner.run_splits(small_bank, ["crc_bh", "chain_composite", "bh_iid", "by_ref", "lr_a2"], K=40,
                            seeds=range(4), alpha=0.2, out_dir=out)
    assert (out / "run_manifest.json").stat().st_mtime_ns == mtime     # resumed, not recomputed
    assert df2.equals(df)
    lines = (out / "splits.jsonl").read_text().splitlines()
    assert len(lines) == 20 and json.loads(lines[0])["method"] == "crc_bh"
    summ = pl.read_parquet(out / "summary.parquet")
    assert "fdr_mean" in summ.columns and summ.height == 5
    with pytest.raises(ValueError):
        runner.evaluate(small_bank, "b7", splits.random_splits(small_bank.n, 40, range(2)), 0.1)


def test_bootstrap_holm_validity():
    adj, rej = bootstrap.holm([0.01, 0.04, 0.03], 0.05)
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
    assert rej.tolist() == [True, False, False]
    z, p = bootstrap.validity_z(0.12, 0.01, 0.1)
    assert z == pytest.approx(2.0) and p == pytest.approx(0.02275, abs=1e-4)
    rng_vals = np.random.default_rng(0).normal(size=50)
    ci = bootstrap.cluster_bootstrap(np.arange(50), lambda rows, seeds: float(rng_vals[rows].mean()), B=200, splits=3)
    assert ci.se == pytest.approx(1 / np.sqrt(50), rel=0.35) and ci.lo < ci.estimate < ci.hi
    tab = bootstrap.validity_table({"a": (0.11, 0.01), "b": (0.09, 0.01)}, 0.1)
    assert tab[0]["max_standardised_excess"] == pytest.approx(1.0) and not tab[1]["violation"]


def test_bootstrap_method_fn_runs(small_bank):
    fn = bootstrap.method_stat_fn(small_bank, "crc_bh", K=40, alpha=0.2)
    ci = bootstrap.cluster_bootstrap(np.arange(small_bank.n), fn, B=20, splits=5)
    assert np.isfinite(ci.se) and 0 <= ci.estimate <= 1


def test_prereg_hash_and_immutability(tmp_path):
    p = tmp_path / "PREREG.json"
    h = prereg.write_prereg(p, {"alpha_grid": [0.1, 0.2], "split_seeds": np.arange(3)})
    assert prereg.verify_prereg(p, h) and len(h) == 64
    assert prereg.write_prereg(p, {"split_seeds": [0, 1, 2], "alpha_grid": [0.1, 0.2]}) == h   # canonical, no-op
    with pytest.raises(FileExistsError):
        prereg.write_prereg(p, {"alpha_grid": [0.05]})
    h2 = prereg.write_prereg(p, {"alpha_grid": [0.05]}, overwrite=True)
    assert h2 != h and prereg.verify_prereg(p, h2)
    with pytest.raises(ValueError):
        prereg.write_prereg(tmp_path / "x.json", {"alpha_grid": [0.1]}, strict=True)


def test_gates_helpers(small_bank):
    assert gates.auc([1, 2], [0, 2]) == pytest.approx(0.625)            # (1 + 0 + 1 + 1/2)/4, tie counted 1/2
    assert np.isnan(gates.auc([], [1.0]))
    w = gates.within_incident_auc([np.array([3.0, 0.0, 1.0])], [np.array([1, 0, -1])])
    assert w["mean"] == 1.0 and w["n_incidents"] == 1
    nr = gates.n_over_r([100, 50, 30], [2, 0, 3])
    assert nr["median"] == pytest.approx(30.0) and nr["pooled"] == pytest.approx(26.0) and nr["n_benign"] == 1
    st = gates.str_at_alpha(small_bank, 0.2, K=40, n_splits=5)
    assert 0 <= st["str_mean"] <= 1 and st["K"] == 40


def test_b7_b8():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.05).astype(np.int8)
    s = rng.normal(size=y.size) + 3 * y
    iso = IsotonicB7().fit(s, y)
    grid = np.linspace(-3, 6, 50)
    assert np.all(np.diff(iso.predict(grid)) >= 0)
    m = iso.claim_mask(grid)
    assert not np.any(np.diff(m.astype(int)) < 0)                         # an upper set of scores
    a, b = platt_fit(s, y)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(s[:, None], y)
    assert a == pytest.approx(lr.coef_[0, 0], rel=0.05) and b == pytest.approx(lr.intercept_[0], rel=0.05)
    pl8 = PlattB8().fit(s, y)
    sel = pl8.select_mask(s[:500], 0.1)
    if sel.any():
        assert np.mean(1 - pl8.predict(s[:500][sel])) <= 0.1 + 1e-12
        assert s[:500][sel].min() >= s[:500][~sel].max()                  # top-score set
    incs = [Incident("a", np.array([5.0, 0.0, -1.0]), np.array([1, 0, 0], np.int8))]
    bank = build_loss_tables(incs, rng.normal(size=100))
    assert b7_R(bank, iso).tolist() == [int(iso.claim_mask(np.array([5.0, 0.0, -1.0])).sum())]
