"""Monte-Carlo checks of the theory (DESIGN §4, §7.6) on the synthetic generator.

Fast tests use INDEPENDENT replicates (disjoint blocks of freshly generated incidents), so the
3-SE slack is an honest one. ``test_power_grid`` (the reviewer's recall-vs-AUC table, §4.9) is
@slow and runs only when selected explicitly with ``-m slow``; it prints its table.
"""
import math
import time

import numpy as np
import pytest

from certichain.baselines.lee_ren import lee_ren_evalues
from certichain.certify import elond
from certichain.certify.crc import crc_threshold_from_sums
from certichain.certify.evalues import ebh
from certichain.certify.losses import build_loss_tables
from certichain.certify.ltt import ltt_threshold_from_sums
from certichain.certify.pvalues import sort_null_pool
from certichain.eval import gates, runner, splits, synth

# heavy-tailed incident shifts (t_2.5), heavy-tailed nulls (t_4), decoy clusters, 0.3% hallucination tail
HARSH = synth.SynthConfig(mu=3.0, tau=0.8, shift_df=2.5, null_df=4.0, decoy_rate=1.5, decoy_shift=2.0,
                          halluc_frac=0.003, n_median=120, n_sigma=0.6)
N_REP, K, M = 200, 100, 50


def _se(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / math.sqrt(x.size)


@pytest.fixture(scope="module")
def harsh():
    s0 = synth.synth_null_pool(100, 101, HARSH)
    bank = build_loss_tables(synth.synth_pool(N_REP * (K + M), 202, HARSH), s0)
    return bank


def _blocks(a):
    return a.reshape(N_REP, K + M, *a.shape[1:])


def _crc_block_metric(bank, loss, alpha, metric):
    L = _blocks(bank.t(loss))
    lam = crc_threshold_from_sums(L[:, :K].sum(1), K, alpha)
    F = _blocks(bank.t("fdp_strict"))[:, K:, :]
    f = F[np.arange(N_REP), :, lam]                        # [N_REP, M]
    return metric(f), lam


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_crc_bh_fdr_valid_under_dependence_and_heavy_tails(harsh, alpha):
    fdr, lam = _crc_block_metric(harsh, "Lmono", alpha, lambda f: f.mean(1))
    assert np.mean(lam > 0) > 0.9                          # non-trivial: CRC certifies in most replicates
    assert fdr.mean() <= alpha + 3 * _se(fdr), (fdr.mean(), _se(fdr))


@pytest.mark.parametrize("gamma", [0.2, 0.3])
def test_crc_fdx_exceedance_valid(harsh, gamma):
    alpha = 0.1
    exc, _ = _crc_block_metric(harsh, f"fdx{int(gamma * 100):03d}", alpha, lambda f: (f > gamma).mean(1))
    assert exc.mean() <= alpha + 3 * _se(exc), (exc.mean(), _se(exc))


def test_ltt_coverage(harsh):
    alpha, delta = 0.1, 0.1
    F = _blocks(harsh.t("fdp_strict"))
    risk_true = F[:, K:, :].reshape(-1, F.shape[2]).mean(0)          # 10 000 independent test incidents
    lam = ltt_threshold_from_sums(F[:, :K].sum(1), K, alpha, delta)
    cov = risk_true[lam] <= alpha
    c = cov.mean()
    assert c >= 1 - delta - 3 * math.sqrt(max(c * (1 - c), 1e-4) / N_REP), c
    assert np.mean(lam > 0) > 0.5


def test_lemma_a_evalue_mean_at_most_one(harsh):
    # (i) exact Bernoulli case: E[E] = 1 - (1-p)^(K+1) <= 1
    rng = np.random.default_rng(0)
    for p, Kb in ((0.05, 20), (0.3, 150)):
        Q = (rng.random((40000, Kb + 1)) < p).astype(float)
        E = (Kb + 1) * Q[:, -1] / (Q[:, :-1].sum(1) + 1)
        exact = 1 - (1 - p) ** (Kb + 1)
        assert abs(E.mean() - exact) <= 4 * _se(E) and exact <= 1
    # (ii) i.i.d. incident FDPs of fixed BH rules from the harsh generator (heavy-tailed, mostly-zero Q).
    # With Q_{K+1} independent of S = sum_{k<=K} Q_k, E[E] = (K+1) E[Q] E[1/(S+1)] exactly; estimating the
    # two factors separately (S from disjoint K-blocks) is far less noisy than averaging E itself.
    F = harsh.t("fdp_strict")
    for col in (30, 45, 55):
        Q = F[:, col]
        nb = Q.size // K
        inv = 1.0 / (Q[: nb * K].reshape(nb, K).sum(1) + 1.0)
        est = (K + 1) * Q.mean() * inv.mean()
        se = (K + 1) * math.sqrt((Q.mean() * inv.std(ddof=1)) ** 2 / nb + (inv.mean() * Q.std(ddof=1)) ** 2 / Q.size)
        assert est <= 1 + 3 * se, (col, est, se)


def test_elond_sup_fdp_at_most_alpha():
    cfg = synth.SynthConfig(mu=4.0, tau=0.5, shift_df=3.0, n_median=150, decoy_rate=1.0)
    s0 = synth.synth_null_pool(100, 31, cfg)
    train = elond.build_stream_bank(synth.synth_stream_pool(200, 32, cfg), s0)
    theta = elond.choose_look_levels(train, 0.1)
    n_blk, Kc, Mt = 30, 150, 100
    bank = elond.build_stream_bank(synth.synth_stream_pool(n_blk * (Kc + Mt), 33, cfg), s0)
    res = {}
    for b in range(n_blk):
        rows = np.arange(b * (Kc + Mt), (b + 1) * (Kc + Mt))
        out = elond.evaluate_stream_split(bank, rows[:Kc], rows[Kc:], 0.1, theta, order_seed=b)
        for m in ("elond", "elond_pooled", "crc_looks", "crc_naive_repeated"):
            res.setdefault(m, []).append(out[m]["sup_fdp"])
        res.setdefault("certified_looks", []).append(out["elond"]["n_looks_certified"])
    for m in ("elond", "elond_pooled", "crc_looks"):
        assert np.mean(res[m]) <= 0.1 + 3 * _se(res[m]), (m, np.mean(res[m]))
    assert np.mean(res["certified_looks"]) > 0.1          # e-LOND does certify (the check is not vacuous)


def test_example1_naive_composition_inflates_maxchain_fdr():
    L, n, alpha, Kc = 5, 20, 0.1, 150
    f = synth.example1_f(L, n, alpha, Kc)
    assert f == 10
    a1 = alpha - (1 - alpha) / Kc
    assert min(1, alpha * L / (1 - alpha)) == pytest.approx(0.5556, abs=1e-4)         # asymptote
    s0 = sort_null_pool(np.random.default_rng(0).standard_normal(10000))
    incs = [synth.example1_incident(s0, L=L, n=n, f=f, seed=0, incident_id=f"e{i}") for i in range(Kc + 50)]
    bank = build_loss_tables(incs, s0)
    df = runner.run_splits(bank, ["chain_naive", "chain_composite", "chain_rt"], K=Kc, seeds=range(5), alpha=alpha)
    naive = df.filter(df["method"] == "chain_naive")
    comp = df.filter(df["method"] == "chain_composite")
    assert np.allclose(naive["fdr"].to_numpy(), f / (n * L + f)) and f / (n * L + f) <= a1      # edge FDR <= alpha
    assert np.allclose(naive["maxchain_fdr"].to_numpy(), 0.5)                                 # = f/n
    assert (comp["maxchain_fdr"].to_numpy() <= alpha).all() and (comp["fdr"].to_numpy() <= alpha).all()
    assert (df.filter(df["method"] == "chain_rt")["chain_fwer"].to_numpy() <= alpha).all()


def test_prop2_floor_unboosted_hierarchical_ebh():
    # (a) deterministic floor: any e-BH rejection set with e <= K+1 has R >= m / (alpha (K+1))
    rng = np.random.default_rng(1)
    for _ in range(300):
        Kb, m, a = int(rng.integers(5, 200)), int(rng.integers(10, 3000)), float(rng.choice([0.05, 0.1, 0.2]))
        e = np.where(rng.random(m) < rng.uniform(0, 0.6), Kb + 1.0, rng.uniform(0, Kb + 1, m) * (rng.random(m) < 0.5))
        R = int(ebh(e, a).sum())
        assert R == 0 or R >= m / (a * (Kb + 1)) - 1e-9
    # (b) Lee-Ren Eq. 16 unboosted on synthetic splits with m / sum r ~ 100, K = 150, alpha = 0.1
    cfg = synth.SynthConfig(mu=3.0, n_median=400, n_sigma=0.0, tau=0.5, shift_df=float("inf"))
    s0 = synth.synth_null_pool(60, 41, cfg)
    bank = build_loss_tables(synth.synth_pool(200, 42, cfg), s0)
    alpha, Kc = 0.1, 150
    sp = splits.random_splits(bank.n, Kc, range(60), n_test=50)
    hits, bounds, rec = [], [], []
    for _, c, t in sp:
        cal = [(-bank.scores(k), bank.labels(k) != 1) for k in c]
        test = [-bank.scores(j) for j in t]
        d = lee_ren_evalues(cal, test, "A2", alpha)
        e = np.concatenate(d["e"])
        assert e.max() <= Kc + 1 + 1e-9
        rej = ebh(e, alpha)
        m = e.size
        r_tot = sum(int((bank.labels(j) == 1).sum()) for j in t)
        R = int(rej.sum())
        assert R == 0 or R >= m / (alpha * (Kc + 1)) - 1e-9
        assert alpha * (Kc + 1) * r_tot < m
        bounds.append(alpha / (1 - alpha * (Kc + 1) * r_tot / m))
        hits.append(R > 0)
        y = np.concatenate([bank.labels(j) for j in t])
        rec.append(np.sum(rej & (y == 1)) / max(r_tot, 1))
    p_hat = np.mean(hits)
    assert p_hat <= np.mean(bounds) + 3 * math.sqrt(max(p_hat * (1 - p_hat), 1e-4) / len(hits))
    assert np.mean(rec) <= np.mean(bounds)
    assert 0.10 < np.mean(bounds) < 0.13                  # ~0.118 at m / sum r ~ 100 (DESIGN §4.4 example)


# --------------------------------------------------------------------------- slow: reviewer's grid
@pytest.fixture
def slow_only(request):
    if "slow" not in (request.config.getoption("-m") or ""):
        pytest.skip("slow: run with -m slow")


GRID = [("mu=4.5 tau=0.5", dict(mu=4.5, tau=0.5)), ("mu=3 tau=0.5", dict(mu=3.0, tau=0.5)),
        ("mu=3 tau=1.0", dict(mu=3.0, tau=1.0)), ("mu=2 tau=0.5", dict(mu=2.0, tau=0.5)),
        ("mu=3 tau=0.5 +0.3% halluc", dict(mu=3.0, tau=0.5, halluc_frac=0.003))]
METHODS = ["crc_bh", "crc_bh_raw", "bh_ref", "bh_iid", "ltt", "crc_fdx020"]
N_POOLS, SPLITS_PER_POOL, POOL, KG, ALPHA = 10, 20, 300, 150, 0.1


def _grid_row(cfg: synth.SynthConfig, seed: int) -> dict:
    per_pool = {m: {} for m in METHODS}
    aucs, nr = [], []
    for p in range(N_POOLS):
        s0 = synth.synth_null_pool(200, 10_000 * seed + 2 * p, cfg)
        pool = synth.synth_pool(POOL, 10_000 * seed + 2 * p + 1, cfg)
        bank = build_loss_tables(pool, s0)
        aucs.append(gates.bank_within_auc(bank)["mean"])
        nr.append(gates.n_over_r(bank.col("N"), bank.col("n_true"))["pooled"])
        df = runner.run_splits(bank, METHODS, K=KG, seeds=range(SPLITS_PER_POOL), alpha=ALPHA, salt=p)
        for m in METHODS:
            sub = df.filter(df["method"] == m)
            for c in ("fdr", "recall_pooled", "recall_mean", "pooled_fdr", "fdx_020", "abstain"):
                v = sub[c].to_numpy().astype(float) if c in sub.columns else np.array([np.nan])
                if np.isfinite(v).any():                      # e.g. 'abstain' exists only for CRC-type methods
                    per_pool[m].setdefault(c, []).append(float(np.nanmean(v)))
    out = {"auc": float(np.mean(aucs)), "N_over_r": float(np.mean(nr))}
    for m in METHODS:
        for c, v in per_pool[m].items():
            out[f"{m}.{c}"] = float(np.mean(v))
            out[f"{m}.{c}.se"] = float(_se(v))
    return out


@pytest.mark.slow
def test_power_grid(slow_only):
    t0 = time.time()
    rows = {}
    for variant, base in (("t3-shift", synth.SynthConfig()), ("normal-shift", synth.SynthConfig(shift_df=math.inf))):
        for i, (name, kw) in enumerate(GRID):
            rows[(variant, name)] = _grid_row(base.with_(**kw), seed=1 + i + (100 if variant == "normal-shift" else 0))
    print(f"\n# power grid: K={KG}, alpha={ALPHA}, {N_POOLS} pools x {SPLITS_PER_POOL} splits (pool {POOL}), "
          f"SE = between-pool SD/sqrt({N_POOLS}); {time.time() - t0:.0f}s")
    hdr = ("variant", "setting", "AUC", "N/r", "CRC FDR", "+-SE", "CRC recall(pool)", "recall(mean)",
           "raw-mono FDR gap", "raw-mono recall gap", "BH-ref FDR", "BH-ref recall", "BH-iid FDR", "BH-iid recall",
           "LTT FDR", "LTT recall", "FDX(0.2) exc", "abstain")
    print(" | ".join(hdr))
    for (variant, name), r in rows.items():
        vals = [variant, name, f"{r['auc']:.4f}", f"{r['N_over_r']:.0f}", f"{r['crc_bh.fdr']:.3f}",
                f"{r['crc_bh.fdr.se']:.3f}", f"{r['crc_bh.recall_pooled']:.3f}", f"{r['crc_bh.recall_mean']:.3f}",
                f"{r['crc_bh_raw.fdr'] - r['crc_bh.fdr']:+.4f}",
                f"{r['crc_bh_raw.recall_pooled'] - r['crc_bh.recall_pooled']:+.4f}",
                f"{r['bh_ref.fdr']:.3f}", f"{r['bh_ref.recall_pooled']:.3f}", f"{r['bh_iid.fdr']:.3f}",
                f"{r['bh_iid.recall_pooled']:.3f}", f"{r['ltt.fdr']:.3f}", f"{r['ltt.recall_pooled']:.3f}",
                f"{r['crc_fdx020.fdx_020']:.3f}", f"{r['crc_bh.abstain']:.3f}"]
        print(" | ".join(vals))
    # few-shot curve at mu = 3 (both variants)
    print("\n# few-shot (mu=3, tau=0.5): K -> CRC-BH FDR / recall(pooled) / abstain, mean over pools")
    for variant, base in (("t3-shift", synth.SynthConfig()), ("normal-shift", synth.SynthConfig(shift_df=math.inf))):
        cfg = base.with_(mu=3.0, tau=0.5)
        acc = {}
        for p in range(N_POOLS):
            s0 = synth.synth_null_pool(200, 777_000 + 2 * p, cfg)
            bank = build_loss_tables(synth.synth_pool(POOL, 777_001 + 2 * p, cfg), s0)
            for Kf in (10, 20, 50, 150):
                df = runner.run_splits(bank, ["crc_bh"], K=Kf, seeds=range(SPLITS_PER_POOL), alpha=ALPHA, salt=p,
                                       n_test=POOL - 150)
                acc.setdefault(Kf, []).append((float(df["fdr"].mean()), float(df["recall_pooled"].mean()),
                                               float(df["abstain"].mean())))
        print(variant, {Kf: tuple(round(float(x), 3) for x in np.mean(v, axis=0)) for Kf, v in acc.items()})
    for r in rows.values():
        assert r["crc_bh.fdr"] <= ALPHA + 3 * r["crc_bh.fdr.se"] + 1e-9
