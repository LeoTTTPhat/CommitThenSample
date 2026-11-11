"""scoring/cascade.py (DESIGN §3.4.4): initial band thresholds, symmetric shrink, nesting, io."""
from __future__ import annotations

import numpy as np

from certichain.scoring import cascade as C


def toy(seed=0, n=20000):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int8)
    y[rng.random(n) < 0.01] = 1
    y[(y == 0) & (rng.random(n) < 0.3)] = -1
    s = rng.normal(0, 1, n) + 4.0 * (y == 1) + 1.0 * (y == -1)
    return s, y


def test_initial_thresholds_definitions():
    s, y = toy()
    lo, hi = C.initial_thresholds(s, y)
    pos = s[y == 1]
    assert np.mean(pos < lo) <= 0.005 + 1e-12
    # largest such threshold: the next positive score up would exceed the 0.5% budget
    k = int(np.floor(0.005 * pos.size))
    assert np.isclose(lo, np.sort(pos)[k])
    sel = s >= hi
    assert np.mean(y[sel] != 1) <= 0.01 + 1e-12
    # smallest such threshold: no lower score threshold reaches FDP <= 1%
    lower = np.sort(np.unique(s[s < hi]))[-50:]
    assert all(np.mean(y[s >= t] != 1) > 0.01 for t in lower)


def test_shrink_reaches_target_both_rules():
    s, y = toy(1)
    for rule in ("cand", "pos"):
        b = C.fit_band(s, y, kappa_target=0.10, rule=rule)
        assert b.kappa_train <= 0.10 + 1e-9
        assert b.q_lo >= b.q_lo0 - 1e-12 and b.q_hi <= b.q_hi0 + 1e-12
        assert np.isclose(np.mean(b.contains(s)), b.kappa_train)
        assert np.isclose(b.pos_in_band + b.pos_above + b.pos_below, 1.0)
    pos_rule = C.fit_band(s, y, 0.10, rule="pos")
    cand_rule = C.fit_band(s, y, 0.10, rule="cand")
    assert pos_rule.pos_in_band >= cand_rule.pos_in_band        # the pos rule keeps the band on the true links
    assert pos_rule.kappa_train > 0.09                          # continuous (interpolated) shrink hits the target


def test_no_shrink_when_already_small_and_nesting():
    s, y = toy(2)
    big = C.fit_band(s, y, kappa_target=0.99, kappa_max=0.99)
    assert np.isclose(big.q_lo, big.q_lo0) and np.isclose(big.q_hi, big.q_hi0)
    grid = C.band_grid(s, y, (0.05, 0.10, 0.20), rule="pos")
    b05, b10, b20 = grid["0.05"], grid["0.10"], grid["0.20"]
    assert b05.kappa_train <= b10.kappa_train <= b20.kappa_train
    for small, large in ((b05, b10), (b10, b20)):
        assert small.q_lo >= large.q_lo - 1e-12 and small.q_hi <= large.q_hi + 1e-12   # nested bands


def test_band_json_round_trip_with_inf(tmp_path):
    s = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0, 1, -1, 1])
    b = C.fit_band(s, y, kappa_target=0.5, kappa_max=0.5)
    b.save(tmp_path / "band.json")
    c = C.Band.load(tmp_path / "band.json")
    assert np.array_equal(b.contains(s), c.contains(s))
    assert c.q_hi == b.q_hi


# ----------------------------------------------------------------------------- driver: synthetic results tree
import json  # noqa: E402

import polars as pl  # noqa: E402
import pytest  # noqa: E402

from certichain.gt.semantics import TRANSITION_SCHEMA  # noqa: E402
from certichain.io.schema import PAIR_LABEL_SCHEMA, conform  # noqa: E402
from test_features_stats import synth_incident  # noqa: E402


def make_tree(root, n_train=3, n_dev=2, n_caltest=4, ds="cf_ton"):
    """A tiny scripts/07-shaped interim tree under ``root`` (CERTICHAIN_RESULTS) for driver tests."""
    base = root / "interim" / ds
    out = {}
    k = 0
    for role, n in (("train", n_train), ("dev", n_dev), ("caltest", n_caltest)):
        for i in range(n):
            inc = f"{ds}/{role}/{i:06d}"
            units, side, cands, y = synth_incident(500 + k)
            k += 1
            uid = units["unit_id"].to_list()
            pos = [j for j, v in enumerate(y.tolist()) if v == 1]
            cid = cands.sort("cand_id")["cand_id"].to_list()
            tr_rows = []
            if pos:
                c = cands.sort("cand_id").row(pos[0], named=True)
                tr_rows.append({"src_ep": "c0.brute", "dst_ep": "c0.session", "kind": "obs", "enab_u": c["u"],
                                "first_v": c["v"], "in_C": True, "cand_id": c["cand_id"]})
                ep = {c["u"]: "c0.brute", c["v"]: "c0.session"}
            else:
                ep = {}
            for sub, df in (("units", units), ("units_side", side), ("candidates", cands)):
                p = base / sub / f"{inc}.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(p)
            p = base / "labels_pairs" / f"{inc}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"cand_id": cid, "y": y, "bridge": [False] * len(cid)}, schema=PAIR_LABEL_SCHEMA).write_parquet(p)
            p = base / "transitions" / f"{inc}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            conform(pl.DataFrame(tr_rows) if tr_rows else pl.DataFrame(schema=TRANSITION_SCHEMA),
                    TRANSITION_SCHEMA).write_parquet(p)
            p = base / "gt_units" / f"{inc}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"unit_id": list(ep), "episode": list(ep.values()), "campaign": ["c0"] * len(ep),
                          "mixed": [False] * len(ep)},
                         schema={"unit_id": pl.Int64, "episode": pl.Utf8, "campaign": pl.Utf8,
                                 "mixed": pl.Boolean}).write_parquet(p)
            for sub, obj in (("gt", {"transitions": [["c0.brute", "c0.session"]] if ep else [],
                                     "meta": {"campaigns": {"c0": {"template": "T-CRED", "kind": "main"}}}}),
                             ("stats", {"N": len(cid), "transition_stats": {
                                 "n_T": 1 if ep else 0, "n_T_obs": 1 if ep else 0, "n_T_hid": 0,
                                 "n_T_obs_in_C": 1 if ep else 0, "n_E_link": 1 if ep else 0, "n_bridges": 0}})):
                p = base / sub / f"{inc}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(obj))
            out[inc] = (units, cands, y, uid)
    (base / "summary_07.json").write_text(json.dumps({"code_fp": ["test"], "n_incidents": len(out)}))
    return out


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    monkeypatch.setenv("CERTICHAIN_RESULTS", str(tmp_path))
    from certichain.scoring import driver as D
    D.set_candidate_filter(None)
    D._KEEP_CACHE.clear()
    return tmp_path, make_tree(tmp_path), D


def test_driver_roles_and_prereg_guard(tree):
    root, incs, D = tree
    assert D.list_incidents("cf_ton", "train") == [f"cf_ton/train/{i:06d}" for i in range(3)]
    assert D.role_of("cf_ton", "cf_ton/caltest/000001") == "caltest" and D.role_of("iot23", "34-1_b000") == "test"
    D.load_labels("cf_ton", "cf_ton/dev/000000")                         # dev labels are always available
    with pytest.raises(D.PreregGuard):
        D.load_labels("cf_ton", "cf_ton/caltest/000000")
    with pytest.raises(D.PreregGuard):
        D.load_transitions("cf_ton", "cf_ton/caltest/000000")
    (root / D.PREREG_NAME).write_text("{}\n")
    assert D.load_labels("cf_ton", "cf_ton/caltest/000000").height > 0  # unsealed after the PREREG exists


def test_candidate_filter_view_is_consistent(tree):
    root, incs, D = tree
    inc = "cf_ton/train/000000"
    full_c, full_y = D.load_cands("cf_ton", inc), D.load_labels("cf_ton", inc)
    D.set_candidate_filter("src_type")
    try:
        c, y = D.load_cands("cf_ton", inc), D.load_labels("cf_ton", inc)
        units = D.load_units("cf_ton", inc)
        keep = D.continuation_keep(units, full_c, "src_type")
        assert c.height == int(keep.sum()) and c["cand_id"].to_list() == y["cand_id"].to_list()
        tu = dict(zip(units["unit_id"].to_list(), units["type"].to_list()))
        su = dict(zip(units["unit_id"].to_list(), units["src"].to_list()))
        assert all(not (su[a] == su[b] and tu[a] == tu[b]) for a, b in zip(c["u"].to_list(), c["v"].to_list()))
        f = D.load_features("cf_ton", inc)
        assert f.height == c.height and "features__src_type" in str(D.features_path("cf_ton", inc))
        tr = D.load_transitions("cf_ton", inc)
        kept = set(c["cand_id"].to_list())
        assert all((cid is None) or (cid in kept) for cid in tr["cand_id"].to_list())
        I = D.build_incident("cf_ton", inc, np.zeros(c.height))
        assert I.scores.size == c.height and np.all(I.trans[I.trans >= 0] >= 0)
        for p, t in enumerate(I.trans):
            if t >= 0:
                assert I.y[p] == 1
    finally:
        D.set_candidate_filter(None)
    assert D.load_cands("cf_ton", inc).equals(full_c) and D.load_labels("cf_ton", inc).equals(full_y)


def test_build_incident_positions_and_strata(tree):
    root, incs, D = tree
    inc = "cf_ton/dev/000001"
    c = D.load_cands("cf_ton", inc)
    I = D.build_incident("cf_ton", inc, np.arange(c.height, dtype=float))
    I.validate()
    assert I.u.tolist() == c["u"].to_list() and I.unit_ts.size == D.load_units("cf_ton", inc).height
    assert set(I.strata) >= {"decoy_rich", "benign_only", "short_chain"}
    assert I.n_trans == D.load_stats("cf_ton", inc)["transition_stats"]["n_T"]


def test_llm_group_plans_dedup_and_freeze(tree):
    root, incs, D = tree
    import shutil
    # fake S_TON stats scores + band so the cf_ton groups can be planned (label-free)
    md = D.models_dir("S_TON")
    md.mkdir(parents=True, exist_ok=True)
    for inc in D.list_incidents("cf_ton"):
        c = D.load_cands("cf_ton", inc)
        s = np.linspace(-3, 3, c.height) if c.height else np.zeros(0)
        D.write_scores("cf_ton", "S_TON.stat", inc, pl.DataFrame({"cand_id": c["cand_id"], "s": s}))
    band = C.Band(q_lo=-1.0, q_hi=1.0, kappa_train=0.3, kappa_target=0.3, rule="pos", q_lo0=-1.0, q_hi0=1.0,
                  kappa0=0.3, pos_in_band=1.0, pos_above=0.0, pos_below=0.0, fdp_above_train=0.0, n_train=1, n_pos=1)
    band.save(md / "band.json")
    (md / "lgbm_stat.txt").write_text("fake model")
    (md / "bands.json").write_text(json.dumps({"grid": {"0.20": band.to_json()}}))
    tr_items, tr_plan = D.plan_group("cf_ton", "train_band")
    ct_items, _ = D.plan_group("cf_ton", "caltest_band")
    b5_items, b5_plan = D.plan_group("cf_ton", "b5_arm")
    assert tr_items and all(D.role_of("cf_ton", i) == "train" for i, _ in tr_items)
    assert not set(b5_items) & set(ct_items)                         # each candidate scored once
    two = D.b5_incidents("cf_ton", n=2)
    assert two == D.b5_incidents("cf_ton", n=2) and len(two) == 2     # deterministic, label-free draw
    arm = D.b5_incidents("cf_ton")                                    # pool (4) <= 15: the whole pool
    assert {i for i, _ in b5_items} <= set(arm) and len(arm) == 4
    assert b5_plan["n_dedup_earlier"] == b5_plan["n_group_candidates"] - len(b5_items)
    again, _ = D.plan_group("cf_ton", "train_band")                 # frozen list re-used
    assert again == tr_items
    (md / "lgbm_stat.txt").write_text("refit model")                # a refit invalidates frozen plans
    with pytest.raises(RuntimeError):
        D.plan_group("cf_ton", "train_band")
    shutil.rmtree(D.llm_root("cf_ton"))
    assert D.plan_group("cf_ton", "train_band")[0] == tr_items


_TINY_LLM_CODE = r"""
import sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, sys.argv[1])
from test_llm_scorer import PREFIX, ToyTokenizer, tiny_model
from certichain.llm.mlx_scorer import MLXScorer, ScorerConfig
from certichain.scoring import driver as D
mx.set_default_device(mx.cpu)
tok = ToyTokenizer()
sc = MLXScorer.from_model(tiny_model(), tok, PREFIX, ScorerConfig(require_gpu_lock=False, batch_size=4, shard_size=16))
count = lambda t: len(tok.encode(t))
kw = dict(scorer=sc, count_tokens=count, check_memory=False, log=lambda m: None)
recs = D.run_groups("local/test-model", PREFIX, [("cf_ton", "train_band")], **kw)
items, _ = D.plan_group("cf_ton", "train_band")
assert recs[0]["n_items"] == len(items) > 0
df = D.read_llm("cf_ton", sc.model_tag, sc.prompt_hash)
assert sorted(zip(df["incident"].to_list(), df["cand_id"].to_list())) == sorted(items)
lk = D.llm_lookup("cf_ton", sc.model_tag, sc.prompt_hash)
inc, cid = items[0]
assert np.isfinite(lk[inc][cid])
shards = sorted((D.llm_root("cf_ton") / "train_band" / sc.model_tag / sc.prompt_hash).glob("part-*.parquet"))
mt = [f.stat().st_mtime_ns for f in shards]
D.run_groups("local/test-model", PREFIX, [("cf_ton", "train_band")], **kw)
assert [f.stat().st_mtime_ns for f in shards] == mt          # resumed: nothing re-scored
try:
    D.run_groups("local/test-model", PREFIX, [("cf_ton", "train_band")], frozen_hash="0" * 16, **kw)
    raise SystemExit("changed prompt hash was not refused")
except RuntimeError:
    pass
print("OK", len(items))
"""


def test_run_groups_tiny_model_shards_and_lookup(tree):
    """LLM driver end to end on a tiny random MLX model (CPU): frozen plan -> cards -> resumable shards ->
    (incident, cand_id) lookup; a rerun scores nothing new; a changed prompt hash is refused. Runs in a
    subprocess: mlx_lm pulls in a second OpenMP runtime that aborts LightGBM in the same process."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    pytest.importorskip("mlx.core")
    root, incs, D = tree
    md = D.models_dir("S_TON")
    md.mkdir(parents=True, exist_ok=True)
    for inc in D.list_incidents("cf_ton"):
        c = D.load_cands("cf_ton", inc)
        D.write_scores("cf_ton", "S_TON.stat", inc,
                       pl.DataFrame({"cand_id": c["cand_id"], "s": np.linspace(-3, 3, c.height)}))
    C.Band(q_lo=-0.5, q_hi=0.5, kappa_train=0.2, kappa_target=0.2, rule="pos", q_lo0=-0.5, q_hi0=0.5, kappa0=0.2,
           pos_in_band=1.0, pos_above=0.0, pos_below=0.0, fdp_above_train=0.0, n_train=1, n_pos=1).save(md / "band.json")
    (md / "lgbm_stat.txt").write_text("fake")
    env = dict(os.environ, CERTICHAIN_RESULTS=str(root))
    r = subprocess.run([sys.executable, "-I", "-c", _TINY_LLM_CODE, str(Path(__file__).parent)], env=env,
                       capture_output=True, text=True, timeout=600)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    assert r.returncode == 0 and last.startswith("OK"), r.stdout[-2000:] + r.stderr[-4000:]
