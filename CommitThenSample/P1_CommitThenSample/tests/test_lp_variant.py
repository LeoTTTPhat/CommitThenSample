"""Recipe variant of the LP streams (rev1, review R3-M1(a)): thinning of the seed host's pure zero-payload
connections (streams/lp.thin_zero_payload, recipe.zero_variant_spec).

* the default path is unchanged (rebuilt original stream == the stored results/streams copy, every array);
* the payload keep decisions of a variant are bit-identical to the original stream;
* only the seed host's pure zero-payload connections are thinned, per connection, at rate ~ q (z0: all removed);
* the zero-payload uniforms come from an independent child stream of the seed (SeedSequence(seed, spawn_key=(0,))).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ctsample.streams import lp

RES = Path(__file__).resolve().parents[1] / "results"


# ------------------------------------------------------------------------------------------ #
# synthetic parts
# ------------------------------------------------------------------------------------------ #
def _synthetic_parts(n_zero_conn=6000, seed=0):
    """One background-like part (no thinning config) and one foreground part with a seed MAC 'aa', another
    device 'bb'; pure zero-payload connections of 'aa' (1-3 units each), payload connections of 'aa' with
    zero-payload segments (malicious and benign), zero-payload connections of 'bb'."""
    rng = np.random.default_rng(seed)
    rows = []
    fid = 0

    def add(mac, n_units, pay_pattern, mal, label):
        nonlocal fid
        fid += 1
        fu = fid.to_bytes(16, "little")
        for j in range(n_units):
            rows.append({"device_mac": mac, "pay": bool(pay_pattern[j]), "flow_uid": fu,
                         "is_malicious": bool(mal and pay_pattern[j]), "label": label, "s": 60 + j})

    for _ in range(n_zero_conn):
        add("aa", int(rng.integers(1, 4)), [0, 0, 0], False, "Malicious")
    for _ in range(40):
        add("aa", 3, [0, 1, 0], True, "Malicious")           # malicious payload connection with zero segments
    for _ in range(30):
        add("aa", 2, [0, 1], False, "Benign")                 # benign payload connection with a zero segment
    for _ in range(500):
        add("bb", 2, [0, 0], False, "Benign")                 # another device's zero-payload connections
    perm = rng.permutation(len(rows))
    rows = [rows[i] for i in perm]
    n = len(rows)
    ep = np.sort(rng.integers(0, 50, n))
    ts = ep * lp.T_US + np.arange(n)
    t = pl.DataFrame(rows).with_columns(pl.Series("first_ts_us", ts), pl.col("flow_uid").cast(pl.Binary))
    fg = {"source": "FG", "frame": t, "sepoch": ep.astype(np.int64), "delta_us": 0,
          "malpay": (t["pay"] & t["is_malicious"]).to_numpy()}
    bgf = pl.DataFrame({"device_mac": ["cc"] * 10, "pay": [False] * 10, "flow_uid": [bytes(16)] * 10,
                        "is_malicious": [False] * 10, "label": ["Benign"] * 10, "s": [60] * 10,
                        "first_ts_us": np.arange(10)}).with_columns(pl.col("flow_uid").cast(pl.Binary))
    bg = {"source": "BG", "frame": bgf, "sepoch": np.zeros(10, np.int64), "delta_us": 0, "malpay": np.zeros(10, bool)}
    return [bg, fg]


def _conn_table(t: pl.DataFrame) -> pl.DataFrame:
    return t.group_by("flow_uid").agg(pl.len().alias("n"), pl.col("pay").any().alias("any_pay"),
                                      pl.col("device_mac").first().alias("mac"))


def test_zero_thinning_synthetic_rate_and_scope():
    parts = _synthetic_parts()
    before = parts[1]["frame"]
    zt = {"variant": "zq", "keep": "q", "sources": {"FG": ["aa"]}}
    q = 0.3
    rep = lp.thin_zero_payload(parts, zt, q, [7, 1000])
    after = parts[1]["frame"]
    assert parts[0]["frame"].height == 10                                   # unconfigured part untouched
    assert len(parts[1]["sepoch"]) == after.height == len(parts[1]["malpay"])
    cb, ca = _conn_table(before), _conn_table(after)
    # payload connections (with their zero segments) and the other device are untouched
    keep_b = cb.filter(pl.col("any_pay") | (pl.col("mac") != "aa")).sort("flow_uid")
    keep_a = ca.filter(pl.col("any_pay") | (pl.col("mac") != "aa")).sort("flow_uid")
    assert keep_b.equals(keep_a)
    # pure zero-payload connections of the seed host: all-or-nothing per connection, rate ~ q
    zb = cb.filter(~pl.col("any_pay") & (pl.col("mac") == "aa"))
    za = ca.filter(~pl.col("any_pay") & (pl.col("mac") == "aa"))
    j = zb.join(za, on="flow_uid", how="left", suffix="_a")
    kept = j["n_a"].is_not_null().to_numpy()
    assert np.all(j.filter(pl.col("n_a").is_not_null()).select((pl.col("n") == pl.col("n_a")).all()).item())
    n = len(kept)
    sd = math.sqrt(n * q * (1 - q))
    assert abs(kept.sum() - q * n) <= 4.5 * sd
    assert rep["n_pure_zero_conns"] == n and rep["n_pure_zero_conns_kept"] == int(kept.sum())
    p = rep["parts"][0]
    assert p["n_pure_zero_units_before"] == int(zb["n"].sum())
    assert p["n_pure_zero_units_after"] == int(za["n"].sum())
    assert p["n_zero_dropped_units"] == before.height - after.height
    assert p["n_zero_in_mal_payload_conns"] == 80 and p["n_zero_in_other_payload_conns"] == 30
    # deterministic
    parts2 = _synthetic_parts()
    lp.thin_zero_payload(parts2, zt, q, [7, 1000])
    assert parts2[1]["frame"].equals(after)


def test_zero_thinning_limits_and_independent_stream():
    for keep, q, expect_all_kept in (("q", 1.0, True), (0.0, 0.4, False)):
        parts = _synthetic_parts(n_zero_conn=500)
        before = parts[1]["frame"]
        rep = lp.thin_zero_payload(parts, {"variant": "x", "keep": keep, "sources": {"FG": ["aa"]}}, q, [3, 10000])
        after = parts[1]["frame"]
        if expect_all_kept:
            assert after.equals(before) and rep["n_units_dropped"] == 0
        else:                                                   # z0: no pure zero-payload connection of 'aa' left
            ca = _conn_table(after)
            assert ca.filter(~pl.col("any_pay") & (pl.col("mac") == "aa")).height == 0
            assert rep["n_pure_zero_conns_kept"] == 0 and rep["keep_prob"] == 0.0
    # the zero-payload draws are an independent child stream; the payload thinning stream is unchanged
    u_pay = lp.thin_connections(np.arange(100), 1000, 99_000, 0.001, [3, 10000])["u"]
    u_ref = np.random.Generator(np.random.PCG64(np.random.SeedSequence([3, 10000]))).random(100)
    u_zero = np.random.Generator(np.random.PCG64(np.random.SeedSequence([3, 10000], spawn_key=lp.ZERO_SPAWN_KEY))).random(100)
    u_spawn = np.random.Generator(np.random.PCG64(np.random.SeedSequence([3, 10000]).spawn(1)[0])).random(100)
    assert np.array_equal(u_pay, u_ref) and np.array_equal(u_zero, u_spawn) and not np.allclose(u_zero, u_pay)


# ------------------------------------------------------------------------------------------ #
# real data (34-1 at p = 0.1 %: q = 0.104, so both thinning rules act)
# ------------------------------------------------------------------------------------------ #
def _have_data():
    need = [RES / "units" / "ciciot2023_benign" / "BenignTraffic1" / "DONE",
            RES / "units" / "iot23" / "CTU-IoT-Malware-Capture-34-1" / "DONE",
            RES / "labels" / "iot23" / "CTU-IoT-Malware-Capture-34-1.parquet", RES / "inventory.json"]
    return all(p.exists() for p in need)


@pytest.fixture(scope="module")
def built3(tmp_path_factory):
    if not _have_data():
        pytest.skip("S1/S3 outputs not present")
    from ctsample.streams import recipe as R
    cfg, inv = R.load_config(), R.load_inventory()
    base = R.lp_iot23_spec(cfg, inv, "CTU-IoT-Malware-Capture-34-1", 0.001)
    out = tmp_path_factory.mktemp("streams_variant")
    specs = [base, R.zero_variant_spec(base, "zq", "q"), R.zero_variant_spec(base, "z0", 0.0)]
    for s in specs:
        lp.build_stream(s, out_root=out, force=True, log=lambda m: None)
    return specs, [out / s.stream_id for s in specs]


def _arrays(d: Path) -> dict:
    out = {p.name: np.load(p) for p in sorted(d.glob("*.npy")) if not p.name.startswith("util_")}
    z = np.load(d / "cols_pay.npz")
    out.update({f"cols_pay[{k}]": z[k] for k in z.files})
    return out


def test_default_path_bit_identical_to_stored_stream(built3):
    specs, dirs = built3
    stored = RES / "streams" / specs[0].stream_id
    if not (stored / "DONE").exists():
        pytest.skip("stored original stream not present")
    a, b = _arrays(dirs[0]), _arrays(stored)
    assert set(a) == set(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k
    assert pl.read_parquet(dirs[0] / "units.parquet").equals(pl.read_parquet(stored / "units.parquet"))
    m = json.loads((dirs[0] / "meta.json").read_text())
    assert "zero_thin" not in m["extra"] and "zero_thin" not in m["extra"]["recipe"]["extra"]
    assert m["stream_id"] == json.loads((stored / "meta.json").read_text())["stream_id"]


def test_variant_keeps_payload_decisions_and_thins_seed_zero_connections(built3):
    from ctsample.sim.tables import StreamTables
    specs, dirs = built3
    base = specs[0]
    seed_mac_new = base.foregrounds[0].mac_map[base.extra["window"]["seed_mac"]]
    U = [pl.read_parquet(d / "units.parquet") for d in dirs]
    T = [StreamTables.load(d) for d in dirs]
    M = [json.loads((d / "meta.json").read_text()) for d in dirs]
    q = M[0]["extra"]["q"]
    assert 0 < q < 1
    pay0 = U[0].filter(pl.col("pay"))
    pidx0 = np.nonzero(np.asarray(T[0].pay))[0]
    for v in (1, 2):
        ex = M[v]["extra"]
        assert ex["q"] == q and M[v]["stream_id"] == M[0]["stream_id"]          # same payload q, same integer id
        for k in ("n_mal_payload_units", "n_payload_units", "n_mal_connections_kept", "realised_prevalence",
                  "n_incidents", "sum_s_pay"):
            assert ex[k] == M[0]["extra"][k], k
        # payload units: same units in the same order with the same costs, estimand values and labels
        payv = U[v].filter(pl.col("pay"))
        assert payv["unit_gid"].equals(pay0["unit_gid"]) and payv["source"].equals(pay0["source"])
        pidx = np.nonzero(np.asarray(T[v].pay))[0]
        for name in ("s", "label", "content", "incident", "epoch"):
            assert np.array_equal(np.asarray(getattr(T[v], name))[pidx], np.asarray(getattr(T[0], name))[pidx0]), name
        for qn in T[0].query_names:
            assert np.array_equal(T[v].f_dense(qn)[pidx], T[0].f_dense(qn)[pidx0]), qn
            assert np.array_equal(np.asarray(T[v].b_col(qn))[pidx], np.asarray(T[0].b_col(qn))[pidx0]), qn
        # every unit not of the seed host is kept, in the same order
        nseed0 = U[0].filter(pl.col("device_mac") != seed_mac_new)
        nseedv = U[v].filter(pl.col("device_mac") != seed_mac_new)
        assert nseedv["unit_gid"].equals(nseed0["unit_gid"])
        # payload features of the other devices are unchanged (rolling statistics are per device)
        X0, Xv = np.load(dirs[0] / "features_pay.npy"), np.load(dirs[v] / "features_pay.npy")
        oth = (pay0["device_mac"] != seed_mac_new).to_numpy()
        assert np.array_equal(X0[oth], Xv[oth])
        # seed host's pure zero-payload connections: per connection, rate ~ q (zq) / none left (z0)
        s0 = U[0].filter(pl.col("device_mac") == seed_mac_new)
        sv = U[v].filter(pl.col("device_mac") == seed_mac_new)
        c0 = s0.group_by("flow_uid_orig").agg(pl.len().alias("n"), pl.col("pay").any().alias("any_pay"))
        cv = sv.group_by("flow_uid_orig").agg(pl.len().alias("n_v"))
        j = c0.join(cv, on="flow_uid_orig", how="left")
        jp = j.filter(pl.col("any_pay"))
        assert jp["n_v"].is_not_null().all() and (jp["n"] == jp["n_v"]).all()        # payload connections intact
        jz = j.filter(~pl.col("any_pay"))
        kept = jz["n_v"].is_not_null().to_numpy()
        assert (jz.filter(pl.col("n_v").is_not_null())["n"] == jz.filter(pl.col("n_v").is_not_null())["n_v"]).all()
        zrep = ex["zero_thin"]
        assert zrep["n_pure_zero_conns"] == len(kept) and zrep["n_pure_zero_conns_kept"] == int(kept.sum())
        if v == 1:
            sd = math.sqrt(len(kept) * q * (1 - q))
            assert len(kept) > 500 and abs(kept.sum() - q * len(kept)) <= 4.5 * sd
            assert zrep["keep_prob"] == q
        else:
            assert kept.sum() == 0 and zrep["keep_prob"] == 0.0
        # the seed host's rolling unit count can only drop
        r_units = X0[:, 33], Xv[:, 33]
        assert np.all(r_units[1] <= r_units[0]) and np.any(r_units[1][~oth] < r_units[0][~oth])


# ------------------------------------------------------------------------------------------ #
# script options: the default paths are unchanged, variants are routed to their own roots
# ------------------------------------------------------------------------------------------ #
def _load_script(name: str):
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location("sv_" + name.replace(".py", "").replace("-", "_"), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_script_options_default_unchanged_and_variants_isolated():
    e3 = _load_script("12_e3.py")
    a = e3.load_cfg()
    jobs = e3.build_jobs(a)
    assert a.get("out_dir", "E3") == "E3" and [g for g in dict.fromkeys(j["group"] for j in jobs)] == ["in_domain", "zero_shot"]
    assert {j["stream"] for j in jobs} == set(a["streams_e3"]["in_domain"]) | set(a["streams_e3"]["zero_shot"])
    for v in ("zq", "z0"):
        b = e3.load_cfg(e3.P1 / "config" / "experiments" / f"E3_rev1_{v}.yaml")
        assert e3._settings(a) == e3._settings(b) and b["out_dir"] == f"E3_rev1_{v}"
        jb = e3.build_jobs(b)
        assert all(j["stream"].endswith(f"_{v}") for j in jb if j["group"] == "in_domain")
        assert all(not j["stream"].endswith(("_zq", "_z0")) for j in jb if j["group"] == "in_domain_orig")
    from ctsample.streams import recipe as R
    cfg, inv = R.load_config(), R.load_inventory()
    d5 = _load_script("05_designs.py")
    assert d5.MODELS == RES / "models" and not any(s.endswith(("_zq", "_z0")) for s, _, _ in d5.test_streams(cfg, inv))
    vt = d5.variant_test_streams(cfg, inv, "zq")
    assert len(vt) == 6 and all(s.endswith("_p10000_zq") and pk == "p10000" for s, _, pk in vt)
    d5._set_variant("zq")
    assert d5.MODELS == RES / "models_zq"
    for v, keep in (("zq", 1.0), ("z0", 0.0)):
        m4 = _load_script("04_train_utility.py")
        assert m4.MODELS == RES / "models" and m4.VARIANT is None and m4.TRAIN_ZERO_KEEP == 1.0
        m4._set_variant(v)
        assert m4.MODELS == RES / f"models_{v}" and m4.CACHE == RES / f"models_{v}" / "_cache"
        assert m4.PREVALENCES == [0.01] and m4.TRAIN_ZERO_KEEP == keep
    assert R.training_stream_ids(cfg, inv, "CTU-IoT-Malware-Capture-7-1", 0.01) == \
        [x[:-3] for x in R.training_stream_ids(cfg, inv, "CTU-IoT-Malware-Capture-7-1", 0.01, "zq")]
