"""ChainForge end to end on a synthetic mini library (fast; no real data needed).

Checks the incident format contract read by gt/chainforge_episodes.py, label quarantine,
rec_id permutation, source-level disjointness, structure preservation, and determinism from
the seed. A last test runs one real CF-TON incident when the real library exists.
"""
from __future__ import annotations

import json
import os

import numpy as np
import polars as pl
import pytest

from certichain.io.labels import label_like_columns
from certichain.io.schema import RECORD_SCHEMA, conform

TYPES = ("scanning", "password", "injection", "xss", "backdoor", "ransomware", "dos", "ddos")
POOL_CODE = {"train": 1, "caltest": 2}


def _rows_for(etype, att, vic, others, t0, code, k0):
    """Synthetic TON-like rows of one episode (attacker att, primary victim vic)."""
    rows = []
    k = k0

    def add(ts, src, dst, dport, proto="tcp", state="SF", ob=100, rb=200, op=3, rp=3, svc=""):
        nonlocal k
        rows.append({"rec_id": code * 10**6 + k, "ts": float(ts), "te": float(ts) + 0.3, "ts_res": 1.0, "src": src,
                     "sport": 40000 + k % 20000, "dst": dst, "dport": dport, "proto": proto, "service": svc,
                     "obytes": ob, "rbytes": rb, "opkts": op, "rpkts": rp, "conn_state": state, "history": "",
                     "bulk": False, "n_merged": 1, "src_file": "data/ton_iot/P/Network_dataset_1.csv",
                     "src_sha256": "ab" * 32, "src_locator": f"lines:{k}-{k}"})
        k += 1

    if etype == "scanning":
        for i in range(60):
            add(t0 + i * 5, att, [vic, *others][i % (1 + len(others))], 1000 + i, state="S0", ob=0, rb=0)
    elif etype in ("password", "injection", "xss"):
        for i in range(40):
            add(t0 + i * 10, att, vic if i % 4 else others[0], 80, svc="http")
    elif etype == "backdoor":
        for i in range(30):
            add(t0 + i * 15, vic, att, 8080, state="REJ", ob=0, rb=0)
    elif etype == "ransomware":
        for i in range(30):
            add(t0 + i * 15, att, vic, 445)
            add(t0 + i * 15 + 1, vic, att, 4444, state="OTH", ob=0, rb=0)
    else:  # dos / ddos
        for i in range(80):
            add(t0 + i * 3, att, vic if i % 3 else others[-1], 80, svc="http")
        add(t0 + 250, att, "52.59.177.21", 443)            # external target (ext→ext once A is external)
    return rows, k


def _build_fixture(root):
    from certichain.chainforge.background import build_background_library
    from certichain.chainforge.episodes import ton_role_table
    from certichain.io.paths import atomic_write_json, atomic_write_parquet

    lib = root / "interim" / "ton" / "library"
    lib.mkdir(parents=True)
    eps, recs, labs = [], [], []
    for pool, code in POOL_CODE.items():
        k = 0
        for ti, etype in enumerate(TYPES):
            for j in range(3):
                att = f"192.168.1.{30 + (ti + j) % 10}"
                vic = ["192.168.1.193", "192.168.1.195", "192.168.1.190"][j]
                others = ["192.168.1.152", "192.168.1.1", "192.168.1.49"]
                t0 = 1_556_000_000 + code * 10**6 + (ti * 3 + j) * 3600
                rows, k = _rows_for(etype, att, vic, others, t0, code, k)
                eid = f"{etype}-{att.rsplit('.', 1)[1]}-b{j:03d}-s000-{pool}"
                eps.append({"episode_id": eid, "type": etype, "attacker": att, "pool": pool,
                            "files": ["data/ton_iot/P/Network_dataset_1.csv"]})
                for r in rows:
                    recs.append({**r, "lib_eid": eid})
                    labs.append({"rec_id": r["rec_id"], "label": "1", "detailed_label": etype, "lib_eid": eid})
    ep = pl.DataFrame(eps)
    rec = conform(pl.DataFrame(recs), {**RECORD_SCHEMA, "lib_eid": pl.Utf8})
    table = ton_role_table(ep, rec)
    atomic_write_parquet(table, lib / "episodes.parquet")
    lab = pl.DataFrame(labs)
    for pool in POOL_CODE:
        ids = table.filter(pl.col("pool") == pool)["lib_eid"].implode()
        atomic_write_parquet(rec.filter(pl.col("lib_eid").is_in(ids)), lib / f"records_{pool}.parquet")
        atomic_write_parquet(lab.filter(pl.col("lib_eid").is_in(ids)), lib / f"labels_{pool}.parquet")
    atomic_write_json(lib / "library.json", {"dataset": "ton", "n_episodes": table.height})
    # backgrounds: 7 devices over 5 h; device .6 beacons every 60 s to an external NTP-like host
    pools = {}
    for pool, code in POOL_CODE.items():
        rows, k = [], 0
        t_base = 1_554_000_000 + code * 10**6
        devs = ["192.168.1.6", "192.168.1.133", "192.168.1.152", "192.168.1.190", "192.168.1.193", "192.168.1.195",
                "192.168.1.250"]
        rng = np.random.default_rng(code)
        for t in range(0, 5 * 3600, 20):
            d = devs[int(rng.integers(len(devs)))]
            peer = ["192.168.1.190", "8.8.8.8", "224.0.0.251", "192.168.1.255"][int(rng.integers(4))]
            rows.append({"rec_id": (100 + code) * 10**6 + k, "ts": float(t_base + t), "te": float(t_base + t) + 0.5,
                         "ts_res": 1.0, "src": d, "sport": 50000 + k % 1000, "dst": peer if peer != d else "8.8.8.8",
                         "dport": 53, "proto": "udp", "service": "dns", "obytes": 40, "rbytes": 80, "opkts": 1,
                         "rpkts": 1, "conn_state": "SF", "history": "", "bulk": False, "n_merged": 1,
                         "src_file": "data/ton_iot/R/normal_IoT/conn.log", "src_sha256": "cd" * 32,
                         "src_locator": f"lines:{k}-{k}"})
            k += 1
        for t in [b + i * 60 for b in range(0, 5 * 3600, 1800) for i in range(10)]:   # 10-min beacon bursts
            rows.append({"rec_id": (100 + code) * 10**6 + k, "ts": float(t_base + t), "te": float(t_base + t) + 1.0,
                         "ts_res": 1.0, "src": "192.168.1.6", "sport": 123, "dst": "91.189.89.199", "dport": 123,
                         "proto": "udp", "service": "", "obytes": 48, "rbytes": 48, "opkts": 1, "rpkts": 1,
                         "conn_state": "SF", "history": "", "bulk": False, "n_merged": 1,
                         "src_file": "data/ton_iot/R/normal_IoT/conn.log", "src_sha256": "cd" * 32,
                         "src_locator": f"lines:{k}-{k}"})
            k += 1
        bg = conform(pl.DataFrame(rows), RECORD_SCHEMA).sort("ts")
        path = root / "interim" / "ton" / "backgrounds" / f"{pool}.parquet"
        atomic_write_parquet(bg, path)
        pools[pool] = {"source": f"synthetic_{pool}", "path": path}
    build_background_library("ton", pools, force=True)


@pytest.fixture(scope="module")
def cf_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("cfres")
    old = os.environ.get("CERTICHAIN_RESULTS")
    os.environ["CERTICHAIN_RESULTS"] = str(root)
    try:
        _build_fixture(root)
        from certichain.chainforge.background import BackgroundLibrary
        from certichain.chainforge.episodes import EpisodeLibrary
        yield root, EpisodeLibrary.load("ton"), BackgroundLibrary.load("ton")
    finally:
        if old is None:
            os.environ.pop("CERTICHAIN_RESULTS", None)
        else:
            os.environ["CERTICHAIN_RESULTS"] = old


CFG = {"dataset": "ton", "n_camp_p": [0.0, 0.0, 1.0, 0.0], "decoy_rate": 3.0, "ts_round": 1.0}


def _gen(env, split="train", idx=0, cfg=None, variant="cf_ton"):
    from certichain.chainforge import generator as G
    _, lib, bg = env
    return G.generate_incident(variant, split, idx, cfg or CFG, lib, bg)


def test_format_contract_and_quarantine(cf_env, tmp_path):
    from certichain.chainforge import generator as G
    out = _gen(cf_env)
    d = G.write_incident(out, tmp_path / "train" / "000000")
    assert sorted(p.name for p in d.iterdir()) == ["gen.json", "gt.json", "labels.parquet", "meta.json",
                                                    "records.parquet"]
    rec = pl.read_parquet(d / "records.parquet")
    lab = pl.read_parquet(d / "labels.parquet")
    meta = json.loads((d / "meta.json").read_text())
    gt = json.loads((d / "gt.json").read_text())
    assert list(rec.columns) == list(RECORD_SCHEMA) and not label_like_columns(rec.columns)
    assert list(lab.columns) == ["rec_id", "label", "detailed_label", "episode_id", "campaign_id", "decoy_kind",
                                 "template_node"]
    assert set(meta) == {"inc_id", "variant", "pool", "split_role", "seed_device", "t0", "t1", "scope_rule",
                         "internal_prefixes", "generator_seed"}
    assert meta["inc_id"] == "cf_ton/train/000000" and meta["pool"] == "train"
    assert set(gt) == {"campaigns", "episodes", "transitions", "hidden"}
    # no GT leaks into the case metadata
    assert "episode" not in json.dumps(meta) and "campaign" not in json.dumps(meta)
    # labels ↔ gt consistency
    eps = gt["episodes"]
    for eid in lab["episode_id"].drop_nulls().unique().to_list():
        assert eid in eps
    for a, b in gt["transitions"]:
        assert eps[a]["campaign_id"] == eps[b]["campaign_id"]
    kinds = {c: v["kind"] for c, v in gt["campaigns"].items()}
    bad = lab.filter(pl.col("campaign_id").is_not_null()).with_columns(
        pl.col("campaign_id").replace_strict(kinds, default=None).alias("k"))
    assert ((bad["k"] == "main") == bad["decoy_kind"].is_null()).all()
    assert lab.filter(pl.col("episode_id").is_null())["campaign_id"].null_count() == lab.filter(
        pl.col("episode_id").is_null()).height
    assert lab.filter(pl.col("episode_id").is_null())["label"].unique().to_list() == ["0"]
    # P2-B's adapter accepts the files
    from certichain.gt.chainforge_episodes import build_chainforge_gt
    from certichain.units.sessionize import event_units
    eu = event_units(rec, [meta["seed_device"]], internal_prefixes=meta["internal_prefixes"])
    g = build_chainforge_gt(eu.units, lab, gt)
    assert g.transitions == {tuple(t) for t in gt["transitions"]}


def test_rec_id_permutation_rounding_and_order(cf_env):
    from certichain.io.hashing import tiehash_series
    rec = _gen(cf_env)["records"]
    n = rec.height
    assert sorted(rec["rec_id"].to_list()) == list(range(n))
    assert rec["rec_id"].to_list() != list(range(n))
    # not time-ordered: rank correlation with row order ≈ 0
    r = np.corrcoef(np.arange(n), rec["rec_id"].to_numpy())[0, 1]
    assert abs(r) < 0.1
    assert (rec["ts"] % 1 == 0).all() and (rec["te"] % 1 == 0).all() and set(rec["ts_res"].to_list()) == {1.0}
    th = tiehash_series(rec).to_numpy()
    ts = rec["ts"].to_numpy()
    ok = (np.diff(ts) > 0) | ((np.diff(ts) == 0) & (np.diff(th.astype(np.float64)) >= 0))
    assert ok.all()                                   # sorted by (ts, tiehash)


def test_structure_preserved_and_roles(cf_env):
    out = _gen(cf_env)
    _, lib, _ = cf_env
    rec, lab, gen = out["records"], out["labels"], out["gen"]
    j = rec.join(lab, on="rec_id")
    store = pl.read_parquet(cf_env[0] / "interim" / "ton" / "library" / "records_train.parquet")
    for c in gen["campaigns"]:
        roles = c["roles"]
        for e in c["episodes"]:
            if e["hidden"] or not e["lib_eid"]:
                continue
            em = j.filter(pl.col("episode_id") == e["eid"]).with_columns(
                pl.col("src_locator").str.extract(r"rec=(\d+)$").cast(pl.Int64).alias("orig"))
            src = store.filter(pl.col("lib_eid") == e["lib_eid"]).rename({"rec_id": "orig"})
            m = em.join(src, on="orig", suffix="_src")
            assert m.height == em.height > 0
            for col in ("obytes", "rbytes", "opkts", "rpkts", "conn_state", "service", "dport", "proto", "history"):
                assert (m[col] == m[f"{col}_src"]).all(), col
            off = (m["ts"] - m["ts_src"]).unique()
            assert off.len() == 1                      # one constant shift per episode
            row = lib.row(e["lib_eid"])
            binds = e["binds"]
            if binds.get("victim"):
                vic = roles[binds["victim"]]
                assert vic in set(m.filter((pl.col("src_src") == row["slots"]["victim"]))["src"].to_list()
                                  + m.filter(pl.col("dst_src") == row["slots"]["victim"])["dst"].to_list())
            act = roles[binds["actor"]]
            assert act in set(m.filter(pl.col("src_src") == row["slots"]["actor"])["src"].to_list()
                              + m.filter(pl.col("dst_src") == row["slots"]["actor"])["dst"].to_list())
    # password → backdoor on the SAME victim, pivot: victim of password = actor of scanning′
    for c in gen["campaigns"]:
        eps = {e["node"]: e for e in c["episodes"]}
        if c["template"] in ("T-CRED", "T-BOT") and "password" in eps and "backdoor" in eps:
            assert eps["password"]["binds"]["victim"] == eps["backdoor"]["binds"]["victim"] == "X"
        if c["template"] == "T-PIVOT":
            assert eps["password"]["binds"]["victim"] == eps["scanning_2"]["binds"]["actor"] == "X"


def test_source_disjointness_and_determinism(cf_env):
    _, lib, _ = cf_env
    a = _gen(cf_env, "train", 4)
    b = _gen(cf_env, "train", 4)
    assert a["records"].equals(b["records"]) and a["labels"].equals(b["labels"]) and a["gt"] == b["gt"]
    c = _gen(cf_env, "dev", 4)
    assert not c["records"].equals(a["records"])          # train and dev streams differ
    for split, pool in (("train", "train"), ("dev", "train"), ("caltest", "caltest")):
        for idx in range(3):
            out = _gen(cf_env, split, idx)
            assert out["meta"]["pool"] == pool
            for cc in out["gen"]["campaigns"] + out["gen"]["decoys"]:
                for e in cc["episodes"]:
                    if e["lib_eid"]:
                        assert lib.row(e["lib_eid"])["pool"] == pool
                    if e["la_id"]:
                        assert e["la_id"].startswith(pool)
            files = set(out["records"]["src_locator"].str.extract(r"rec=(\d+)$").cast(pl.Int64).to_list())
            codes = {f // 10**6 for f in files}
            assert codes <= ({POOL_CODE[pool], 100 + POOL_CODE[pool]})


def test_decoys_d2_shares_infra_and_d3(cf_env):
    cfg = {**CFG, "n_camp_p": [0.0, 1.0, 0.0, 0.0], "decoy_kind_p": {"D1": 0.0, "D2": 1.0, "D3": 0.0},
           "templates": {"train": ["T-CRED"]}}
    out = _gen(cf_env, "train", 1, cfg)
    camps = out["gen"]["campaigns"]
    d2 = [d for d in out["gen"]["decoys"] if d["kind"] == "D2"]
    assert d2, out["gen"]["log"]
    for d in d2:
        host = next(c for c in camps if c["cid"] == d["host_cid"])
        assert d["roles"]["C2"] == host["roles"]["C2"]               # shared external infrastructure
        assert d["roles"]["X"] != host["roles"]["X"]                 # another device
        if "A" in d["roles"] and "A" in host["roles"]:
            assert d["roles"]["A"] != host["roles"]["A"]             # its own attacker
        assert len(d["episodes"]) >= 2
        assert out["gt"]["campaigns"][d["cid"]]["kind"] == "D2"
        assert any(a.startswith(d["cid"] + ".") for a, _ in out["gt"]["transitions"])
    cfg3 = {**CFG, "decoy_kind_p": {"D1": 0.0, "D2": 0.0, "D3": 1.0}, "decoy_rate": 4.0}
    n_ok = 0
    for split in ("train", "caltest"):
        out3 = _gen(cf_env, split, 2, cfg3)
        lab = out3["labels"]
        _, _, bg = cf_env
        cat = bg.lookalikes(out3["meta"]["pool"])
        for d in (d for d in out3["gen"]["decoys"] if d["kind"] == "D3"):
            e = d["episodes"][0]
            if e["hidden"]:
                continue
            n_ok += 1
            sub = lab.filter(pl.col("campaign_id") == d["cid"])
            assert sub.height > 0 and set(sub["decoy_kind"].to_list()) == {"D3"}
            assert set(sub["label"].to_list()) == {"0"}                    # benign look-alike
            la = cat.filter(pl.col("la_id") == e["la_id"]).row(0, named=True)
            b0, b1 = out3["meta"]["t0"], out3["meta"]["t1"]
            assert la["tile_t1"] <= b0 or la["tile_t0"] >= b1               # from another background window
            srcs = set(sub.join(out3["records"], on="rec_id")["src"].to_list())
            assert srcs == {d["roles"]["X"]}                                  # re-hosted on the campaign victim
    assert n_ok > 0


def test_d1_other_attacker_same_victim(cf_env):
    cfg = {**CFG, "n_camp_p": [0.0, 1.0, 0.0, 0.0], "decoy_kind_p": {"D1": 1.0, "D2": 0.0, "D3": 0.0},
           "decoy_rate": 3.0}
    out = _gen(cf_env, "caltest", 3, cfg)
    camp_ips = {ip for c in out["gen"]["campaigns"] for r, ip in c["roles"].items() if r not in ("X", "Y")}
    victims = {ip for c in out["gen"]["campaigns"] for r, ip in c["roles"].items() if r in ("X", "Y")}
    for d in out["gen"]["decoys"]:
        assert d["kind"] == "D1"
        ext = {ip for r, ip in d["roles"].items() if r not in ("X", "Y")}
        assert ext and not ext & camp_ips                              # a DIFFERENT attacker
        assert d["roles"]["X"] in victims                               # on a campaign victim
        if d["anchor"]["episode"]:
            assert d["anchor"]["lag_s"] <= 1440.0 + 1e-9


def test_evidence_loss_marks_hidden(cf_env):
    cfg = {**CFG, "evidence_loss": 0.6, "decoy_rate": 0.0}
    out = _gen(cf_env, "train", 5, cfg)
    hid = set(out["gt"]["hidden"])
    assert hid
    for eid in hid:
        assert out["labels"].filter(pl.col("episode_id") == eid).height == 0
    base = _gen(cf_env, "train", 5, {**CFG, "decoy_rate": 0.0})
    # paired: same structure, the loss variant only hides episodes
    assert set(base["gt"]["episodes"]) == set(out["gt"]["episodes"])


REAL = os.path.join(os.path.dirname(__file__), "..", "results", "interim", "ton", "library", "episodes.parquet")


@pytest.mark.skipif(not os.path.exists(REAL), reason="real TON library not built")
def test_real_ton_incident_contract():
    old = os.environ.pop("CERTICHAIN_RESULTS", None)
    try:
        from certichain.chainforge import generator as G
        from certichain.chainforge.background import BackgroundLibrary
        from certichain.chainforge.episodes import EpisodeLibrary
        lib, bg = EpisodeLibrary.load("ton"), BackgroundLibrary.load("ton")
        out = G.generate_incident("cf_ton", "train", 7, {"dataset": "ton"}, lib, bg)
        rec, lab = out["records"], out["labels"]
        assert list(rec.columns) == list(RECORD_SCHEMA) and rec["rec_id"].n_unique() == rec.height
        assert set(rec["history"].unique().to_list()) <= {""}            # field parity (P2-A)
        assert rec["http_host"].null_count() == rec.height and rec["tls_sni"].null_count() == rec.height
        assert (rec["ts"] % 1 == 0).all() and (rec["te"] % 1 == 0).all()
        assert lab.height == rec.height
    finally:
        if old is not None:
            os.environ["CERTICHAIN_RESULTS"] = old


GREAL = os.path.join(os.path.dirname(__file__), "..", "results", "interim", "gotham", "library", "episodes.parquet")


@pytest.mark.skipif(not os.path.exists(GREAL), reason="real Gotham library not built")
def test_real_gotham_incident_roles():
    import yaml
    old = os.environ.pop("CERTICHAIN_RESULTS", None)
    try:
        from certichain.chainforge import generator as G
        from certichain.chainforge.background import BackgroundLibrary
        from certichain.chainforge.episodes import EpisodeLibrary
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "cf_gotham.yaml")))
        lib, bg = EpisodeLibrary.load("gotham"), BackgroundLibrary.load("gotham")
        cfg = {**cfg, "templates": {"train": ["T-MIRAI"]}, "n_camp_p": [0.0, 1.0, 0.0, 0.0]}
        out = G.generate_incident("cf_gotham", "train", 3, cfg, lib, bg)
        rec = out["records"]
        assert list(rec.columns) == list(RECORD_SCHEMA) and rec["rec_id"].n_unique() == rec.height
        c = out["gen"]["campaigns"][0]
        r = c["roles"]
        assert len({r[k] for k in ("SCANNER", "LOADER", "DLSRV", "CNC", "REPSRV")}) == 5   # distinct role IPs
        assert r["X"] == out["meta"]["seed_device"] and r["X"] != r["Y"]
        eps = {e["node"]: e for e in c["episodes"]}
        assert eps["REP0"]["hidden"] and "c0.REP0" in out["gt"]["hidden"]
        j = rec.join(out["labels"], on="rec_id")
        itt = j.filter(pl.col("episode_id") == "c0.ITT")
        if itt.height:
            assert set(itt["dst"].to_list()) | set(itt["src"].to_list()) >= {r["X"], r["LOADER"]}
        itt2 = j.filter(pl.col("episode_id") == "c0.ITT_2")
        if itt2.height:
            assert r["Y"] in set(itt2["dst"].to_list()) | set(itt2["src"].to_list())   # pivot: loader → Y
        assert (pl.Series(out["labels"]["episode_id"]).is_null().sum() > 0)
    finally:
        if old is not None:
            os.environ["CERTICHAIN_RESULTS"] = old


def test_campaign_phase_time_shift_keeps_lags_and_removes_integer_starts(cf_env):
    """CF-GOTHAM v2 (E15 fix): one sub-second phase per campaign/decoy; intra-campaign lags exact; no integer starts."""
    cfg_x = {**CFG, "ts_round": None, "time_shift": "exact"}
    cfg_p = {**CFG, "ts_round": None, "time_shift": "campaign_phase"}
    for idx in range(3):
        a, b = _gen(cf_env, "train", idx, cfg_x), _gen(cf_env, "train", idx, cfg_p)
        assert a["gt"] == b["gt"] and a["meta"] == b["meta"]
        ja = a["records"].join(a["labels"], on="rec_id")
        jb = b["records"].join(b["labels"], on="rec_id")
        bg_a = ja.filter(pl.col("episode_id").is_null()).sort("src_locator")["ts"]
        bg_b = jb.filter(pl.col("episode_id").is_null()).sort("src_locator")["ts"]
        assert bg_a.equals(bg_b)                                         # background untouched
        k = ["src_locator", "episode_id"]
        j = (ja.filter(pl.col("episode_id").is_not_null()).select(*k, "campaign_id", "ts")
             .join(jb.select(*k, pl.col("ts").alias("ts2")), on=k))
        d = j.with_columns((pl.col("ts2") - pl.col("ts")).alias("d"))
        spread = d.group_by("campaign_id").agg((pl.col("d").max() - pl.col("d").min()).alias("s"))["s"]
        assert float(spread.max()) < 1e-6                                # lags inside a campaign preserved exactly
        assert float(d["d"].abs().max()) < 1.0                           # sub-second shift only
        sp = jb.filter(pl.col("episode_id").is_not_null())["ts"]
        assert float(((sp - sp.floor()) == 0).mean()) < 0.01             # v1: integer starts; v2: none
