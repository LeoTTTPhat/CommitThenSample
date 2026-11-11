"""streams/lp.py: thinning hits the target prevalence within the binomial error; whole-epoch offsets keep
unit boundaries; the address remap preserves f (and every non-address field of the metadata)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ctsample.streams import lp
from ctsample.units import features as F

RES = Path(__file__).resolve().parents[1] / "results"


def test_thinning_within_binomial_error():
    rng = np.random.default_rng(0)
    n_conn = 3000
    sizes = rng.geometric(0.6, n_conn)                     # units per malicious payload connection
    n_mal = int(sizes.sum())
    n_ben = 140_000
    for p in (0.001, 0.01, 0.05):
        zs = []
        for seed in range(60):
            th = lp.thin_connections(np.arange(n_conn), n_mal, n_ben, p, [7, seed])
            q = th["q"]
            kept = int(sizes[th["keep"]].sum())
            sd = math.sqrt(q * (1 - q) * float((sizes ** 2).sum()))
            if q < 1:
                zs.append((kept - q * n_mal) / sd)
                assert abs(kept - q * n_mal) <= 4.5 * sd
            else:
                assert kept == n_mal
        if zs:
            assert abs(np.mean(zs)) < 4 / math.sqrt(len(zs))
            # the target itself: E[kept] / (n_ben + E[kept]) = p exactly
            assert abs(q * n_mal / (n_ben + q * n_mal) - p) < 1e-12
    # determinism: same seed -> same keep mask
    a = lp.thin_connections(np.arange(n_conn), n_mal, n_ben, 0.01, [3, 10000])["keep"]
    b = lp.thin_connections(np.arange(n_conn), n_mal, n_ben, 0.01, [3, 10000])["keep"]
    assert np.array_equal(a, b)


def test_incidents_split_at_gaps():
    T = 1_000_000
    ts = np.array([0, 10, 20, 700, 2000, 2010, 5, 15]) * T
    dev = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    fam = np.array(["C&C"] * 6 + ["C&C", "Attack"])
    content = np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    order = np.argsort(ts, kind="stable")
    inc, table = lp.incidents_from_units(dev[order], fam[order], ts[order], content[order])
    inc_u = np.empty_like(inc)
    inc_u[order] = inc
    # device 0 C&C: {0,10,20} | gap 680 s | {700} | gap > 600 | {2010}; device 1: two families
    assert inc_u[0] == inc_u[1] == inc_u[2]
    assert len({inc_u[0], inc_u[3], inc_u[5]}) == 3 and inc_u[4] == -1
    assert inc_u[6] != inc_u[7]
    assert sorted(r["k"] for r in table) == [1, 1, 1, 1, 3]


def test_remap_ip16():
    a = np.frombuffer(lp.ip16("192.168.1.195") * 3 + lp.ip16("8.8.8.8"), dtype=np.uint8).reshape(4, 16).copy()
    n = lp.remap_ip16(a, lp.ip16("192.168.1.195"), lp.ip16("192.168.137.234"))
    assert n == 3 and bytes(a[0]) == lp.ip16("192.168.137.234") and bytes(a[3]) == lp.ip16("8.8.8.8")


# ------------------------------------------------------------------------------------------ #
def _have_data():
    need = [RES / "units" / "ciciot2023_benign" / "BenignTraffic1" / "DONE",
            RES / "units" / "iot23" / "CTU-IoT-Malware-Capture-34-1" / "DONE",
            RES / "labels" / "iot23" / "CTU-IoT-Malware-Capture-34-1.parquet", RES / "inventory.json"]
    return all(p.exists() for p in need)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not _have_data():
        pytest.skip("S1/S3 outputs not present")
    from ctsample.streams import recipe as R
    cfg, inv = R.load_config(), R.load_inventory()
    spec = R.lp_iot23_spec(cfg, inv, "CTU-IoT-Malware-Capture-34-1", 0.001)
    out = tmp_path_factory.mktemp("streams")
    lp.build_stream(spec, out_root=out, force=True, log=lambda m: None)
    return spec, out / spec.stream_id


def test_real_stream_prevalence_offsets_remap(built):
    import polars as pl
    from ctsample.sim.tables import StreamTables
    from ctsample.streams import recipe as R
    spec, d = built
    T = StreamTables.load(d)
    T.validate()
    ex = lp.load_stream_extra(d)
    ut = pl.read_parquet(d / "units.parquet")
    assert ut.height == T.n_units
    # ---- prevalence within the cluster-binomial error
    fg = spec.foregrounds[0]
    src, _ = R.load_source("iot23", fg.source, fg.src_epoch0, fg.src_epoch0 + fg.n_epochs)
    mp = src.filter(pl.col("pay") & pl.col("is_malicious").fill_null(False))
    sizes = mp.group_by("flow_uid").len()["len"].to_numpy()
    q = ex["q"]
    assert 0 < q < 1
    kept = ex["n_mal_payload_units"]
    sd = math.sqrt(q * (1 - q) * float((sizes ** 2).sum()))
    assert abs(kept - q * sizes.sum()) <= 4 * sd
    # ---- whole-epoch offsets keep unit boundaries (first and last packet in the unit's stream epoch)
    c = F.cols_from_meta(ut["meta"].to_arrow())
    t0 = ex["t0_stream_us"]
    ep = ut["epoch"].to_numpy().astype(np.int64)
    assert np.array_equal((c["first_ts"] - t0) // lp.T_US, ep)
    pay = c["pay"]
    assert np.array_equal((c["last_ts"][pay] - t0) // lp.T_US, ep[pay])
    assert np.array_equal(np.asarray(T.epoch), ep)
    # canonical order inside each epoch
    keyb = np.concatenate([c["proto"].astype(np.uint8)[:, None], c["ip_a"],
                           (c["port_a"] & 0xFFFF).astype("<u2").view(np.uint8).reshape(-1, 2), c["ip_b"],
                           (c["port_b"] & 0xFFFF).astype("<u2").view(np.uint8).reshape(-1, 2)], axis=1)
    rows = list(zip(ep.tolist(), c["first_ts"].tolist(), [bytes(r) for r in keyb]))
    assert rows == sorted(rows)
    # ---- remap: seed IP gone, new IP present, f/b preserved, non-address metadata preserved
    seed_ip, new_ip = list(fg.ip_map.items())[0]
    allip = {bytes(x) for x in np.concatenate([c["ip_a"], c["ip_b"]])}
    assert lp.ip16(seed_ip) not in allip and lp.ip16(new_ip) in allip
    is_fg = (ut["source"] == fg.source).to_numpy()
    st = ut.filter(pl.Series(is_fg)).select(["stream_index", "unit_gid"])
    j = st.join(src, on="unit_gid", how="left")
    assert j.height == int(is_fg.sum()) and j["epoch"].null_count() == 0
    sidx = j["stream_index"].to_numpy()
    for qi, qn in enumerate(T.query_names):
        f_stream = T.f_dense(qn)[sidx]
        if qn.startswith("L"):
            continue
        f_src = np.where(j["pay"].to_numpy(), j[f"f_{qn}"].fill_null(0).to_numpy(), 0)
        b_src = np.where(j["pay"].to_numpy(), j[f"b_{qn}"].fill_null(0).to_numpy(), 0)
        assert np.array_equal(f_stream, f_src), qn
        assert np.array_equal(T.b_col(qn)[sidx], b_src), qn
    c_src = F.cols_from_meta(j["meta"].to_arrow())
    c_st = F.take_cols(c, sidx)
    for k in ("pk0", "pk1", "pb0", "pb1", "fb0", "fb1", "syn", "synack", "fin", "rst", "psh", "cs", "s", "cont",
              "fps", "pay", "sizes", "iats", "port_a", "port_b", "proto"):
        assert np.array_equal(c_src[k], c_st[k]), k
    d_us = c_st["first_ts"] - c_src["first_ts"]
    assert np.all(d_us == d_us[0]) and d_us[0] % lp.T_US == (ex["t0_stream_us"] - R.source_t0("iot23", fg.source)) % lp.T_US
    # static features (incl. the recomputed direction class) are unchanged by the remap
    assert np.array_equal(F.static_features(c_src), F.static_features(c_st))
    Xp = np.load(d / "features_pay.npy")
    assert Xp.shape == (int(np.asarray(T.pay).sum()), F.N_FEATURES)
