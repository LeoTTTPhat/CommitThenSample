"""Verify variant streams against their base streams and tabulate the zero-payload thinning.

    .venv/bin/python -I P1_CommitThenSample/scripts/checks/sv_verify_variant.py <variant> [<out_json>]   (P1-X-SV, rev1)
checks per stream: payload units identical (order, s, label, content, incident, epoch, every f/b column, payload
metadata columns except device_id), non-seed units identical (test streams: unit_gid order), q identical;
q = 1 streams: every array identical to the base.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ctsample.sim.tables import StreamTables  # noqa: E402
from ctsample.utils.io import RESULTS  # noqa: E402

v = sys.argv[1]
idx = json.loads((RESULTS / "streams" / f"index_{v}.json").read_text())["streams"]
rows = []
ok_all = True
for r in sorted(idx, key=lambda r: (r["kind"], r["stream_id"])):
    sid, base = r["stream_id"], r["base_stream_id"]
    dv, db = RESULTS / "streams" / sid, RESULTS / "streams" / base
    Tv, Tb = StreamTables.load(dv), StreamTables.load(db)
    mv, mb = json.loads((dv / "meta.json").read_text()), json.loads((db / "meta.json").read_text())
    ev, eb = mv["extra"], mb["extra"]
    pv, pb = np.nonzero(np.asarray(Tv.pay))[0], np.nonzero(np.asarray(Tb.pay))[0]
    bad = []
    if ev["q"] != eb["q"]:
        bad.append("q")
    if mv["stream_id"] != mb["stream_id"]:
        bad.append("int stream id")
    if len(pv) != len(pb):
        bad.append("n payload")
    else:
        for name in ("s", "label", "content", "incident", "epoch"):
            if not np.array_equal(np.asarray(getattr(Tv, name))[pv], np.asarray(getattr(Tb, name))[pb]):
                bad.append(name)
        for qn in Tb.query_names:
            if not np.array_equal(Tv.f_dense(qn)[pv], Tb.f_dense(qn)[pb]) or \
                    not np.array_equal(np.asarray(Tv.b_col(qn))[pv], np.asarray(Tb.b_col(qn))[pb]):
                bad.append(f"f/b {qn}")
        zv, zb = np.load(dv / "cols_pay.npz"), np.load(db / "cols_pay.npz")
        for k in zb.files:
            if k == "device_id":
                continue
            if not np.array_equal(zv[k], zb[k]):
                bad.append(f"cols_pay {k}")
        if (dv / "units.parquet").exists():
            uv = pl.read_parquet(dv / "units.parquet", columns=["unit_gid", "pay", "source", "device_mac"])
            ub = pl.read_parquet(db / "units.parquet", columns=["unit_gid", "pay", "source", "device_mac"])
            if not uv.filter(pl.col("pay"))["unit_gid"].equals(ub.filter(pl.col("pay"))["unit_gid"]):
                bad.append("payload unit_gid order")
            seed_new = list(eb["recipe"]["foregrounds"][0]["mac_map"].values())[0]
            if not uv.filter(pl.col("device_mac") != seed_new)["unit_gid"].equals(
                    ub.filter(pl.col("device_mac") != seed_new)["unit_gid"]):
                bad.append("non-seed units")
    z = ev["zero_thin"]
    zp = z["parts"][0]
    identical = None
    if z["keep_prob"] >= 1.0:
        names = sorted(p.name for p in db.glob("*.npy") if not p.name.startswith("util_"))
        identical = all(np.array_equal(np.load(dv / n), np.load(db / n)) for n in names) and \
            np.array_equal(np.load(dv / "features_pay.npy"), np.load(db / "features_pay.npy"))
        if not identical:
            bad.append("q=1 stream not identical")
    ok_all &= not bad
    rows.append({"stream_id": sid, "base": base, "kind": r["kind"], "q": ev["q"], "keep_prob": z["keep_prob"],
                 "n_units_base": mb["n_units"], "n_units_variant": mv["n_units"],
                 "seed_zero_before": zp["n_seed_zero_units_before"], "seed_zero_after": zp["n_seed_zero_units_after"],
                 "pure_zero_conns_before": zp["n_pure_zero_conns_before"], "pure_zero_conns_after": zp["n_pure_zero_conns_after"],
                 "pure_zero_units_before": zp["n_pure_zero_units_before"], "pure_zero_units_after": zp["n_pure_zero_units_after"],
                 "zero_in_payload_conns": zp["n_zero_in_mal_payload_conns"] + zp["n_zero_in_other_payload_conns"],
                 "realised_keep": z["realised_keep_rate"],
                 "s_all_over_s_pay_base": eb["s_all_over_s_pay"], "s_all_over_s_pay_variant": ev["s_all_over_s_pay"],
                 "all_arrays_identical_to_base": identical, "checks_failed": bad})
    print(f"{sid:28s} q={ev['q']:.4f} keep={z['keep_prob']:.4f} units {mb['n_units']:>9,} -> {mv['n_units']:>9,} "
          f"seed zero {zp['n_seed_zero_units_before']:>9,} -> {zp['n_seed_zero_units_after']:>9,} "
          f"conns {zp['n_pure_zero_conns_before']:>9,} -> {zp['n_pure_zero_conns_after']:>9,} "
          f"(rate {z['realised_keep_rate']:.4f}) S_all/S_pay {eb['s_all_over_s_pay']:.3f} -> {ev['s_all_over_s_pay']:.3f} "
          f"{'OK' if not bad else 'FAIL ' + str(bad)}{' identical' if identical else ''}", flush=True)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps({"variant": v, "streams": rows, "all_checks_ok": bool(ok_all)}, indent=2))
print("ALL OK" if ok_all else "SOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
