#!/usr/bin/env python
"""E8 — one forensic report end to end (DESIGN §6.1 row E8; paper §VIII E8 box).

LP-IoT23(48-1, 1 %) through the real pipeline of E5 (bench/overhead.py: virtual merged capture → flow table
→ Gateway with real ECVRF / Ed25519 / BLAKE3, real drand rounds, witness, counters with the loss/skew model,
CtS-AI policy of fold 48-1), a causal prefix of 60 epochs. The investigator requests every unit of the infected
host d in [e1, tighten_until) (case information, independent of Z), receives the disclosure (stores of the epochs,
the requested retained units with frames and proofs, counter records, prime bundle if e1 > 0) and runs
Verify + V8 statements (P1-C's estimation/policy.statements_from_v8):

  * transcript: PASS/FAIL/WARN/SKIP per check over the window;
  * statements for the 6 pre-registered queries over [e1, e2) at per-statement α/6 (declared family of 6, so the
    family-wise level is α); lower bounds L⁺ (and U⁺ on clean windows for indicators) with the sure count S;
  * retrospective tightening: min over e'' ≥ e2 of U⁺ of [e1, e'') (DESIGN §3.9);
  * exact metadata answers for d in the window (units, scan units M1, bytes per service port);
  * privacy cost: benign content bytes disclosed and third-party metadata records disclosed;
  * 200 Philox Monte Carlo replications of the selection on the committed π → error bands of L⁺ / U⁺ and their
    coverage of the true value.

Stages: run | report | mc | all.   Output: results/E8/…, results/summary/E8_case.json (box: figures/box_e8.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from ctsample.utils.io import P1_ROOT, RESULTS, atomic_write_json

CFG8 = P1_ROOT / "config" / "experiments" / "E8.yaml"
OUT = RESULTS / "E8_rev1"      # rev1 protocol (committed τ_i); --rev rev0 = the original results/E8
SUMMARY = RESULTS / "summary"
SUMMARY_NAME = "E8_case_rev1.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def e5mod():
    spec = importlib.util.spec_from_file_location("e5", P1_ROOT / "scripts" / "14_e5.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def configs():
    e5 = e5mod()
    c8 = yaml.safe_load(CFG8.read_text())
    cfg = e5.load_cfg()                                 # counter model, trial settings of E5
    for k in ("stream", "fold", "model_seed", "p_key", "rho", "design", "controller", "master_key", "ahead"):
        cfg[k] = c8[k]
    return e5, cfg, c8


def run_dir(c8) -> Path:
    return OUT / "runs" / f"e8_p{c8['p_loss']:g}_s{c8['counter_seed']}"


def stage_run() -> Path:
    e5, cfg, c8 = configs()
    return e5.stage_honest(cfg, float(c8["p_loss"]), int(c8["counter_seed"]), tag=run_dir(c8).name,
                           n_epochs=int(c8["run_epochs"]), out_root=OUT / "runs")


# ============================================================================================== f / b
def _per_epoch_fb(ctx, T, q: str):
    """f and b of query q on the COMMITTED units (mapped to the stream's units by (epoch, key))."""
    si = ctx.units["stream_index"]
    n = len(si)
    f = np.zeros(n)
    b = np.zeros(n)
    hu, hf = T.hits(q)
    fs = np.zeros(T.n_units)
    fs[np.asarray(hu, dtype=np.int64)] = np.asarray(hf, dtype=np.float64)
    bs = np.asarray(T.b_col(q), dtype=np.float64)
    ok = si >= 0
    f[ok] = fs[si[ok]]
    b[ok] = bs[si[ok]]
    pay = ctx.units["pay"].astype(bool)
    f[~pay] = 0.0
    b[~pay] = 0.0
    return f, b


def _split(ctx, x, epochs):
    return {e: x[ctx.epoch_rows(e)] for e in epochs}


# ============================================================================================== report
def stage_report() -> dict:
    from ctsample.estimation.cs import retro_tighten
    from ctsample.estimation.policy import Query, QueryRegistry, report, statements_from_v8
    from ctsample.sim.tables import StreamTables
    e5, cfg, c8 = configs()
    rd = run_dir(c8)
    ctx = e5.TrialCtx(cfg, rd)
    d = ctx.target
    e1, e2 = (int(x) for x in c8["window"])
    eT = int(c8["tighten_until"])
    epochs = list(range(e1, eT))
    T = StreamTables.load(RESULTS / "streams" / cfg["stream"])
    grid = json.loads((RESULTS / "models" / cfg["fold"] / str(cfg["model_seed"]) / "grid.json").read_text())
    gq = grid[cfg["p_key"]][f"{float(cfg['rho']):g}"][cfg["design"]]
    V_ref = {q["q"]: float(gq["V_per_epoch"].get(q["q"]) or 0.0) * int(gq.get("V_ref_epochs", 2880))
             for q in c8["queries"]}
    phi0 = float(ctx.pristine.phi0)
    # ---- disclosure: every unit of d in the epochs (case information, independent of Z)
    dev = ctx.units["device"]
    req = {e: [int(i) for i in np.nonzero(dev[ctx.epoch_rows(e)] == d)[0]] for e in epochs}
    view = ctx.view(epochs)
    disc = ctx.disclose(view, epochs, requested=req)
    bundle = ctx.prime(epochs)
    # ---- statements (declared family of the 6 report queries; tightening queries are auxiliary)
    reg = QueryRegistry(float(c8["alpha"]))
    reg.declare_family("E8 report", len(c8["queries"]))
    fb = {q["q"]: _per_epoch_fb(ctx, T, q["q"]) for q in c8["queries"]}
    F = {q: _split(ctx, v[0], epochs) for q, v in fb.items()}
    B = {q: _split(ctx, v[1], epochs) for q, v in fb.items()}
    DEV = _split(ctx, dev, epochs)
    queries = [Query(name=q["name"], q=q["q"], devices=(d,), window=(e1, e2), kind=q["kind"], family="E8 report")
               for q in c8["queries"]]
    for q in queries:
        reg.register(q)
    holder = {}

    def hook(rep):
        v8 = rep.v8_inputs()
        main = [statements_from_v8(v8, q, F[q.q], B[q.q], DEV, V_ref=max(V_ref[q.q], 1e-6), bbar=1.0, phi0=phi0,
                                   registry=reg, n_statements=len(queries)) for q in queries]
        tight = {}
        for q in queries:
            if q.kind == "count":
                continue
            seq = []
            for ee in range(e2, eT + 1):
                q2 = Query(name=q.name, q=q.q, devices=(d,), window=(e1, ee), kind=q.kind, family="E8 report")
                s = statements_from_v8(v8, q2, F[q.q], B[q.q], DEV, V_ref=max(V_ref[q.q], 1e-6), bbar=1.0,
                                       phi0=phi0, registry=reg, n_statements=len(queries))
                seq.append(np.nan if s["U_plus"] is None else float(s["U_plus"]))
            tt = retro_tighten(np.asarray(seq))
            tight[q.q] = {"U_plus_by_end": seq, "tightened_U_plus": None if np.isnan(tt[0]) else float(tt[0]),
                          "ends": list(range(e2, eT + 1))}
        holder["tight"] = tight
        return report(main)

    from ctsample.protocol.prime import verify_with_prime
    from ctsample.protocol.verifier import TrustAnchors, Verifier
    from ctsample.bench import overhead as O
    trust = TrustAnchors.from_disclosure(disc, ctx.dep.keys.pk_W, pk_I=ctx.dep.keys.pk_I, beacon=ctx.dep.beacon)
    ver = Verifier(trust, O.fresh_policy(ctx.pristine), eps_cell=ctx.eps, meta_codec=ctx.codec, statements_hook=hook)
    t0 = time.perf_counter()
    rep = verify_with_prime(ver, disc, bundle, epochs=epochs)
    vsec = time.perf_counter() - t0
    # ---- transcript
    by_check = {}
    for r in rep.results:
        by_check.setdefault(r.check, {}).setdefault(r.status, 0)
        by_check[r.check][r.status] += 1
    non_pass = [r.line() for r in rep.results if r.status in ("FAIL", "WARN")]
    in_win = [e for e in range(e1, e2)]
    win_fail = [r.line() for r in rep.results if r.status == "FAIL" and r.epoch in in_win]
    # ---- truth and exact metadata answers for d over [e1, e2)
    sl = slice(int(ctx.ptr[e1]), int(ctx.ptr[e2]))
    md = dev[sl] == d
    truth = {q["q"]: float(fb[q["q"]][0][sl][md].sum()) for q in c8["queries"]}
    from ctsample.units import encode as enc
    stores = {e: view.records[e].store for e in range(e1, e2)}
    n_units = n_pay = n_scan = pk = by = 0
    port_bytes: dict = {}
    for e, st in stores.items():
        for r in st.records:
            if r.device_id != d:
                continue
            n_units += 1
            n_pay += int(r.pay)
            pk += r.n_pkts
            by += r.n_bytes
            m = enc.dec_metadata(r.meta)
            if not r.pay and m["proto"] == 6 and enc.CONN_STATES[m["conn_state"]] in ("S0", "REJ", "RSTOS0", "SH"):
                n_scan += 1
            key = (m["proto"], m["port_b"])
            port_bytes[key] = port_bytes.get(key, 0) + r.n_bytes
    top_ports = sorted(port_bytes.items(), key=lambda kv: -kv[1])[:5]
    # ---- privacy cost of this disclosure
    lab = ctx.labels
    benign_b = mal_b = 0
    n_disc_units = 0
    for e, ed in disc.epochs.items():
        rows = ctx.epoch_rows(e)
        for i, du in ed.units.items():
            n_disc_units += 1
            sz = sum(f.caplen + 16 for f in du.frames)
            if lab[rows][i] >= 1:
                mal_b += sz
            else:
                benign_b += sz
    third = sum(sum(1 for r in ed.store.records if r.device_id != d) for ed in disc.epochs.values())
    own = sum(sum(1 for r in ed.store.records if r.device_id == d) for ed in disc.epochs.values())
    stm = rep.statements
    res = {"stream": cfg["stream"], "fold": cfg["fold"], "device": int(d), "device_mac": ctx.info["devices"]["seed_mac"],
           "window": [e1, e2], "epochs_disclosed": [epochs[0], epochs[-1]],
           "window_utc": [_utc(ctx.dep.clock.t_e(e1)), _utc(ctx.dep.clock.t_e(e2))],
           "alpha": c8["alpha"], "transcript": by_check, "non_pass": non_pass[:40], "window_fail": win_fail,
           "W_size": int(sum(len(v.W) for e, v in rep.epochs.items() if e1 <= e < e2)), "verify_s": vsec,
           "statements": stm, "tightening": holder.get("tight"), "truth": truth, "V_ref": V_ref, "phi0": phi0,
           "metadata": {"units": n_units, "payload_units": n_pay, "scan_units_M1": n_scan, "packets": pk, "bytes": by,
                        "top_ports_by_bytes": [{"proto": k[0], "port": k[1], "bytes": v} for k, v in top_ports]},
           "privacy": {"disclosed_retained_units": n_disc_units, "benign_bytes_disclosed": benign_b,
                       "malicious_bytes_disclosed": mal_b, "third_party_records_disclosed": third,
                       "own_records_disclosed": own},
           "disclosure_size": disc.size_report(), "prime": rep.prime,
           "retained_of_d_in_window": int((ctx.units["Z"][sl] & md).sum()),
           "payload_units_of_d_in_window": int((ctx.units["pay"][sl] & md).sum()),
           "unmatched_committed_units": int((ctx.units["stream_index"] < 0).sum())}
    (OUT / "report").mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT / "report" / "report.json", _clean(res))
    log("report written; transcript", by_check)
    return res


def _utc(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(x, bytes):
        return x.hex()
    return x


# ============================================================================================== mc
def stage_mc() -> dict:
    from ctsample.estimation.policy import Query, QueryRegistry, statements_from_v8
    from ctsample.sim.montecarlo import philox_epoch_uniforms, rep_keys, stable_id
    from ctsample.sim.tables import StreamTables
    e5, cfg, c8 = configs()
    rd = run_dir(c8)
    ctx = e5.TrialCtx(cfg, rd)
    d = ctx.target
    e1, e2 = (int(x) for x in c8["window"])
    epochs = list(range(e1, e2))
    T = StreamTables.load(RESULTS / "streams" / cfg["stream"])
    grid = json.loads((RESULTS / "models" / cfg["fold"] / str(cfg["model_seed"]) / "grid.json").read_text())
    gq = grid[cfg["p_key"]][f"{float(cfg['rho']):g}"][cfg["design"]]
    V_ref = {q["q"]: float(gq["V_per_epoch"].get(q["q"]) or 0.0) * int(gq.get("V_ref_epochs", 2880))
             for q in c8["queries"]}
    phi0 = float(ctx.pristine.phi0)
    raw = ctx.records(epochs)
    reg = QueryRegistry(float(c8["alpha"]))
    reg.declare_family("E8 report", len(c8["queries"]))
    queries = [Query(name=q["name"], q=q["q"], devices=(d,), window=(e1, e2), kind=q["kind"], family="E8 report")
               for q in c8["queries"]]
    for q in queries:
        reg.register(q)
    fb = {q["q"]: _per_epoch_fb(ctx, T, q["q"]) for q in c8["queries"]}
    F = {q: _split(ctx, v[0], epochs) for q, v in fb.items()}
    B = {q: _split(ctx, v[1], epochs) for q, v in fb.items()}
    dev = ctx.units["device"]
    DEV = _split(ctx, dev, epochs)
    pay = {e: ctx.units["pay"][ctx.epoch_rows(e)].astype(bool) for e in epochs}
    taus = {e: raw[e]["taus"] for e in epochs}
    pis = {e: np.array([int(t) / 2.0 ** 64 for t in taus[e]]) for e in epochs}
    R = int(c8["mc_reps"])
    keys = rep_keys(stable_id(rd.name), stable_id("E8-mc"), range(R))
    n_e = np.array([len(taus[e]) for e in epochs])
    U = philox_epoch_uniforms(keys, n_e)
    off = np.concatenate([[0], np.cumsum(n_e)])
    sl = slice(int(ctx.ptr[e1]), int(ctx.ptr[e2]))
    md = dev[sl] == d
    truth = {q.q: float(fb[q.q][0][sl][md].sum()) for q in queries}
    out = {q.q: {"L_plus": [], "U_plus": [], "S": []} for q in queries}
    for r in range(R):
        v8 = {}
        for k, e in enumerate(epochs):
            Z = (U[r, off[k]:off[k + 1]] < pis[e]) & pay[e]
            v8[e] = {"tau": taus[e], "Z": Z.tolist(), "pay": pay[e].tolist(), "W": [], "W_all": False,
                     "requested": [int(i) for i in np.nonzero(DEV[e] == d)[0]], "failed": False, "floor_mode": 0,
                     "flagged_devices": []}
        for q in queries:
            s = statements_from_v8(v8, q, F[q.q], B[q.q], DEV, V_ref=max(V_ref[q.q], 1e-6), bbar=1.0, phi0=phi0,
                                   registry=reg, n_statements=len(queries))
            out[q.q]["L_plus"].append(float(s["L_plus"]))
            out[q.q]["S"].append(float(s["S"]))
            out[q.q]["U_plus"].append(np.nan if s["U_plus"] is None else float(s["U_plus"]))
    res = {"reps": R, "window": [e1, e2], "truth": truth, "queries": {}}
    for q in queries:
        L = np.asarray(out[q.q]["L_plus"])
        Uu = np.asarray(out[q.q]["U_plus"])
        S = np.asarray(out[q.q]["S"])
        iss = ~np.isnan(Uu)
        res["queries"][q.q] = {"name": q.name, "kind": q.kind, "truth": truth[q.q],
                               "L_plus_q05_50_95": np.quantile(L, [0.05, 0.5, 0.95]).tolist(),
                               "S_q05_50_95": np.quantile(S, [0.05, 0.5, 0.95]).tolist(),
                               "U_plus_q05_50_95": np.quantile(Uu[iss], [0.05, 0.5, 0.95]).tolist() if iss.any() else None,
                               "ucs_issued_share": float(iss.mean()),
                               "lcs_cover": float(np.mean(L <= truth[q.q] + 1e-9)),
                               "ucs_cover": float(np.mean(Uu[iss] >= truth[q.q] - 1e-9)) if iss.any() else None}
    (OUT / "mc").mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT / "mc" / "mc.json", _clean(res))
    log("mc written")
    return res


def stage_summary() -> None:
    rep = json.loads((OUT / "report" / "report.json").read_text())
    mc = json.loads((OUT / "mc" / "mc.json").read_text()) if (OUT / "mc" / "mc.json").exists() else None
    run = json.loads((run_dir(configs()[2]) / "run.json").read_text())
    SUMMARY.mkdir(parents=True, exist_ok=True)
    atomic_write_json(SUMMARY / SUMMARY_NAME, {"report": rep, "mc": mc,
                                                  "run": {k: run[k] for k in ("n_epochs", "Bbar", "policy", "loss",
                                                                              "retention", "spend_over_target",
                                                                              "devices", "validation", "machine")}})
    log("summary written")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "report", "mc", "summary", "all"])
    ap.add_argument("--rev", default="rev1", choices=["rev1", "rev0"])
    a = ap.parse_args(argv)
    global OUT, SUMMARY_NAME
    if a.rev == "rev0":
        OUT, SUMMARY_NAME = RESULTS / "E8", "E8_case.json"
    if a.stage in ("run", "all"):
        stage_run()
    if a.stage in ("report", "all"):
        stage_report()
    if a.stage in ("mc", "all"):
        stage_mc()
    if a.stage in ("summary", "all"):
        stage_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
