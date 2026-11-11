"""E7: which design choices matter (DESIGN Sec. 6.1 row E7; Prop. 4, Thm. 1 + Cor. 1). Analytic + 200 MC.

(a) detection--variance frontier over gamma in {0.1,0.2,0.3,0.5}: the floor consumes gamma of the budget, so
    worst-case (floor-only) detection 1-(1-phi0(gamma))^k rises with gamma while the utility channel's
    model-based variance V^wr rises with gamma (less budget left). Analytic over the LP-IoT23 folds; the
    headline stream also gets 200 MC repetitions of the realised LCB width.
(b) two-tier trade-off g_mu(phi_t) for phi_t in {0.1,0.3,0.5,1}, m in {1,3,10}, H = 1 KB, with the measured
    benign head share C_H/C_T (Prop. 4(ii)). phi_t = 1 (single tier) is optimal for m = 1.
(c) upper-bound (UCB) null slack vs phi0 against the Thm. 1 / Cor. 1 capacity bound (ratio ~ 1.19).
Supplement: LCS K in {16,32,64} radius ablation; budget mode horizon vs per-epoch proportional.

Outputs: results/E7/<cfg_hash>/{frontier.json, two_tier.json, capacity.json, supplement.json, DONE}.
Summaries: results/summary/E7_frontier.parquet, E7_two_tier.json, E7_capacity.json, E7.json.

    .venv/bin/python memguard/memguard.py run --name p1_e7 --workers 8 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/16_e7.py [--mc] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctsample.config import cfg_hash, load_yaml  # noqa: E402
from ctsample.utils.io import RESULTS, atomic_write_json, is_done, mark_done  # noqa: E402
from ctsample.utils.mem import check_memory  # noqa: E402

STREAMS = RESULTS / "streams"
MODELS = RESULTS / "models"
DESIGNS = RESULTS / "designs"
E7 = RESULTS / "E7"
SUMMARY = RESULTS / "summary"
CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "experiments" / "E7.yaml"


# --------------------------------------------------------------------------------------------- #
def _load(sid):
    from ctsample.sim.tables import StreamTables
    return StreamTables.load(STREAMS / sid)


def frontier(cfg_e7) -> dict:
    """(a) detection--variance frontier over gamma, analytic, per LP-IoT23 fold + the headline stream."""
    from ctsample.design.controller import calibrate_lambda
    from ctsample.design.floor import phi0_from_training
    from ctsample.design.probs import committed_floor, design_pi, quantise_utility, sqrt_eff
    from ctsample.sim.analytic import var_ht_queries
    rho = float(cfg_e7.get("rho", 0.05))
    seed = cfg_e7["seeds"][0]
    ppm = cfg_e7["streams"]["headline_ppm"]
    rows = []
    for fold in cfg_e7["streams"]["frontier_folds"]:
        sid = f"lp_iot23_{fold}_p{ppm}"
        if not is_done(STREAMS / sid):
            continue
        T = _load(sid)
        pay = np.asarray(T.pay, bool)
        pay_idx = np.nonzero(pay)[0]
        s = np.asarray(T.s, float)
        sp = s[pay_idx]
        sq = sqrt_eff(sp)
        E = T.n_epochs
        Bbar = rho * s.sum() / E
        s_all, s_pay = float(s.sum()), float(sp.sum())
        util = f"U5_s{seed}"
        a = quantise_utility(np.asarray(T.util(util), float)[pay_idx]) if util in T.utils else np.ones(len(sp))
        content = (np.asarray(T.content, bool) if T.content is not None else np.asarray(T.incident) >= 0)
        inc = np.asarray(T.incident)
        # reference incident size k = the largest content incident (persistent C&C)
        kc = np.bincount(inc[content & (inc >= 0)]) if (content & (inc >= 0)).any() else np.array([0])
        k_ref = int(kc.max()) if kc.size else 0
        for gamma in cfg_e7["gamma_grid"]:
            phi0 = phi0_from_training(gamma, rho, s_all, s_pay)
            cf = committed_floor(phi0)
            # budget-matched mixture (fixed lambda for each component on the stream's own metadata)
            lamA = calibrate_lambda(sp, a, Bbar, cf, E, component="A")
            lamS = calibrate_lambda(sp, None, Bbar, cf, E, component="S")
            pi = design_pi("mixture", [lamA, lamS], a, sp, cf, tau_mix=0.2, sqrt_s=sq)
            pi_full = np.zeros(T.n_units)
            pi_full[pay_idx] = pi
            v = var_ht_queries(T, pi_full)
            wrv = float(np.nanmax(v["rel_var"])) if len(v["rel_var"]) else float("nan")
            geo = float(np.exp(np.nanmean(np.log(v["rel_var"][np.isfinite(v["rel_var"]) & (v["rel_var"] > 0)]))))
            det_floor = 1.0 - (1.0 - cf) ** k_ref
            det_k = {str(kk): float(1.0 - (1.0 - cf) ** kk) for kk in (1, 5, 10, 50, 200)}
            rows.append({"fold": fold, "stream": sid, "gamma": float(gamma), "phi0": cf, "k_ref": k_ref,
                         "worst_rel_var": wrv, "geo_rel_var": geo, "detect_floor_kref": det_floor,
                         "detect_k1": det_k["1"], "detect_k10": det_k["10"], "detect_k50": det_k["50"],
                         "detect_k200": det_k["200"],
                         "share_at_floor": float(np.mean(pi <= cf * (1 + 1e-9))), "lamA": float(lamA),
                         "lamS": float(lamS), "spend_ratio": float(np.sum(pi * sp) / (Bbar * E))})
    return {"rho": rho, "ppm": ppm, "rows": rows, "frontier_stream": cfg_e7["streams"]["frontier_stream"],
            "note": ("worst_rel_var is near-flat and geo_rel_var decreases slightly in gamma at rho=5%: most "
                     "units sit at the floor (share_at_floor 0.5-0.7), so a larger floor raises those units' "
                     "pi and lowers their HT variance, while detection rises with gamma -- the knee is in "
                     "detection vs gamma (diminishing returns), per-unit detection = phi0 = detect_k1.")}


def two_tier(cfg_e7) -> dict:
    """(b) g_mu(phi_t) with measured head shares (Prop. 4(ii))."""
    from ctsample.design.floor import worst_case_retention, two_tier_retention, phi_cap_two_tier, phi_t_star
    rho = float(cfg_e7.get("rho", 0.05))
    gamma = 0.3
    H = cfg_e7["two_tier"]["H_bytes"]
    # measured benign head shares: CIC background and IoT-23 34-1 foreground (DESIGN Sec. 0.1), plus the
    # LP-stream aggregate from E4's head_tail_costs if available.
    head_shares = {"CIC": 0.10, "IoT23_34-1_fg": 0.028}
    e4 = RESULTS / "summary" / "E4_mimicry.json"
    if e4.exists():
        m = json.loads(e4.read_text())
        for sid, d in m.items():
            if d.get("two_tier") and d["two_tier"].get("head_share") is not None:
                head_shares[sid] = float(d["two_tier"]["head_share"])
    curves = {}
    phi_t_grid = np.asarray(cfg_e7["two_tier"]["phi_t_grid"], float)
    phi_star = 0.015                               # realistic single-tier floor (DESIGN Sec. 0.1); keeps
    for label, h in head_shares.items():           # phi_cap(phi_t) < 1 so the g_mu curves are not clipped
        C_H, C_T = h, 1.0 - h                       # normalised so C^0 = 1
        E = 1.0
        Bbar = phi_star / gamma                      # => gamma*Bbar*E/C0 = phi_star (single-tier floor)
        cur = {}
        for m in cfg_e7["two_tier"]["m_grid"]:
            unp, pad = two_tier_retention(phi_t_grid, m, gamma, Bbar, E, C_H, C_T)
            g = worst_case_retention(phi_t_grid, m, gamma, Bbar, E, C_H, C_T)
            pc = phi_cap_two_tier(phi_t_grid, gamma, Bbar, E, C_H, C_T)
            star = phi_t_star(m, gamma, Bbar, E, C_H, C_T)
            cur[str(m)] = {"phi_t": phi_t_grid.tolist(), "phi_cap": np.asarray(pc).tolist(),
                           "unpadded": np.asarray(unp).tolist(), "padded": np.asarray(pad).tolist(),
                           "worst": np.asarray(g).tolist(),
                           "unpadded_ratio": (np.asarray(unp) / phi_star).tolist(),
                           "padded_ratio": (np.asarray(pad) / phi_star).tolist(),
                           "phi_t_star": star[0], "g_star": star[1], "phi_star": phi_star}
        curves[label] = {"head_share": h, "m": cur}
    return {"gamma": gamma, "H_bytes": H, "phi_star": phi_star, "curves": curves}


def capacity(cfg_e7) -> dict:
    """(c) UCB null slack vs phi0 against the Thm. 1 / Cor. 1 bound."""
    from ctsample.estimation.capacity import null_slack_ratio
    c = cfg_e7["capacity"]
    alpha_side = c["alpha"] / 2.0
    phi0s = np.linspace(c["phi0_grid_min"], c["phi0_grid_max"], c["n_phi0"])
    rows = []
    for phi0 in phi0s:
        ns = null_slack_ratio(float(phi0), alpha_side)
        rows.append({"phi0": float(phi0), "r_plus": ns["r_plus"], "r_plus_bound": ns["r_plus_bound"],
                     "H_cont": ns["H_cont"], "H_units": ns["H_units"], "cbar": ns["cbar"],
                     "ratio": ns["r_plus"] / ns["H_cont"], "ratio_log": ns["ratio_log"]})
    # the two reference floors
    refs = {}
    for phi0 in (0.015, 0.029):
        refs[str(phi0)] = null_slack_ratio(phi0, alpha_side)
    return {"alpha_side": alpha_side, "rows": rows, "refs": refs,
            "cor1_ratio_log80_40": math.log(1 / (alpha_side * 0.5)) / math.log(1 / alpha_side)}


def supplement(cfg_e7, do_mc: bool) -> dict:
    """LCS K-grid radius ablation and the budget-mode (horizon vs per-epoch) comparison."""
    from ctsample.estimation.cs import lcs_grid, lcs_radius
    out = {"lcs_K": {}}
    # LCS radius vs K at a representative per-epoch V_ref x horizon (Cor.-style null radius)
    V_ref = 2880.0 * 0.05                           # a representative per-epoch V (floor-regime) x 2880 epochs
    for K in cfg_e7["supplement"]["lcs_K_grid"]:
        etas, logw = lcs_grid(V_ref, alpha_side=0.025, K=K)
        # radius of the LCS e-process at the horizon with no evidence (V=W=0): the grid mixture penalty ~ log K
        r0 = float(np.ravel(lcs_radius(np.zeros(1), etas, logw, math.log(2 / 0.05)))[0])
        out["lcs_K"][str(K)] = {"K": K, "null_radius": r0, "n_etas": len(etas)}
    # budget mode: horizon controller can move budget across epochs; per-epoch proportional cannot. Reported
    # via the spend/target spread already measured by P1-D; here we record the design note + the per-epoch
    # minimum floor self-protection gamma*rho (Prop. 4(iv) self-protection remark).
    out["budget_mode"] = {"horizon": "single pooled lambda (DESIGN Sec. 3.5); moves effort across epochs",
                          "per_epoch": "B_e = rho Sum_{U_e} s; cannot move budget across epochs at IoT rates",
                          "self_protection_floor_min": "phi_e >= gamma*rho under a per-epoch proportional budget"}
    return out


def run_frontier_mc(cfg_e7, chash) -> dict:
    """200 MC repetitions of the realised LCB width on the headline stream across gamma (panel a band)."""
    from ctsample.design.floor import phi0_from_training
    from ctsample.design.probs import committed_floor
    from ctsample.sim.montecarlo import DesignSpec, Estimand, MCSpec, calibrated_design, run_mc, stable_id
    sid = cfg_e7["streams"]["frontier_stream"]
    if not is_done(STREAMS / sid):
        return {"status": "stream missing"}
    T = _load(sid)
    rho = float(cfg_e7.get("rho", 0.05))
    seed = cfg_e7["seeds"][0]
    out = {"stream": sid, "reps": cfg_e7.get("mc_reps", 200), "gamma": {}}
    qn = None
    qpath = MODELS / cfg_e7["streams"]["frontier_stream"].split("_")[2] / str(seed) / "Qnorm.json"
    for gamma in cfg_e7["gamma_grid"]:
        try:
            d = calibrated_design(T, "U6tau0.2", rho=rho, gamma=gamma, kind="mixture", util=f"U5_s{seed}",
                                  controller="expected", tau_mix=0.2)
            ests = [Estimand(q, q) for q in T.query_names[:6]]
            mc = MCSpec(reps=cfg_e7.get("mc_reps", 200), stream_id=T.stream_id, width_reps=cfg_e7.get("mc_reps", 200))
            res = run_mc(T, d, ests, mc, out_dir=E7 / chash / f"mc_g{gamma:g}", workers=1)
            out["gamma"][str(gamma)] = {"status": "ok", "phi0": committed_floor(d.phi0)}
        except Exception as e:  # noqa: BLE001
            out["gamma"][str(gamma)] = {"status": f"err: {type(e).__name__}: {e}"}
        check_memory()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", action="store_true", help="also run the 200-rep frontier MC (panel a band)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg_e7 = __import__("yaml").safe_load(CFG_PATH.read_text())
    exp = load_yaml(overrides={})
    chash = cfg_hash(exp)[:12]
    out_dir = E7 / chash
    SUMMARY.mkdir(parents=True, exist_ok=True)
    if is_done(out_dir) and not a.force and not a.mc:
        print(f"[e7] already done at {out_dir}", flush=True)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fr = frontier(cfg_e7)
    atomic_write_json(out_dir / "frontier.json", fr)
    tt = two_tier(cfg_e7)
    atomic_write_json(out_dir / "two_tier.json", tt)
    cap = capacity(cfg_e7)
    atomic_write_json(out_dir / "capacity.json", cap)
    sup = supplement(cfg_e7, a.mc)
    atomic_write_json(out_dir / "supplement.json", sup)
    if a.mc:
        atomic_write_json(out_dir / "frontier_mc.json", run_frontier_mc(cfg_e7, chash))
    # summaries
    import polars as pl
    if fr["rows"]:
        pl.DataFrame(fr["rows"]).write_parquet(SUMMARY / "E7_frontier.parquet")
    atomic_write_json(SUMMARY / "E7_two_tier.json", tt)
    atomic_write_json(SUMMARY / "E7_capacity.json", cap)
    atomic_write_json(SUMMARY / "E7.json", {"cfg_hash": chash, "frontier_folds": len({r["fold"] for r in fr["rows"]}),
                      "gamma_grid": cfg_e7["gamma_grid"], "generated": time.strftime("%Y-%m-%d %H:%M:%S")})
    mark_done(out_dir, {"panels": ["frontier", "two_tier", "capacity", "supplement"]})
    print(f"[e7] done in {time.time() - t0:.0f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
