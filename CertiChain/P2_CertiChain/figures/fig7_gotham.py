#!/usr/bin/env python
"""Fig. 7 -- Gotham 2025 case study (E11): CRC-BH with the transfer scorer S_TON calibrated on all 150 CF-GOTHAM-T
incidents (DESIGN §5.1.3), applied to the 9 Gotham-real incidents. (a) per-incident strict FDP with exact
Clopper-Pearson intervals (conditional on the incident; no FDR claim) and STR vs the ceiling; (b) the certified
claims of the mirai-dos incident on a time line (true / false); (c) the claim record of a payload-linked certified
bridge (BF0(y) ~> ITT(y), true because no decoded Mirai report names y, GT rule §5.2.3). Owner: P2-G.

  .venv/bin/python -I P2_CertiChain/figures/fig7_gotham.py
Also writes results/exp/E11/gotham_v2/claim_record_example.json and certified_claims.parquet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from certichain.certify.crc import certificate  # noqa: E402
from certichain.eval import experiments as X  # noqa: E402
from certichain.io.paths import atomic_write_json, atomic_write_parquet  # noqa: E402
from certichain.scoring import driver as D  # noqa: E402

CAL = ("cf_gotham_t_v2", "S_TON.stat")       # v2 bank (post-PREREG deviation P1); v1 gives identical claims
GK = "gotham_v2"
TEST = ("gotham_real", "S_TON.stat")
SHORT = {"mirai_infection_17_15": "victim .17.15", "mirai_infection_18_10": "victim .18.10",
         "mirai_infection_18_17": "victim .18.17", "mirai_infection_18_18": "victim .18.18",
         "mirai_infection_20_19": "victim .20.19", "mirai_dos_17_15": "mirai-dos .17.15",
         "merlin_17_15": "Merlin .17.15", "benign_18_17_mirai_dos": "benign .18.17", "benign_20_19_merlin": "benign .20.19"}


def certified_claims(alpha: float) -> tuple[pl.DataFrame, int, dict]:
    cb, tb = X.load_bank(*CAL), X.load_bank(*TEST)
    col = X.crc_full_bank(cb, alpha)
    cert = certificate(cb.t("Lmono", "bh"), alpha, col, cb.grids["bh"], scorer_id=CAL[1])
    rows = []
    for k, iid in enumerate(tb.index["incident_id"].to_list()):
        R = int(tb.t("R", "bh")[k, col])
        if R == 0:
            continue
        inc = X.inc_of(iid)
        sl = tb.slice(k)
        cands = D.load_cands(TEST[0], inc)
        units = D.load_units(TEST[0], inc)
        gtu = D.load_gt_units(TEST[0], inc)
        ep = dict(zip(gtu["unit_id"].to_list(), gtu["episode"].to_list()))
        cu = dict(zip(cands["cand_id"].to_list(), zip(cands["u"].to_list(), cands["v"].to_list())))
        uinfo = {r["unit_id"]: r for r in units.select("unit_id", "ts", "te", "src", "dsts", "type", "rec_ids").iter_rows(named=True)}
        lab = D.load_labels(TEST[0], inc)
        br = dict(zip(lab["cand_id"].to_list(), lab["bridge"].fill_null(False).to_list()))
        meta = json.loads((X._incident_dir(TEST[0], inc) / "meta.json").read_text())
        for rank, (c, y, p, s) in enumerate(zip(tb.csr["cand"][sl][:R], tb.csr["y"][sl][:R], tb.csr["p"][sl][:R],
                                                tb.csr["score"][sl][:R])):
            u, v = cu[int(c)]
            rows.append({"incident": inc, "rank": rank + 1, "cand_id": int(c), "y": int(y), "bridge": bool(br.get(int(c))),
                         "p": float(p), "score": float(s), "u": int(u), "v": int(v),
                         "u_type": uinfo[u]["type"], "v_type": uinfo[v]["type"], "u_src": uinfo[u]["src"],
                         "v_src": uinfo[v]["src"], "u_ts": uinfo[u]["ts"] - meta["t0"], "v_ts": uinfo[v]["ts"] - meta["t0"],
                         "u_ep": ep.get(u), "v_ep": ep.get(v)})
    return pl.DataFrame(rows, infer_schema_length=None), col, cert


def claim_record(cl: pl.DataFrame, cert: dict, alpha: float) -> dict:
    """Claim record of the first certified payload-linked bridge (BF0 ~> ITT) in a mirai-infection incident."""
    cand = cl.filter((pl.col("y") == 1) & pl.col("bridge") & pl.col("incident").str.contains("mirai_infection"))
    if not cand.height:
        return {}
    r = cand.row(0, named=True)
    inc = r["incident"]
    units = D.load_units(TEST[0], inc)
    recs = pl.read_parquet(X._incident_dir(TEST[0], inc) / "records.parquet", columns=["rec_id", "src_file", "src_locator", "ts"])
    rmap = {a: (b, c) for a, b, c in zip(recs["rec_id"].to_list(), recs["src_file"].to_list(), recs["src_locator"].to_list())}

    def ev(uid):
        rids = units.filter(pl.col("unit_id") == uid)["rec_ids"][0].to_list()[:2]
        return [f"{Path(rmap[x][0]).stem.replace('iotsim-', '')} {rmap[x][1].replace('frames:', 'fr ')}" for x in rids if x in rmap]
    return {"incident": inc, "claim": f"unit {r['u']} -> unit {r['v']} (candidate {r['cand_id']})",
            "u": {"type": r["u_type"], "src": r["u_src"], "t_minus_t0_s": round(r["u_ts"], 1), "evidence": ev(r["u"])},
            "v": {"type": r["v_type"], "src": r["v_src"], "t_minus_t0_s": round(r["v_ts"], 1), "evidence": ev(r["v"])},
            "score": round(r["score"], 3), "p_ref": round(r["p"], 4), "rank_in_incident": r["rank"],
            "gt": f"{r['u_ep']} ~> {r['v_ep']} (bridge over hidden REP0; payload-derived)",
            "certificate": {k: cert[k] for k in ("alpha", "lambda_hat", "lambda_hat_index", "K", "calibration_risk_at_lambda_hat",
                                                 "calibration_table_blake3", "scorer_id", "rule", "loss")}}


def main() -> None:
    S.setup()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    out = X.exp_root("E11") / GK
    alpha = 0.2
    cl, col, cert = certified_claims(alpha)
    atomic_write_parquet(cl, out / "certified_claims.parquet")
    rec = claim_record(cl, cert, alpha)
    atomic_write_json(out / "claim_record_example.json", rec)
    per = pl.read_parquet(out / "per_incident.parquet").filter((pl.col("alpha") - alpha).abs() < 1e-9)
    order = ["mirai_infection_17_15", "mirai_infection_18_10", "mirai_infection_18_17", "mirai_infection_18_18",
             "mirai_infection_20_19", "mirai_dos_17_15", "merlin_17_15", "benign_18_17_mirai_dos", "benign_20_19_merlin"]
    per = per.with_columns(pl.col("incident").str.replace("gotham_real/", "").alias("inc"))
    pm = {r["inc"]: r for r in per.iter_rows(named=True)}
    fig = plt.figure(figsize=(S.DCOL, 3.45))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.05, 0.95, 1.15], height_ratios=[1.25, 1], wspace=0.55, hspace=0.75)
    # (a) per-incident FDP with exact CIs
    ax = fig.add_subplot(gs[0, 0])
    for i, inc in enumerate(order):
        r = pm.get(inc)
        if r is None:
            continue
        if r["R"] == 0:
            ax.plot([i], [0], marker="o", mfc="white", mec=S.INK2, ms=3.5)
            ax.text(i, -0.2, "0", ha="center", fontsize=7, color=S.INK2)
            continue
        col_ = S.BLUE
        ax.errorbar([i], [r["fdp"]], yerr=[[r["fdp"] - r["fdp_cp_lo"]], [r["fdp_cp_hi"] - r["fdp"]]], fmt="o", ms=3.5,
                    color=col_, elinewidth=0.9, capsize=1.5)
        ax.text(i, -0.2, f"{r['R']}", ha="center", fontsize=7, color=S.INK)
    # exploratory in-domain comparison: S_GOT2 scorer with the CF-GOTHAM-T v2 S_GOT2 bank (open squares)
    pd2 = X.exp_root("E11") / "gotham_v2_sgot2" / "per_incident.parquet"
    sgot2_counts: list[str] = []
    if pd2.exists():
        per2 = pl.read_parquet(pd2).filter((pl.col("alpha") - alpha).abs() < 1e-9).with_columns(
            pl.col("incident").str.replace("gotham_real/", "").alias("inc"))
        pm2 = {r["inc"]: r for r in per2.iter_rows(named=True)}
        for i, inc in enumerate(order):
            r = pm2.get(inc)
            if r is None or r["R"] == 0:
                continue
            ax.errorbar([i + 0.28], [r["fdp"]], yerr=[[r["fdp"] - r["fdp_cp_lo"]], [r["fdp_cp_hi"] - r["fdp"]]],
                        fmt="s", ms=3.2, color=S.ORANGE, mfc="white", elinewidth=0.8, capsize=1.2)
            sgot2_counts.append(f"{SHORT[inc]} {r['fdp']:.2f} ({r['R']} claims)")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.BLUE, ms=3.5),
                           Line2D([], [], marker="s", ls="", color=S.ORANGE, mfc="white", ms=3.2)],
                  labels=["S$_{TON}$ (primary)", "S$_{GOT2}$ (in-domain)"], loc="center left",
                  bbox_to_anchor=(0.0, 0.53), fontsize=7, handletextpad=0.2, handlelength=1.0, borderpad=0.15,
                  labelspacing=0.25, frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    S.alpha_line(ax, alpha, color=S.ALPHA_COLORS[alpha])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([])
    ax.set_ylim(-0.3, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("per-incident FDP")
    ax.set_title(f"(a) per-incident FDP, exact 95% CI\n($\\alpha$={alpha}; numbers = #claims, S$_{{TON}}$)",
                 loc="left", fontsize=7)
    ax.grid(axis="x", visible=False)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    w = 0.38
    for i, inc in enumerate(order):
        r = pm.get(inc)
        if r is None or "benign" in inc:
            continue
        ax2.bar(i - w / 2, r["str"] or 0, width=w, color=S.BLUE)
        ax2.bar(i + w / 2, r["ceiling"] or 0, width=w, color="white", edgecolor=S.INK2, lw=0.8)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([SHORT[o] for o in order], rotation=60, ha="right", fontsize=7)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("STR")
    ax2.set_title("STR of S$_{TON}$ (filled) vs ceiling (open)", loc="left", fontsize=7)
    ax2.grid(axis="x", visible=False)
    # (b) mirai-dos timeline of certified claims
    ax3 = fig.add_subplot(gs[:, 1])
    sub = cl.filter(pl.col("incident").str.contains("mirai_dos"))
    if sub.height:
        uids = sorted(set(sub["u"].to_list()) | set(sub["v"].to_list()),
                      key=lambda x: (sub.filter((pl.col("u") == x))["u_ts"].to_list() or sub.filter(pl.col("v") == x)["v_ts"].to_list())[0])
        ypos = {u: i for i, u in enumerate(uids)}
        tmap = {}
        for r in sub.iter_rows(named=True):
            tmap[r["u"]] = (r["u_ts"] / 60, r["u_type"])
            tmap[r["v"]] = (r["v_ts"] / 60, r["v_type"])
        for r in sorted(sub.iter_rows(named=True), key=lambda r: r["y"]):
            xu, xv = r["u_ts"] / 60, r["v_ts"] / 60
            colr = S.BLUE if r["y"] == 1 else S.GRID
            ax3.annotate("", xy=(xv, ypos[r["v"]]), xytext=(xu, ypos[r["u"]]),
                         arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.1 if r["y"] == 1 else 0.6,
                                         shrinkA=2, shrinkB=2, mutation_scale=6), zorder=3 if r["y"] == 1 else 2)
            if r["y"] == 1:
                ax3.text((xu + xv) / 2 - 0.6, (ypos[r["u"]] + ypos[r["v"]]) / 2, f"p={r['p']:.3f}", fontsize=7,
                         color=S.BLUE, ha="right", va="center", zorder=6,
                         bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
        for u, (t, ty) in tmap.items():
            ax3.plot([t], [ypos[u]], "o", ms=2.8, color=S.INK2, zorder=4)
        ax3.set_yticks(range(len(uids)))
        abbrev = lambda t: t.replace("BENIGN_GENERIC_", "benign ").replace("_PERIODIC", "").replace("_ACTIVITY", "").replace("_", " ").lower()  # noqa: E731
        ax3.set_yticklabels([abbrev(tmap[u][1]) for u in uids], fontsize=7)
        ax3.set_xlabel("start time since $t_0$ (min)")
        nt, nf = int((sub["y"] == 1).sum()), int((sub["y"] != 1).sum())
        ax3.set_title(f"(b) mirai-dos: {nt} true, {nf} false claims", loc="left", fontsize=7)
        ax3.grid(axis="y", visible=False)
    # (c) claim record excerpt
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.axis("off")
    if rec:
        c = rec["certificate"]
        txt = (f"claim record (excerpt)\n"
               f"incident  {rec['incident'].replace('gotham_real/', '')}\n"
               f"u  {rec['u']['type'].lower()}\n   src {rec['u']['src']}, t+{rec['u']['t_minus_t0_s']:.0f}s\n"
               f"   ev {rec['u']['evidence'][0] if rec['u']['evidence'] else '-'}\n"
               f"v  {rec['v']['type'].lower()}\n   src {rec['v']['src']}, t+{rec['v']['t_minus_t0_s']:.0f}s\n"
               f"   ev {rec['v']['evidence'][0] if rec['v']['evidence'] else '-'}\n"
               f"score {rec['score']}, p_ref {rec['p_ref']}\n"
               f"GT  BF0 ~> ITT (hidden REP0;\n     payload-derived bridge)\n"
               f"certificate\n  alpha {c['alpha']}, K {c['K']}\n  lambda^ {c['lambda_hat']:.3f} (idx {c['lambda_hat_index']})\n"
               f"  cal. risk {c['calibration_risk_at_lambda_hat']:.3f}\n  table b3 {c['calibration_table_blake3'][:16]}...\n"
               f"  scorer {c['scorer_id']}")
        ax4.text(-0.05, 1.0, txt, family="monospace", fontsize=7, va="top", ha="left", transform=ax4.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f4", ec=S.GRID, lw=0.6))
        ax4.set_title("(c) payload-linked bridge", loc="left", fontsize=7)
    n11 = S.numbers("E11")
    g = lambda k, d=float("nan"): n11.get(k, {}).get("value", d)  # noqa: E731
    cb = X.load_bank(*CAL)
    Rc, Vc = cb.t("R", "bh")[:, col], cb.t("V", "bh")[:, col]
    cal_fdp = float(np.where(Rc > 0, Vc / np.maximum(Rc, 1), 0.0).mean())
    smry = n11.get(f"{GK}|summary|a={alpha}", {})
    s2 = n11.get(f"{GK}_sgot2|summary|a={alpha}", {})
    s2a = n11.get(f"{GK}_sgot2|summary|a=0.1", {})
    v1 = n11.get(f"gotham|summary|a={alpha}", {})
    cap = (f"Fig. 7 (Gotham 2025 case study, E11). Required facts (results/exp/E11/{GK}/*, {GK}_sgot2/*, "
           f"results/summary/E11_numbers.json): scorer S_TON (CF-TON-trained, never saw Gotham), CRC-BH calibrated on all "
           f"K = {cb.n} CF-GOTHAM-T v2 incidents (training-pool episodes and backgrounds; regenerated bank of the "
           f"post-PREREG deviation P1), lambda-hat index {col} (lambda = {cert['lambda_hat']:.3f}) at alpha = {alpha} and "
           f"also at alpha = 0.1 (same index; the original v1 bank gives the same index and exactly the same Gotham claims, "
           f"v1: {v1.get('cal_incidents_with_claims')} calibration incidents with claims); at lambda-hat only "
           f"{smry.get('cal_incidents_with_claims')} of the {cb.n} calibration incidents carry claims and "
           f"{int(Vc.sum())} of their {int(Rc.sum())} claims are false (calibration mean FDP {cal_fdp:.3f}, calibration "
           f"risk bound {cert['calibration_risk_at_lambda_hat']:.3f}). On the 9 Gotham-real incidents: (a) per-incident "
           f"strict FDP with exact Clopper-Pearson 95% intervals (conditional on the incident; no FDR claim): "
           f"{smry.get('value')} of 7 attack incidents receive claims with FDP {smry.get('fdp_min', float('nan')):.2f}-"
           f"{smry.get('fdp_max', float('nan')):.2f}, 3 attack incidents and both benign windows receive none (no wrongful "
           f"implication); STR vs ceiling (1 - rho_T) r_C,T under the seed-priority GT (rho_T pooled "
           f"{g(GK + '|gt|seed_priority|rho_T'):.3f}; default rule {g(GK + '|gt|default|rho_T'):.3f}). (b) mirai-dos: CRC "
           f"certifies all three C&C -> flood transitions (STR = 1) among 29 claims. (c) the claim record of a payload-linked "
           f"certified bridge (BF0(y) ~> ITT(y), true under the seed-priority GT because no decoded Mirai report names y; "
           f"under the default unit-episode rule the same claim is AMB, and the FDP of victims .18.17 / .18.18 becomes "
           f"1.00). S_TON's within-incident AUC is {g(GK + '|within_auc|gotham_real|S_TON.stat'):.2f} on Gotham-real "
           f"({g(GK + '|within_auc|cf_gotham_t_v2|S_TON.stat'):.2f} on CF-GOTHAM-T v2), the bank/case-study two-sample AUC "
           f"on phi is {g(GK + '|two_sample_auc|' + CAL[0] + '->gotham_real'):.2f} (same value for the v1 bank), and 59-65% of the attack windows' raw records also occur in the bank. Open squares "
           f"(exploratory, not pre-registered): the in-domain scorer S_GOT2 (CF-GOTHAM-T v2 bank, lambda-hat index "
           f"{s2.get('lam_idx')}; within-incident AUC {g(GK + '_sgot2|within_auc|gotham_real|S_GOT2.stat'):.2f} on "
           f"Gotham-real, {g(GK + '_sgot2|within_auc|cf_gotham_t_v2|S_GOT2.stat'):.3f} on the bank; claims in "
           f"{s2.get('cal_incidents_with_claims')} of {cb.n} calibration incidents) gives FDP: " + "; ".join(sgot2_counts) +
           f" at alpha = {alpha} (at alpha = 0.1, index {s2a.get('lam_idx')}: FDP {s2a.get('fdp_min', float('nan')):.2f}-"
           f"{s2a.get('fdp_max', float('nan')):.2f}, the two zero-FDP victims with a single true claim each). A better "
           f"scorer does not rescue the case study: the bank is not exchangeable with the Gotham windows, so the case study "
           f"shows the cost of a non-exchangeable bank (and of a transfer scorer), not a validity claim.")
    S.save(fig, "fig7_gotham", cap)
    print("fig7 written")


if __name__ == "__main__":
    main()
