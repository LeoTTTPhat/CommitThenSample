"""E4: what the floor buys against evasion (DESIGN Sec. 6.1 row E4, Sec. 6.4; Prop. 4). Analytic.

For every LP stream and policy (CtS single tier = U6tau0.2 budget-matched, CtS-size U1_bm, CtS-unif U0_bm,
B3 Time-Machine, B4 top-k, B6 alert-triggered zero-day, B8 uniform, B13 top-k+reserve):

  * realised floor phi0 per stream (committed-training value and the stream's own gamma*rho*Sum_all/Sum_pay);
  * content-evidence miss per incident (no attack) vs the bound (1-phi0)^k;
  * content-evidence miss under the Alg. 7 white-box best response and the black-box surrogate transfer
    (ctsample.adversary.mimicry); deterministic policies (B3/B4/B6) miss w.p. 1, CtS follows (1-phi0)^k;
  * two-tier capped degradation under padding (Prop. 4(ii), measured head share), m in {1,3,10};
  * real-unit detection and the deficit under flooding N in {0,10,100,1000} for the constant, per-epoch
    equalised and flood-guarded floors (Prop. 4(iv));
  * time-to-evidence quantiles of a persistent C&C channel (Prop. 4(iii)).

Outputs (atomic, resumable): results/E4/<stream>/<cfg_hash>/{config.json, incidents.parquet, metrics.json, DONE}.
Summaries: results/summary/E4_incidents.parquet, E4_streams.parquet, E4_mimicry.json, E4.json.

    .venv/bin/python memguard/memguard.py run --name p1_e4 --workers 8 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/13_e4.py [--only <substr>] [--no-mimicry] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctsample.config import cfg_hash, load_yaml  # noqa: E402
from ctsample.utils.io import RESULTS, atomic_write_json, is_done, mark_done  # noqa: E402
from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers  # noqa: E402

STREAMS = RESULTS / "streams"
MODELS = RESULTS / "models"
DESIGNS = RESULTS / "designs"
E4 = RESULTS / "E4"
SUMMARY = RESULTS / "summary"
CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "experiments" / "E4.yaml"


def _init_worker(ppid: int) -> None:
    import threading

    def watch():
        while True:
            if os.getppid() != ppid:
                os._exit(1)
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()


# --------------------------------------------------------------------------------------------- #
# per-stream list (mirrors scripts/05_designs.test_streams; fold = model directory)
# --------------------------------------------------------------------------------------------- #
def stream_list(cfg_e4) -> list[tuple[str, str, str]]:
    from ctsample.streams.recipe import lp_folds, load_config, load_inventory, p_ppm, stream_name
    cfg, inv = load_config(), load_inventory()
    ppms = cfg_e4["streams"]["prevalences_ppm"]
    out = []
    for sc in lp_folds(cfg, inv):
        fold = sc.replace("CTU-IoT-Malware-Capture-", "")
        for ppm in ppms:
            out.append((f"lp_iot23_{fold}_p{ppm}", fold, f"p{ppm}"))
    if cfg_e4["streams"].get("include_edge", True):
        for ppm in ppms:
            out.append((f"lp_edge_p{ppm}", "all", f"p{ppm}"))
    if cfg_e4["streams"].get("include_gotham", True):
        out += [("lp_gotham_p10000", "all", "p10000"), ("lp_gotham_nat", "all", "p10000")]
    return out


# --------------------------------------------------------------------------------------------- #
# analytic helpers
# --------------------------------------------------------------------------------------------- #
def _group_miss(pi, group, mask):
    """Per-incident product of (1-pi) over the content units (group id >= 0, mask = content)."""
    from ctsample.estimation.incident import group_miss
    return group_miss(np.asarray(pi, float), np.asarray(group), mask=np.asarray(mask, bool))


def _pi_payload(stream_dir_designs: Path, name: str, rho: float, n_pay: int) -> np.ndarray:
    return np.load(stream_dir_designs / f"{name}_{rho:g}.npy")


def _scatter_pay_to_content(pi_pay: np.ndarray, content_pay: np.ndarray, inc_pay: np.ndarray):
    """content-unit pi, incident ids and k for the payload units flagged content."""
    cm = content_pay
    return pi_pay[cm], inc_pay[cm]


def two_tier_curve(cfg_e4, gamma, Bbar, E, C_H, C_T):
    """(unpadded, padded, worst) retention ratios to the single tier for phi_t and m grids (Prop. 4(ii))."""
    from ctsample.design.floor import phi_cap_two_tier, two_tier_retention
    phi_t = cfg_e4["two_tier"]["phi_t"]
    C0 = C_H + C_T
    phi_star = min(1.0, gamma * Bbar * E / C0)
    out = {}
    for m in cfg_e4["two_tier"]["m_grid"]:
        unp, pad = two_tier_retention(phi_t, m, gamma, Bbar, E, C_H, C_T)
        out[str(m)] = {"unpadded_ratio": float(unp / phi_star), "padded_ratio": float(pad / phi_star),
                       "phi_cap": float(phi_cap_two_tier(phi_t, gamma, Bbar, E, C_H, C_T)), "phi_star": phi_star}
    out["head_share"] = float(C_H / C0)
    out["phi_t"] = phi_t
    return out


def head_tail_costs(T, cols_pay, H=1024):
    """Benign head cost C_H and tail cost C_T from the first-8 packet sizes of benign payload units.
    Head = the shortest prefix with cumulative payload >= min(P_i, H); head cost ~ its frame bytes
    (approximated from the per-direction frame totals in proportion to payload). C^0 = Sum_benign s."""
    pay = np.asarray(T.pay, bool)
    pay_idx = np.nonzero(pay)[0]
    lab = np.asarray(T.label)[pay_idx] if T.label is not None else np.zeros(len(pay_idx))
    ben = lab == 0
    s = cols_pay["s"].astype(np.float64)
    pb = (cols_pay["pb0"] + cols_pay["pb1"]).astype(np.float64)
    sizes = np.abs(cols_pay["sizes"].astype(np.int64))                     # (N,8) payload sizes
    # cumulative payload over the first 8 packets; head = packets until cumulative >= min(P,H)
    cum = np.cumsum(sizes, axis=1)
    target = np.minimum(pb, H)
    head_pkts = (cum < target[:, None]).sum(axis=1) + 1                    # 1-based count (>=1)
    n_pkts = (cols_pay["pk0"] + cols_pay["pk1"]).astype(np.float64)
    head_pkts = np.minimum(head_pkts, np.maximum(n_pkts, 1))
    head_frac = np.where(n_pkts > 0, head_pkts / n_pkts, 1.0)              # fraction of packets in the head
    # head cost ~ head_frac * s (frame bytes split in packet proportion) capped by s; tail = s - head
    head_cost = np.minimum(s, head_frac * s)
    C_H = float(head_cost[ben].sum())
    C0 = float(s[ben].sum())
    return C_H, C0 - C_H, C0


def flooding_metrics(cfg_e4, T, cols_pay, phi0, gamma, Bbar, content_pay, inc_pay):
    """Real-unit content-evidence detection and deficit under flooding N junk units/epoch of benign-median
    size from the attacker device in the incident epochs, for the three floor regimes (Prop. 4(iv))."""
    pay = np.asarray(T.pay, bool)
    pay_idx = np.nonzero(pay)[0]
    ep_pay = np.asarray(T.epoch)[pay_idx]
    s_pay = cols_pay["s"].astype(np.float64)
    s_all = np.asarray(T.s, float)
    E = T.n_epochs
    s_bar_N = float(np.median(s_pay)) if len(s_pay) else 0.0              # benign-median payload-unit size
    cm = content_pay
    inc_c = inc_pay[cm]
    ep_c = ep_pay[cm]
    inc_epochs = np.unique(ep_c)                                          # incident epochs (where content lives)
    # per-epoch benign payload bytes C0_e and full-capture bytes for B_e
    C0_e = np.bincount(ep_pay, weights=s_pay, minlength=E)
    S_all_e = np.bincount(np.asarray(T.epoch), weights=s_all, minlength=E)
    out = {"s_bar_N": s_bar_N, "N_grid": list(cfg_e4["flooding"]["N_grid"]), "regimes": {}}
    cf = _committed_floor(phi0)
    for regime in cfg_e4["flooding"]["floors"]:
        rows = []
        for N in cfg_e4["flooding"]["N_grid"]:
            if regime == "constant":
                phi_c = np.full(cm.sum(), cf)                             # unchanged by N
                extra_spend = N * cf * s_bar_N * len(inc_epochs)          # enters D_e over incident epochs
                deficit = extra_spend / Bbar                              # in units of Bbar
                guard = deficit > cfg_e4["flooding"]["Dmax_mult"]
            elif regime == "equalised":
                Be = cfg_e4.get("rho", 0.05) * S_all_e
                phi_e = np.clip(gamma * Be / (C0_e + N * s_bar_N + 1e-9), cf / 2.0, 1.0)
                phi_c = phi_e[ep_c]
                deficit = 0.0
                guard = False
            else:  # flood_guard: beyond D_max the content device's floor is capped (same-device flooding)
                # active only if the constant-floor deficit would exceed D_max
                defl = N * cf * s_bar_N * len(inc_epochs) / Bbar
                active = defl > cfg_e4["flooding"]["Dmax_mult"]
                if not active:
                    phi_c = np.full(cm.sum(), cf)
                else:
                    # per incident epoch: C_{d,e} = device payload bytes + N junk; floor = gamma Bbar/(n_active C)
                    # approximate n_active = distinct payload devices per epoch (>=1)
                    dev_pay = np.asarray(T.device)[pay_idx]
                    phi_list = []
                    for i in np.nonzero(cm)[0]:
                        e = ep_pay[i]
                        in_e = ep_pay == e
                        Cde = s_pay[in_e & (dev_pay == dev_pay[i])].sum() + N * s_bar_N
                        n_active = max(1, len(np.unique(dev_pay[in_e])))
                        phi_list.append(min(cf, gamma * Bbar / (n_active * max(Cde, 1.0))))
                    phi_c = np.array(phi_list)
                deficit = defl
                guard = active
            # content-evidence miss per incident under this regime
            ids, miss, k = _group_miss(_full_pi(phi_c, cm, len(pay_idx)), inc_pay, cm)
            rows.append({"N": int(N), "deficit_over_Bbar": float(deficit), "flood_guard_active": bool(guard),
                         "min_phi_content": float(phi_c.min()) if len(phi_c) else float("nan"),
                         "incident_miss": {int(i): float(mi) for i, mi in zip(ids.tolist(), miss.tolist())},
                         "mean_content_detect": float(np.mean(1.0 - miss)) if len(miss) else float("nan")})
        out["regimes"][regime] = rows
    return out


def _full_pi(pi_content, content_pay, n_pay):
    """Scatter content-unit pi back to a payload-length array (non-content units get pi=1 so they do not
    affect the content-evidence miss product, which is masked to content)."""
    full = np.ones(n_pay)
    full[np.nonzero(content_pay)[0]] = pi_content
    return full


def _committed_floor(phi0):
    from ctsample.design.probs import committed_floor
    return float(committed_floor(phi0))


def tte_quantiles(cfg_e4):
    from ctsample.estimation.incident import k_for_detection, tte_survival
    out = {}
    for phi0 in cfg_e4["tte"]["phi0_ref"]:
        q = {str(qq): k_for_detection(phi0, qq) for qq in cfg_e4["tte"]["quantiles"]}
        out[str(phi0)] = q
    return out


# --------------------------------------------------------------------------------------------- #
# per-stream driver
# --------------------------------------------------------------------------------------------- #
def run_stream(sid: str, fold: str, pkey: str, cfg_e4: dict, chash: str, do_mimicry: bool, force: bool) -> dict:
    try:
        import polars as pl
        from ctsample.sim.tables import StreamTables
        out_dir = E4 / sid / chash
        if is_done(out_dir) and not force:
            return {"stream": sid, "skipped": True}
        sd = STREAMS / sid
        dd = DESIGNS / sid
        if not is_done(sd) or not is_done(dd):
            return {"stream": sid, "skipped": True, "reason": "stream/designs missing"}
        rho = float(cfg_e4.get("rho", 0.05))
        gamma = float(cfg_e4.get("gamma", 0.3))
        seeds = cfg_e4.get("seeds", [0, 1, 2])
        T = StreamTables.load(sd)
        pay = np.asarray(T.pay, bool)
        pay_idx = np.nonzero(pay)[0]
        n_pay = len(pay_idx)
        content_pay = (np.asarray(T.content, bool) if T.content is not None else np.asarray(T.incident) >= 0)[pay_idx]
        inc_pay = (np.asarray(T.incident)[pay_idx] if T.incident is not None else np.full(n_pay, -1))
        s_all = np.asarray(T.s, float)
        E = T.n_epochs
        Bbar = rho * s_all.sum() / E
        cols_pay = dict(np.load(sd / "cols_pay.npz")) if (sd / "cols_pay.npz").exists() else None

        # realised floor
        di = json.loads((dd / f"U1_bm_{rho:g}.json").read_text())
        phi0_committed = float(di["phi0_committed"]) if di.get("phi0_committed") else None
        phi0_stream = float(di["phi0_stream"])
        phi0 = phi0_committed if phi0_committed else phi0_stream
        cf = _committed_floor(phi0)

        # ---- no-attack content-evidence miss per policy (equal budget, _bm where applicable) ----
        pol = cfg_e4["policies"]
        policy_pi = {}
        # CtS designs and uniform/size are per-unit pi arrays over payload units
        for key, name in pol.items():
            if key in ("cts_single", "b4", "b13"):
                nm = f"{name}_s{seeds[0]}" if key in ("cts_single", "b4", "b13") else name
                # cts_single is U6tau0.2_bm -> file U6tau0.2_s<k>_bm ; b4 -> B4_s<k>; b13 -> B13_s<k>
                if key == "cts_single":
                    nm = f"U6tau0.2_s{seeds[0]}_bm"
                elif key == "b4":
                    nm = f"B4_s{seeds[0]}"
                elif key == "b13":
                    nm = f"B13_s{seeds[0]}"
            else:
                nm = name
            f = dd / f"{nm}_{rho:g}.npy"
            if not f.exists():
                continue
            policy_pi[key] = np.load(f)

        incidents = {}
        inc_ids = np.unique(inc_pay[content_pay])
        inc_k = {int(i): int((inc_pay[content_pay] == i).sum()) for i in inc_ids}
        rows = []
        for key, pip in policy_pi.items():
            ids, miss, k = _group_miss(pip, inc_pay, content_pay)
            for i, mi, kk in zip(ids.tolist(), miss.tolist(), k.tolist()):
                rows.append({"stream": sid, "fold": fold, "incident": int(i), "k": int(kk), "policy": key,
                             "attack": "none", "miss": float(mi), "phi0": cf, "bound": float((1 - cf) ** kk)})

        # ---- mimicry (Alg. 7) ----
        mimicry_summary = {}
        if do_mimicry and cols_pay is not None and content_pay.any():
            mimicry_summary = run_mimicry(sid, fold, pkey, rho, seeds, cfg_e4, T, cols_pay, content_pay,
                                          inc_pay, dd, rows, cf)

        # B3 / B4 / B6 / B13 under attack (deterministic / reserve): computed in run_attack_deterministic
        run_attack_deterministic(sid, fold, rho, seeds, T, cols_pay, content_pay, inc_pay, dd, rows, cf,
                                  policy_pi)

        # ---- two-tier capped degradation (Prop. 4(ii)) ----
        two_tier = None
        if cols_pay is not None:
            C_H, C_T, C0 = head_tail_costs(T, cols_pay, cfg_e4["two_tier"]["H_bytes"])
            two_tier = two_tier_curve(cfg_e4, gamma, Bbar, E, C_H, C_T)

        # ---- flooding ----
        flooding = None
        if cols_pay is not None and content_pay.any():
            flooding = flooding_metrics(cfg_e4, T, cols_pay, phi0, gamma, Bbar, content_pay, inc_pay)

        # ---- write ----
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "config.json", {"stream": sid, "fold": fold, "rho": rho, "gamma": gamma,
                          "cfg_hash": chash, "phi0_committed": phi0_committed, "phi0_stream": phi0_stream,
                          "committed_floor": cf, "n_pay": n_pay, "E": E, "Bbar": Bbar,
                          "incident_k": inc_k, "policies": {k: pol[k] for k in pol}})
        pl.DataFrame(rows).write_parquet(out_dir / "incidents.parquet")
        metrics = {"stream": sid, "fold": fold, "rho": rho, "committed_floor": cf, "phi0_committed": phi0_committed,
                   "phi0_stream": phi0_stream, "Bbar": Bbar, "E": E, "n_content": int(content_pay.sum()),
                   "incident_k": inc_k, "two_tier": two_tier, "flooding": flooding, "mimicry": mimicry_summary,
                   "tte": tte_quantiles(cfg_e4)}
        atomic_write_json(out_dir / "metrics.json", metrics)
        mark_done(out_dir, {"n_incidents": len(inc_ids), "n_rows": len(rows)})
        check_memory()
        return {"stream": sid, "n_incidents": len(inc_ids), "n_rows": len(rows)}
    except MemoryBudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return {"stream": sid, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def run_mimicry(sid, fold, pkey, rho, seeds, cfg_e4, T, cols_pay, content_pay, inc_pay, dd, rows, cf) -> dict:
    """White-box + black-box Alg. 7 on the CtS-AI design and B4/B13; appends per-incident attack rows."""
    from ctsample.adversary import mimicry as MM
    from ctsample.design.utility import UtilityBundle
    seed = seeds[0]
    mdl = MODELS / fold / str(seed)
    if not (mdl / "DONE").exists():
        return {"status": "no model"}
    b = UtilityBundle.load(mdl, attach_lgbm=True)
    Q = json.loads((mdl / "Qnorm.json").read_text())[pkey]
    active, W, Q2 = Q["active"], Q["W"], Q["Q2"]
    dj = json.loads((dd / f"U6tau0.2_s{seed}_bm_{rho:g}.json").read_text())["spec"]
    cts = {"phi0": dj["phi0"], "lam": dj["lam"], "kind": dj["kind"], "tau_mix": dj["tau_mix"]}
    devs = np.unique(np.asarray(T.device)[np.asarray(T.content, bool)])
    ctx = MM.load_device_context(T, cols_pay, devs)
    spec = MM.MimicrySpec(pad_factors=tuple(cfg_e4["mimicry"]["pad_factors"]),
                          prefix_bytes=tuple(cfg_e4["mimicry"]["prefix_bytes"]),
                          dports=tuple(cfg_e4["mimicry"]["dports"]),
                          iat_factors=tuple(cfg_e4["mimicry"]["iat_factors"]),
                          splits=tuple(cfg_e4["mimicry"]["splits"]),
                          prefix_mss=int(cfg_e4["mimicry"]["prefix_mss"]))
    surrogate = None
    if cfg_e4["mimicry"].get("blackbox", True):
        sfold = _disjoint_fold(fold)
        sm = MODELS / sfold / str(seed)
        if (sm / "DONE").exists():
            sb = UtilityBundle.load(sm, attach_lgbm=True)
            sQ = json.loads((sm / "Qnorm.json").read_text())[pkey]
            surrogate = (sb, sQ["active"], sQ["W"], sQ["Q2"])
    br = MM.best_response(ctx, b, active, W, Q2, spec, cts_design=cts, surrogate=surrogate)
    # map content units back to incidents
    inc_c = ctx.incident[ctx.content]
    for attack, key in (("none", "cts_single"),):     # no-attack CtS already in rows; keep for reference check
        pass
    _append_grouped(rows, sid, fold, inc_c, br["det_min"], "cts_single", "mimicry_wb", cf)
    _append_grouped(rows, sid, fold, inc_c, br["det_base"], "cts_single", "mimicry_base", cf)
    if "det_at_bb" in br:
        _append_grouped(rows, sid, fold, inc_c, br["det_at_bb"], "cts_single", "mimicry_bb", cf)
    summ = {"status": "ok", "seed": seed, "n_content": int(ctx.n_content), "surrogate_fold": _disjoint_fold(fold),
            "frac_content_at_floor_wb": float(np.mean(br["det_min"] <= cf * 1.0001)),
            "split_min_unique": np.unique(br["split_min"]).tolist(),
            "det_min_pctiles": np.percentile(br["det_min"], [0, 50, 90, 100]).tolist(),
            "det_base_median": float(np.median(br["det_base"])),
            "pmal_min_median": float(np.median(br["pmal_min"]) / 65535.0),
            "pmal_base_median": float(np.median(br["pmal_base"]) / 65535.0)}
    return summ


def _disjoint_fold(fold: str) -> str:
    order = ["1-1", "3-1", "7-1", "34-1", "48-1", "49-1"]
    if fold in order:
        i = order.index(fold)
        return order[(i + 1) % len(order)]
    return "1-1"          # zero-shot streams (edge, gotham) use any disjoint IoT-23 fold as the surrogate


def _append_grouped(rows, sid, fold, inc_c, pi_c, policy, attack, cf):
    """Group a per-content-unit detection array by incident and append miss rows."""
    inc_c = np.asarray(inc_c)
    for i in np.unique(inc_c):
        m = inc_c == i
        miss = float(np.exp(np.log1p(-np.clip(pi_c[m], 0, 1)).sum()))
        rows.append({"stream": sid, "fold": fold, "incident": int(i), "k": int(m.sum()), "policy": policy,
                     "attack": attack, "miss": miss, "phi0": cf, "bound": float((1 - cf) ** int(m.sum()))})


def run_attack_deterministic(sid, fold, rho, seeds, T, cols_pay, content_pay, inc_pay, dd, rows, cf, policy_pi):
    """Under-attack misses for the deterministic / reserve policies (DESIGN E4 hypothesis):
      B3  -> a prefix >= N_TM pushes all content past the per-connection cut-off: retention 0, miss = 1;
      B4  -> mimicry drops content below the global top-k threshold: retention 0, miss = 1;
      B6  -> zero-day (attack-family alerts disabled): content retained only by the leftover uniform reserve;
      B13 -> top-k part lost, content retained only by the keyed reserve pi_reserve."""
    seed = seeds[0]
    inc_ids = np.unique(inc_pay[content_pay])
    kc = {int(i): int((inc_pay[content_pay] == i).sum()) for i in inc_ids}

    def add(policy, attack, pi_const):
        for i in inc_ids:
            k = kc[int(i)]
            miss = float((1.0 - pi_const) ** k)
            rows.append({"stream": sid, "fold": fold, "incident": int(i), "k": k, "policy": policy,
                         "attack": attack, "miss": miss, "phi0": cf, "bound": float((1 - cf) ** k)})

    # B3 and B4 miss w.p. 1 under attack (content past cut-off / below threshold)
    add("b3", "evasion", 0.0)
    add("b4", "mimicry", 0.0)
    # B6 zero-day: leftover reserve; B13: keyed reserve (from the committed design params)
    try:
        b6 = json.loads((dd / f"B6_{rho:g}.json").read_text())["params"]
        add("b6", "zero_day", float(b6.get("pi_leftover", 0.0)))
    except Exception:  # noqa: BLE001
        pass
    try:
        b13 = json.loads((dd / f"B13_s{seed}_{rho:g}.json").read_text())["params"]
        add("b13", "mimicry", float(b13.get("pi_reserve", 0.0)))
    except Exception:  # noqa: BLE001
        pass
    # B8 is mimicry-invariant (constant pi); record its (unchanged) miss under attack for completeness
    if "b8" in policy_pi:
        from ctsample.estimation.incident import group_miss
        ids, miss, k = group_miss(policy_pi["b8"], inc_pay, mask=content_pay)
        for i, mi, kk in zip(ids.tolist(), miss.tolist(), k.tolist()):
            rows.append({"stream": sid, "fold": fold, "incident": int(i), "k": int(kk), "policy": "b8",
                         "attack": "mimicry_invariant", "miss": float(mi), "phi0": cf,
                         "bound": float((1 - cf) ** int(kk))})


# --------------------------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-mimicry", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg_e4 = __import__("yaml").safe_load(CFG_PATH.read_text())
    exp = load_yaml(overrides={})
    chash = cfg_hash(exp)[:12]
    SUMMARY.mkdir(parents=True, exist_ok=True)
    todo = stream_list(cfg_e4)
    if a.only:
        todo = [t for t in todo if any(k in t[0] for k in a.only.split(","))]
    nw = max(1, min(a.workers, workers(default=a.workers)))
    t0 = time.time()
    res, failed = [], []
    ex = ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn"), initializer=_init_worker,
                             initargs=(os.getpid(),))
    try:
        pend = {ex.submit(run_stream, sid, fold, pk, cfg_e4, chash, not a.no_mimicry, a.force): sid
                for sid, fold, pk in todo}
        while pend:
            done, _ = wait(set(pend), timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                sid = pend.pop(f)
                r = f.result()
                res.append(r)
                if r.get("error"):
                    failed.append(sid)
                    print(f"[e4] FAILED {sid}: {r['error']}\n{r['traceback']}", flush=True)
                else:
                    print(f"[e4] {r} ({time.time() - t0:.0f}s)", flush=True)
            check_memory()
    except MemoryBudgetExceeded as e:
        print(f"[mem] {e}; exiting {EXIT_MEMORY}", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        return EXIT_MEMORY
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    build_summaries(chash)
    print(f"[e4] {len(res) - len(failed)} ok, {len(failed)} failed in {time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


def build_summaries(chash: str) -> None:
    """Pool per-stream outputs into results/summary/E4_*.parquet/json."""
    import polars as pl
    inc_parts, stream_rows, mim = [], [], {}
    for d in sorted(E4.glob(f"*/{chash}")):
        if not is_done(d):
            continue
        sid = d.parent.name
        try:
            inc_parts.append(pl.read_parquet(d / "incidents.parquet"))
        except Exception:  # noqa: BLE001
            pass
        m = json.loads((d / "metrics.json").read_text())
        stream_rows.append({"stream": sid, "fold": m["fold"], "rho": m["rho"],
                            "committed_floor": m["committed_floor"], "phi0_stream": m["phi0_stream"],
                            "phi0_committed": m.get("phi0_committed"), "n_content": m["n_content"],
                            "head_share": (m["two_tier"] or {}).get("head_share")})
        mim[sid] = {"mimicry": m.get("mimicry"), "two_tier": m.get("two_tier"), "flooding": m.get("flooding"),
                    "tte": m.get("tte")}
    if inc_parts:
        allinc = pl.concat(inc_parts, how="diagonal_relaxed")
        allinc.write_parquet(SUMMARY / "E4_incidents.parquet")
    if stream_rows:
        pl.DataFrame(stream_rows).write_parquet(SUMMARY / "E4_streams.parquet")
    atomic_write_json(SUMMARY / "E4_mimicry.json", mim)
    atomic_write_json(SUMMARY / "E4.json", {"cfg_hash": chash, "n_streams": len(stream_rows),
                      "generated": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    sys.exit(main())
