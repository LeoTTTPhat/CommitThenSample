"""ChainForge audit (DESIGN §5.4.2 held-out check, §5.4.4 controls, E15 inputs, Table III).

GT-SIDE (reads quarantined labels / GT). Run after scripts/07 (P2-B) has produced units,
candidates and GT for a variant:

    .venv/bin/python -I -m certichain.chainforge.audit --variant cf_ton [--variant cf_ton_h ...]

Per variant → results/interim/<variant>/audit/:
    summary.json            (d) basic stats per split: records/incident, campaigns/decoys, |T*|,
                            hidden fraction, N, units, r_C,T, ρ_T; observability diagnostics
                            (enabling unit on the role pair, MIXED units, D2/D3 in scope)
    transition_pairs.parquet   (a) one row per T* transition: template edge, unit types of
                            (enab, first), in C, relation mask of the enabling pair
    reuse.parquet           (b) incidents per library episode, per type and pool (+ background reuse)
    oracle_inputs/<split>.parquet  (c) candidate table for the rule-only oracles B1/B2 of E15:
                            unit types/svc/dir/src/seed flags of u and v, relations, lags, y, bridge,
                            and the GT kind of u/v (quarantined)
Cross-variant: results/interim/cf_ton_h/audit/heldout.json — the unit-type pair sets of the
held-out transitions (CF-TON-H caltest) vs the positives of CF-TON-H train.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from certichain.io.paths import atomic_write_json, atomic_write_parquet, interim, log
from certichain.io.schema import REL_NAMES

HELD_OUT_EDGES = {("T-WEBI", "scanning", "injection"), ("T-WEBI", "injection", "backdoor"),
                  ("T-WEBI", "backdoor", "ransomware"), ("T-WEBX", "scanning", "xss"), ("T-WEBX", "xss", "backdoor")}


def _rel_names(mask: int | None) -> str:
    if mask is None:
        return ""
    return "+".join(n for b, n in sorted(REL_NAMES.items()) if int(mask) & b)


def incident_dirs(variant: str) -> list[Path]:
    return sorted(p.parent for p in interim(variant, "incidents").glob("*/*/meta.json"))


def _p(variant: str, sub: str, inc_id: str, ext: str) -> Path:
    return interim(variant, sub, f"{inc_id}.{ext}")


def load_incident(variant: str, d: Path) -> dict | None:
    meta = json.loads((d / "meta.json").read_text())
    inc = meta["inc_id"]
    need = {k: _p(variant, k, inc, "parquet") for k in ("units", "candidates", "labels_pairs", "gt_units",
                                                         "transitions", "units_side")}
    if not all(p.exists() for p in need.values()) or not _p(variant, "stats", inc, "json").exists():
        return None
    out = {k: pl.read_parquet(p) for k, p in need.items()}
    out["stats"] = json.loads(_p(variant, "stats", inc, "json").read_text())
    out["meta"] = meta
    out["gt"] = json.loads((d / "gt.json").read_text())
    out["gen"] = json.loads((d / "gen.json").read_text())
    out["dir"] = d
    return out


# --------------------------------------------------------------------------- per incident
def transition_rows(inc: dict) -> list[dict]:
    units, tt, gt, gen = inc["units"], inc["transitions"], inc["gt"], inc["gen"]
    cands = inc["candidates"]
    typ = dict(zip(units["unit_id"].to_list(), units["type"].to_list()))
    src = dict(zip(units["unit_id"].to_list(), units["src"].to_list()))
    dsts = dict(zip(units["unit_id"].to_list(), units["dsts"].to_list()))
    rel = {(int(u), int(v)): int(m) for u, v, m in cands.select("u", "v", "rel_mask").iter_rows()}
    roles = {c["cid"]: c["roles"] for c in gen["campaigns"] + gen["decoys"]}
    camps = gt["campaigns"]
    eps = gt["episodes"]
    rows = []
    for r in tt.iter_rows(named=True):
        a, b = r["src_ep"], r["dst_ep"]
        cid = eps[a]["campaign_id"]
        role_ips = set(roles.get(cid, {}).values())
        eu, fv = r["enab_u"], r["first_v"]
        on_role = None
        if eu is not None and fv is not None:
            on_role = bool(src[eu] in role_ips and set(dsts[eu]) & role_ips
                           and src[fv] in role_ips and set(dsts[fv]) & role_ips)
        rows.append({
            "inc_id": inc["meta"]["inc_id"], "split_role": inc["meta"]["split_role"], "cid": cid,
            "kind": camps[cid]["kind"], "template": camps[cid]["template"],
            "node_a": eps[a]["template_node"], "node_b": eps[b]["template_node"],
            "type_a": eps[a]["type"], "type_b": eps[b]["type"], "t_kind": r["kind"],
            "enab_u": eu, "first_v": fv, "in_C": bool(r["in_C"]),
            "utype_enab": typ.get(eu) if eu is not None else None, "utype_first": typ.get(fv) if fv is not None else None,
            "rel": _rel_names(rel.get((eu, fv))) if eu is not None and fv is not None else "",
            "enab_on_role_pair": on_role})
    return rows


def unit_kind_table(inc: dict) -> pl.DataFrame:
    """unit_id → GT kind (main | D1 | D2 | D3 | bg | mixed)."""
    gu, camps = inc["gt_units"], inc["gt"]["campaigns"]
    kinds = {c: d["kind"] for c, d in camps.items()}
    g = gu.with_columns(pl.when(pl.col("mixed")).then(pl.lit("mixed"))
                        .when(pl.col("episode").is_null()).then(pl.lit("bg"))
                        .otherwise(pl.col("campaign").replace_strict(kinds, default="bg")).alias("gt_kind"))
    allu = inc["units"].select("unit_id")
    return allu.join(g.select("unit_id", "gt_kind", pl.col("campaign").alias("gt_campaign")), on="unit_id",
                     how="left").with_columns(pl.col("gt_kind").fill_null("bg"))


def oracle_rows(inc: dict) -> pl.DataFrame:
    u = inc["units"].select("unit_id", "type", "src", "svc_class", "dir_class", "n_dst", "fanout", "ts", "te")
    side = inc["units_side"].select("unit_id", "touches_seed", "src_is_seed")
    k = unit_kind_table(inc)
    ut = u.join(side, on="unit_id", how="left").join(k, on="unit_id", how="left")
    c = inc["candidates"].join(inc["labels_pairs"], on="cand_id", how="left")
    cu = ut.rename({x: f"{x}_u" for x in ut.columns if x != "unit_id"}).rename({"unit_id": "u"})
    cv = ut.rename({x: f"{x}_v" for x in ut.columns if x != "unit_id"}).rename({"unit_id": "v"})
    out = c.join(cu, on="u", how="left").join(cv, on="v", how="left")
    return out.with_columns(pl.lit(inc["meta"]["inc_id"]).alias("inc_id")).drop("ts_u", "te_u", "ts_v", "te_v")


def diag_counts(inc: dict) -> dict:
    gt, gu = inc["gt"], inc["gt_units"]
    camps = gt["campaigns"]
    obs = set(gu.filter(pl.col("episode").is_not_null())["episode"].to_list())
    d = Counter()
    for eid, e in gt["episodes"].items():
        if e["hidden"]:
            continue
        kind = camps[e["campaign_id"]]["kind"]
        d[f"episodes_{kind}"] += 1
        d[f"episodes_{kind}_with_unit"] += int(eid in obs)
    d["mixed_units"] = int(gu["mixed"].sum())
    mixed_main = gu.filter(pl.col("mixed") & pl.col("campaign").is_not_null())
    d["mixed_units_with_campaign"] = mixed_main.height
    return dict(d)


# --------------------------------------------------------------------------- variant audit
def audit_variant(variant: str, write: bool = True) -> dict:
    dirs = incident_dirs(variant)
    trans, oracle, diags, stats, missing = [], defaultdict(list), [], [], []
    for d in dirs:
        inc = load_incident(variant, d)
        if inc is None:
            missing.append(str(d))
            continue
        trans += transition_rows(inc)
        oracle[inc["meta"]["split_role"]].append(oracle_rows(inc))
        diags.append({"split_role": inc["meta"]["split_role"], **diag_counts(inc)})
        st = inc["stats"]
        ts = st["transition_stats"]
        summ = inc["gen"].get("summary", {})
        stats.append({"inc_id": inc["meta"]["inc_id"], "split_role": inc["meta"]["split_role"],
                      "n_records": st["n_records"], "n_units": st["n_units"], "N": st["N"],
                      "U_max_hit": st["U_max_hit"], "N_max_hit": st["N_max_hit"],
                      "n_T": ts["n_T"], "n_T_obs": ts["n_T_obs"], "n_T_hid": ts["n_T_hid"],
                      "n_T_obs_in_C": ts["n_T_obs_in_C"], "n_pos": ts.get("n_pos", 0), "n_amb": ts.get("n_amb", 0),
                      "n_null": ts.get("n_null", 0), "n_campaigns": summ.get("n_campaigns"),
                      "n_decoys": summ.get("n_decoys"), "n_D1": summ.get("n_D1"), "n_D2": summ.get("n_D2"),
                      "n_D3": summ.get("n_D3"), "n_hidden": summ.get("n_hidden"), "L_s": summ.get("L_s"),
                      "d2_redrawn": summ.get("log", {}).get("d2_redrawn", 0)})
    T = pl.DataFrame(trans, infer_schema_length=None) if trans else pl.DataFrame()
    S = pl.DataFrame(stats, infer_schema_length=None) if stats else pl.DataFrame()
    D = pl.DataFrame(diags, infer_schema_length=None).fill_null(0) if diags else pl.DataFrame()
    out = {"variant": variant, "n_incidents": len(dirs), "n_audited": S.height, "n_missing_units": len(missing),
           "per_split": {}}
    for sp in (sorted(S["split_role"].unique().to_list()) if S.height else []):
        s = S.filter(pl.col("split_role") == sp)
        t = T.filter(pl.col("split_role") == sp) if T.height else T
        obs = t.filter(pl.col("t_kind") == "obs") if t.height else t
        dd = D.filter(pl.col("split_role") == sp).drop("split_role").sum().row(0, named=True) if D.height else {}
        q = lambda col: {str(x): float(s[col].quantile(x)) for x in (0.0, 0.5, 0.9, 1.0)}  # noqa: E731
        n_obs = obs.height
        res = {
            "n": s.height, "records_per_incident": q("n_records"), "units_per_incident": q("n_units"), "N": q("N"),
            "campaigns_per_incident": float(s["n_campaigns"].mean()), "decoys_per_incident": float(s["n_decoys"].mean()),
            "decoys": {k: int(s[k].sum()) for k in ("n_D1", "n_D2", "n_D3")}, "d2_redrawn": int(s["d2_redrawn"].sum()),
            "T_star": int(s["n_T"].sum()), "T_star_per_incident": float(s["n_T"].mean()),
            "T_obs": int(s["n_T_obs"].sum()), "T_hid": int(s["n_T_hid"].sum()),
            "rho_T": float(s["n_T_hid"].sum() / max(1, s["n_T"].sum())),
            "r_CT": float(s["n_T_obs_in_C"].sum() / max(1, s["n_T_obs"].sum())),
            "hidden_fraction_episodes": float(s["n_hidden"].sum()),
            "positives": int(s["n_pos"].sum()), "amb": int(s["n_amb"].sum()), "nulls": int(s["n_null"].sum()),
            "N_over_r": float(s["N"].sum() / max(1, s["n_pos"].sum())),
            "U_max_hit": int(s["U_max_hit"].sum()), "N_max_hit": int(s["N_max_hit"].sum()),
            "enab_null": int(obs["enab_u"].is_null().sum()) if n_obs else 0,
            "enab_on_role_pair": float(obs["enab_on_role_pair"].drop_nulls().mean()) if n_obs else None,
            "r_CT_if_enab_on_role_pair": (float(obs.filter(pl.col("enab_on_role_pair") == True)["in_C"].mean())  # noqa: E712
                                          if n_obs and obs.filter(pl.col("enab_on_role_pair") == True).height else None),  # noqa: E712
            "episodes_in_scope": {k[len("episodes_"):-len("_with_unit")]: f"{dd[k]}/{dd[k[:-len('_with_unit')]]}"
                                  for k in sorted(dd) if k.endswith("_with_unit")},
            "mixed_units": int(dd.get("mixed_units", 0)),
            "mixed_units_with_campaign": int(dd.get("mixed_units_with_campaign", 0)),
        }
        if n_obs:
            per_edge = (obs.group_by("kind", "template", "node_a", "node_b")
                        .agg(pl.len().alias("n"), pl.col("in_C").mean().alias("r_CT"),
                             pl.col("enab_u").is_null().sum().alias("enab_null"),
                             pl.col("enab_on_role_pair").mean().alias("enab_on_role"))
                        .sort("kind", "template", "node_a", "node_b"))
            res["per_edge"] = per_edge.to_dicts()
        # hidden-episode fraction (declared hidden / all episodes)
        out["per_split"][sp] = res
    if write:
        ad = interim(variant, "audit")
        ad.mkdir(parents=True, exist_ok=True)
        if T.height:
            atomic_write_parquet(T, ad / "transition_pairs.parquet")
        if S.height:
            atomic_write_parquet(S, ad / "incident_stats.parquet")
        for sp, parts in oracle.items():
            atomic_write_parquet(pl.concat(parts, how="diagonal_relaxed"), ad / "oracle_inputs" / f"{sp}.parquet")
        atomic_write_json(ad / "summary.json", out)
    return out


# --------------------------------------------------------------------------- held-out (a)
def heldout_check(train_variant: str = "cf_ton_h", train_split: str = "train", test_variant: str = "cf_ton_h",
                  test_split: str = "caltest", write: bool = True, extra_train: tuple[str, ...] = ()) -> dict:
    """Unit-type pairs (type(enab), type(first)) of held-out transitions vs train positives.

    A held-out template edge is 'unseen at the unit-type level' when none of its realised
    unit-type pairs occurs among the realised pairs of the train split's transitions."""
    tp = pl.read_parquet(interim(train_variant, "audit", "transition_pairs.parquet"))
    tr = tp.filter((pl.col("split_role") == train_split) & (pl.col("t_kind") == "obs") & pl.col("enab_u").is_not_null())
    for v in extra_train:
        p = interim(v, "audit", "transition_pairs.parquet")
        if p.exists():
            x = pl.read_parquet(p)
            tr = pl.concat([tr, x.filter((pl.col("split_role") == "train") & (pl.col("t_kind") == "obs")
                                         & pl.col("enab_u").is_not_null())], how="diagonal_relaxed")
    te_all = pl.read_parquet(interim(test_variant, "audit", "transition_pairs.parquet"))
    te = te_all.filter((pl.col("split_role") == test_split) & (pl.col("t_kind") == "obs") & (pl.col("kind") == "main"))
    seen_pairs = set(zip(tr["utype_enab"].to_list(), tr["utype_first"].to_list()))
    seen_edges = set(zip(tr["type_a"].to_list(), tr["type_b"].to_list()))
    out = {"train": f"{train_variant}/{train_split}" + ("+" + "+".join(extra_train) if extra_train else ""),
           "test": f"{test_variant}/{test_split}", "n_train_transitions": tr.height,
           "train_unit_type_pairs": sorted([f"{a}->{b}" for a, b in seen_pairs if a and b]), "edges": []}
    for (tid, a, b) in sorted(HELD_OUT_EDGES):
        e = te.filter((pl.col("template") == tid) & (pl.col("node_a") == a) & (pl.col("node_b") == b))
        ok = e.filter(pl.col("enab_u").is_not_null())
        pairs = Counter(zip(ok["utype_enab"].to_list(), ok["utype_first"].to_list()))
        n_unseen = sum(n for p, n in pairs.items() if p not in seen_pairs)
        out["edges"].append({
            "template": tid, "edge": f"{a}->{b}", "episode_type_pair_in_train": (a, b) in seen_edges,
            "n": e.height, "n_with_enab": ok.height,
            "unit_type_pairs": {f"{p[0]}->{p[1]}": n for p, n in pairs.most_common()},
            "share_unit_type_pair_unseen_in_train": (n_unseen / ok.height) if ok.height else None,
            "unseen_at_unit_type_level": bool(ok.height and n_unseen == ok.height)})
    out["n_edges_unseen"] = sum(1 for e in out["edges"] if e["unseen_at_unit_type_level"])
    if write:
        atomic_write_json(interim(test_variant, "audit", "heldout.json"), out)
    return out


# --------------------------------------------------------------------------- reuse (b)
def reuse_table(variants: tuple[str, ...], ds: str = "ton") -> pl.DataFrame:
    """Incidents per library episode, per (type, pool), pooled over the given variants."""
    from certichain.chainforge.episodes import EpisodeLibrary

    lib = EpisodeLibrary.load(ds).table.select("lib_eid", "type", "pool", "elig_victim", "elig_actor")
    uses = []
    for v in variants:
        p = interim(v, "index.parquet")
        if not p.exists():
            continue
        idx = pl.read_parquet(p).select("inc_id", "lib_eids").explode("lib_eids").drop_nulls()
        uses.append(idx.with_columns(pl.lit(v).alias("variant")))
    if not uses:
        return pl.DataFrame()
    u = pl.concat(uses).rename({"lib_eids": "lib_eid"})
    cnt = u.group_by("lib_eid").agg(pl.len().alias("n_incidents"), pl.col("variant").unique().alias("variants"))
    t = lib.join(cnt, on="lib_eid", how="left").with_columns(pl.col("n_incidents").fill_null(0))
    return (t.group_by("type", "pool").agg(
        pl.len().alias("n_library"), (pl.col("n_incidents") > 0).sum().alias("n_used"),
        pl.col("n_incidents").sum().alias("n_uses"),
        pl.col("n_incidents").filter(pl.col("n_incidents") > 0).mean().alias("mean_incidents_per_used_episode"),
        pl.col("n_incidents").max().alias("max_incidents_per_episode"))
        .with_columns((pl.col("n_uses") / pl.col("n_library")).alias("mean_incidents_per_library_episode"))
        .sort("type", "pool"))


def background_reuse(variants: tuple[str, ...], ds: str = "ton") -> dict:
    """Mean number of incidents whose window covers a background record, per pool."""
    from certichain.chainforge.background import BackgroundLibrary

    bg = BackgroundLibrary.load(ds)
    out = {}
    wins = defaultdict(list)
    import yaml

    from certichain.chainforge.generator import source_pool
    from certichain.io.paths import CONFIG_DIR
    for v in variants:
        p = interim(v, "index.parquet")
        if not p.exists():
            continue
        cfg_p = CONFIG_DIR / f"{v}.yaml"
        cfg = yaml.safe_load(cfg_p.read_text()) if cfg_p.exists() else {}
        idx = pl.read_parquet(p).select("split_role", "window")
        for sp, w in idx.iter_rows():
            pool = source_pool(sp, cfg)
            wins[pool].append((float(w[0]), float(w[1])))
    for pool, ws in wins.items():
        ts = bg.records(pool)["ts"].to_numpy()
        cover = np.zeros(ts.size, dtype=np.int64)
        for a, b in ws:
            lo, hi = np.searchsorted(ts, a, "left"), np.searchsorted(ts, b, "left")
            cover[lo:hi] += 1
        out[pool] = {"n_windows": len(ws), "mean_incidents_per_record": float(cover.mean()),
                     "share_records_never_used": float((cover == 0).mean()), "max": int(cover.max())}
    return out


# --------------------------------------------------------------------------- P2-B's 07 on new variant names
_M07: dict = {}


def _load_07():
    """P2-B's scripts/07 loaded as a module from its (unchanged) file."""
    if "m" not in _M07:
        import importlib.util
        from certichain.io.paths import P2_ROOT
        spec = importlib.util.spec_from_file_location("p2b_scripts_07", P2_ROOT / "scripts" / "07_units_candidates_gt.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _M07["m"] = m
    return _M07["m"]


def _run07_task(task: dict) -> dict:
    m = _load_07()
    m.OUT = Path(task["out_root"])
    try:
        return m.process_incident(task["inc"], task["cfg"], in_worker=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"inc_id": task["inc"]["inc_id"], "ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def run_p2b_07(variant: str, as_variant: str, workers: int = 4, overwrite: bool = False) -> dict:
    """Run P2-B's scripts/07 per-incident processing (its own process_incident, unchanged) on a ChainForge variant
    whose name 07 does not whitelist (e.g. cf_gotham_v2). Incidents are listed exactly like 07's
    incidents_chainforge, from results/interim/<variant>/incidents; ``ds`` = ``as_variant`` (a whitelisted CF
    variant, so the same CF code path and GT adapter are used); outputs go through a temporary root whose
    ``as_variant`` entry is a symlink to results/interim/<variant>, so files land in
    results/interim/<variant>/<sub>/<inc_id>.* (the layout 07 writes for the whitelisted variants)."""
    import os
    import shutil
    from certichain.io.paths import run_pool
    m = _load_07()
    base = interim(variant)
    out_root = interim(f"_07root_{variant}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    os.symlink(base, out_root / as_variant)
    try:
        m.OUT = out_root
        for sub in m.OUT_DIRS:
            (out_root / as_variant / sub).mkdir(parents=True, exist_ok=True)
        incs = []
        for meta_p in sorted((base / "incidents").glob("**/meta.json")):
            d = meta_p.parent
            meta = json.loads(meta_p.read_text())
            seeds = meta.get("seed_device")
            seeds = seeds if isinstance(seeds, list) else [seeds]
            incs.append({"ds": as_variant, "inc_id": meta["inc_id"], "seeds": seeds, "t0": meta.get("t0"),
                         "t1": meta.get("t1"), "records": str(d / "records.parquet"), "incident_dir": str(d),
                         "internal_prefixes": meta.get("internal_prefixes") or [], "n_records": None})
        if not overwrite:
            incs = [i for i in incs if not m.out_path(as_variant, "stats", i["inc_id"], "json").exists()]
        cfg = m.load_cfg()
        tasks = [{"inc": i, "cfg": cfg, "out_root": str(out_root)} for i in incs]
        log(f"07 on {variant} (as {as_variant}): {len(tasks)} pending incidents, {workers} workers")
        res, errs = run_pool(_run07_task, tasks, workers, label=lambda t: t["inc"]["inc_id"])
        fails = [r for r in res if not r.get("ok")] + errs
        for r in fails:
            log(f"07 FAILED {r.get('inc_id') or r.get('task')}: {r.get('error')}")
        summ = m.summarize(as_variant)
        summ["ds"] = variant
        summ["note"] = f"P2-B scripts/07 process_incident run by chainforge.audit.run_p2b_07 (ds={as_variant} code path)"
        atomic_write_json(base / "summary_07.json", summ)
        for sp in (base / "stats").glob("**/*.json"):            # cosmetic: stats 'ds' = the real variant
            st = json.loads(sp.read_text())
            if st.get("ds") == as_variant:
                st["ds"] = variant
                atomic_write_json(sp, st)
        return {"variant": variant, "n_done": len(res) - len(fails), "n_failed": len(fails),
                "n_incidents": summ.get("n_incidents")}
    finally:
        try:
            os.unlink(out_root / as_variant)
            shutil.rmtree(out_root, ignore_errors=True)
        except OSError:
            pass


def frac_probe(variants: tuple[str, ...], role: str = "train") -> dict:
    """E15 fractional-second check on ``role`` incidents (P2-G's probe_units / probe_auc, unchanged): KS of
    unit-start fractional seconds (spliced vs native background, D3 vs background) and probe AUCs for the
    fraction-only / single-feature / timing / all feature sets. Writes <variant>/audit/probe_units_<role>.parquet."""
    import glob as _glob

    from scipy.stats import ks_2samp

    from certichain.eval.experiments import PROBE_SCORER, PROBE_TIMING, probe_auc, probe_units
    sets = {"frac": ("frac_ts", "frac_rec_mean", "frac_rec_zero"), "frac_ts": ("frac_ts",),
            "frac_rec_mean": ("frac_rec_mean",), "n_rec": ("n_rec_probe",), "timing": PROBE_TIMING,
            "all": PROBE_TIMING + PROBE_SCORER}
    out = {}
    for ds in variants:
        base = interim(ds, "stats")
        incs = sorted(str(Path(f).relative_to(base).with_suffix(""))
                      for f in _glob.glob(str(base / "**" / "*.json"), recursive=True) if not Path(f).name.startswith("_"))
        incs = [i for i in incs if i.split("/")[1] == role]
        df = pl.concat([q for q in (probe_units(ds, i) for i in incs) if q.height], how="diagonal_relaxed")
        atomic_write_parquet(df, interim(ds, "audit", f"probe_units_{role}.parquet"))
        nat = df.filter(pl.col("kind") == "background")["frac_ts"].to_numpy()
        sp = df.filter(pl.col("kind").is_in(["main", "D1", "D2", "D3"]))["frac_ts"].to_numpy()
        d3 = df.filter(pl.col("kind") == "D3")["frac_ts"].to_numpy()
        r = {"n_incidents": len(incs), "kinds": {k: int(n) for k, n in df.group_by("kind").len().iter_rows()},
             "ks_spliced_vs_background": dict(zip(("D", "p"), map(float, ks_2samp(sp, nat)))) if sp.size else None,
             "ks_D3_vs_background": dict(zip(("D", "p"), map(float, ks_2samp(d3, nat)))) if d3.size else None,
             "frac_rec_zero_by_kind": {k: float(v) for k, v in df.group_by("kind").agg(pl.col("frac_rec_zero").mean()).iter_rows()}}
        for target in ("ii", "i"):
            for name, feats in sets.items():
                a = probe_auc(df, target, feats)
                r[f"auc_{target}_{name}"] = [round(a["auc"], 3), round(a["lo"], 3), round(a["hi"], 3)]
        out[ds] = r
        atomic_write_json(interim(ds, "audit", f"frac_probe_{role}.json"), r)
        log(f"{ds} frac probe ({role}): {json.dumps(r)[:400]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--no-heldout", action="store_true")
    ap.add_argument("--run07-as", default=None, help="first run P2-B's 07 on each --variant via this whitelisted name")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frac-probe", action="store_true", help="only run the E15 fractional-second check (train)")
    a = ap.parse_args(argv)
    res = {}
    if a.frac_probe:
        frac_probe(tuple(a.variant))
        return 0
    if a.run07_as:
        for v in a.variant:
            log(json.dumps(run_p2b_07(v, a.run07_as.replace("{v}", v), workers=a.workers)))
    for v in a.variant:
        res[v] = audit_variant(v)
        log(f"{v}: audited {res[v]['n_audited']}/{res[v]['n_incidents']} incidents")
    ton = tuple(v for v in a.variant if v.startswith("cf_ton"))
    if ton:
        rt = reuse_table(ton, "ton")
        if rt.height:
            for v in ton:
                atomic_write_parquet(rt, interim(v, "audit", "reuse.parquet"))
            atomic_write_json(interim(ton[0], "audit", "background_reuse.json"), background_reuse(ton, "ton"))
    gth = tuple(v for v in a.variant if v.startswith("cf_gotham"))
    if gth:
        rt = reuse_table(gth, "gotham")
        if rt.height:
            for v in gth:
                atomic_write_parquet(rt, interim(v, "audit", "reuse.parquet"))
            atomic_write_json(interim(gth[0], "audit", "background_reuse.json"), background_reuse(gth, "gotham"))
    if "cf_ton_h" in a.variant and not a.no_heldout and not a.run07_as:
        h = heldout_check()
        log(f"held-out: {h['n_edges_unseen']}/5 held-out transitions unseen at the unit-type level")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
