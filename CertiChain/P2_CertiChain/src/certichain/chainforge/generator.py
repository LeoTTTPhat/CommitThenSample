"""ChainForge-IoT incident generator — DESIGN §5.4.3 Alg. 7 (one RNG stream per incident).

    seed = first 8 bytes (LE) of BLAKE3(f"{seed_variant}|{split_role}|{idx}")   (split_role ∈ train|dev|caltest;
           train and dev both draw from the TRAIN pools but use distinct streams)

 1  n_camp ~ {0: .10, 1: .55, 2: .25, 3: .10}; templates ~ Unif(allowed); episodes ~ Unif(library[pool][type])
    among episodes eligible for the node's binding (no reuse inside an incident while possible); one
    binding mode per node (e.g. flood from the compromised victim or from the same C2); lags per edge
    (gap from the parent's last record start to the child's first record start, LogUniform, ≤ 0.8 W)
    decoys ~ Poisson(2), kind D1 .4 / D2 .2 / D3 .4 (a D2 needs a campaign with an external infra role
    not already shared by another D2; otherwise the kind is redrawn from D1/D3, logged)
 2  span = max campaign (and D2) span; window B ~ pool with L = max(60 min, span + 30 min) + U(0, 30 min)
    at a uniform offset (redrawn until it holds enough template-compatible devices)
 3  seed x ~ Unif(compatible devices of B); campaign 0's victim = x; campaign k ≥ 1's victim = x w.p. ½,
    else an internal neighbour of x in B (in scope by construction); pivot victim Y ~ other compatible
    device; campaign start ~ Unif over the feasible interval; role IPs (TEST-NET attacker/C2/infra,
    devices for X/Y) per chainforge/rewrite.py
 4  D1: one episode by a DIFFERENT (fresh) attacker on the victim of a random campaign step, placed
        before/after that step with the lag law of the type pair (|gap| ≤ 24 min < W);
    D2: a truncated campaign (≥ 2 episodes, contains the infra node) on another device z that reuses
        the campaign's external infrastructure roles (C2 / loader / download / CNC / report) and has its
        own attacker; its infra step starts within ±W of the campaign's infra step; own campaign id,
        internal transitions true;
    D3: an attack-like benign unit from a background tile disjoint from B, re-hosted on the victim of
        a random campaign step, anchored like D1 (default lag law)
 5  evidence loss: each main/D2 episode hidden w.p. ℓ (a uniform is drawn for every episode so ℓ
    variants with the same seed_variant are paired)
 6  merge background + episodes, drop unobservable (ext→ext) records, round ts/te (CF-TON 1 s),
    sort by (ts, tiehash, …), rec_id = random permutation; write records / labels / meta / gt / gen

Outputs per incident (results/interim/<variant>/incidents/<split_role>/<NNNNNN>/):
    records.parquet   RECORD_SCHEMA (label-free; src_locator "<ds>:<file>:<orig locator>:rec=<orig rec_id>")
    labels.parquet    QUARANTINED: rec_id, label, detailed_label, episode_id, campaign_id, decoy_kind, template_node
    meta.json         case metadata only (no GT)
    gt.json           QUARANTINED: campaigns, episodes, transitions, hidden
    gen.json          QUARANTINED generator log: sources, roles, window, lags, placements, counts
"""
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import blake3
import numpy as np
import polars as pl

from certichain.chainforge import templates as TP
from certichain.chainforge.background import BackgroundLibrary, compatible_devices, internal_neighbours
from certichain.chainforge.episodes import EpisodeLibrary
from certichain.chainforge.rewrite import IPPlan, apply_ip_map, round_times, shift_times, visible_mask
from certichain.io.hashing import tiehash_series
from certichain.io.labels import assert_label_free
from certichain.io.paths import atomic_write_json, atomic_write_parquet, interim
from certichain.io.schema import RECORD_SCHEMA, conform

INFRA_ROLES = ("C2", "LOADER", "DLSRV", "CNC", "REPSRV")
EXTERNAL_ROLES = ("A", "C2", "SCANNER", "LOADER", "DLSRV", "CNC", "REPSRV")
DEVICE_ROLES = ("X", "Y")
LABEL_OUT_SCHEMA = {"rec_id": pl.Int64, "label": pl.Utf8, "detailed_label": pl.Utf8, "episode_id": pl.Utf8,
                    "campaign_id": pl.Utf8, "decoy_kind": pl.Utf8, "template_node": pl.Utf8}

DEFAULTS = {
    "n_camp_p": [0.10, 0.55, 0.25, 0.10],
    "decoy_rate": 2.0,
    "decoy_kind_p": {"D1": 0.4, "D2": 0.2, "D3": 0.4},
    "evidence_loss": 0.0,
    "window": {"min_s": 3600.0, "margin_s": 1800.0, "extra_s": 1800.0, "max_s": None},
    "ts_round": 1.0,
    "W": TP.W_CAND,
    "victim_is_seed_p": 0.5,
    "max_window_draws": 200,
    "max_template_resamples": 5,
}


# --------------------------------------------------------------------------- seeds / config
def incident_seed(seed_variant: str, split_role: str, idx: int) -> int:
    d = blake3.blake3(f"{seed_variant}|{split_role}|{idx}".encode("utf-8")).digest()[:8]
    return int.from_bytes(d, "little")


def source_pool(split_role: str, cfg: dict | None = None) -> str:
    """train/dev → TRAIN pools, caltest → cal/test pools; a config may remap a split role (CF-GOTHAM-T: its
    calibration-bank incidents, split_role caltest, are built from the TRAIN pools)."""
    over = ((cfg or {}).get("pool_of_split") or {}).get(split_role)
    if over:
        return over
    return "train" if split_role in ("train", "dev") else "caltest"


def merged_config(cfg: dict | None) -> dict:
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- specs
@dataclass
class EpisodeSpec:
    eid: str
    cid: str
    kind: str                      # main | D1 | D2 | D3
    node: str                      # template node id ("D1" / "D3" for decoys)
    type: str
    lib_eid: str | None = None     # library episode (None: hidden template node or D3)
    la_id: str | None = None       # D3 look-alike id
    binds: dict = field(default_factory=dict)   # slot → role
    mode: str = "default"
    rel_start: float = 0.0
    dur: float = 0.0
    start: float | None = None
    hidden: bool = False
    hidden_reason: str | None = None

    @property
    def end(self) -> float:
        return float(self.start) + float(self.dur)


@dataclass
class CampaignSpec:
    cid: str
    kind: str
    template: str
    nodes: list[str]
    edges: list[tuple[str, str, str, float]]    # (node_a, node_b, edge kind, lag s)
    episodes: dict[str, EpisodeSpec]
    span: float
    start: float | None = None
    roles: dict[str, str] = field(default_factory=dict)        # role → incident IP
    role_devices: dict[str, str] = field(default_factory=dict) # device role → source (testbed) IP
    host_cid: str | None = None                                # D2: campaign whose infra it shares
    anchor: dict | None = None

    def needs(self) -> set[str]:
        return {r for e in self.episodes.values() for r in e.binds.values()}


def _choice_p(rng, items, p):
    p = np.asarray(p, dtype=float)
    return items[int(rng.choice(len(items), p=p / p.sum()))]


def sample_campaign(rng, T: TP.Template, cid: str, kind: str, pool: str, lib: EpisodeLibrary, used: set[str],
                    subset: tuple[str, ...] | None = None) -> CampaignSpec:
    nodes = [n for n in T.topo() if subset is None or n in subset]
    eps: dict[str, EpisodeSpec] = {}
    for nid in nodes:
        nd = T.node(nid)
        eid = f"{cid}.{nid}"
        if nd.hidden:
            eps[nid] = EpisodeSpec(eid=eid, cid=cid, kind=kind, node=nid, type=nd.type, hidden=True,
                                   hidden_reason="template", dur=0.0)
            continue
        mode = nd.modes[0] if len(nd.modes) == 1 else _choice_p(rng, list(nd.modes), [m.p for m in nd.modes])
        binds = mode.as_dict()
        lib_eid = lib.sample(rng, pool, nd.type, binds, used)
        used.add(lib_eid)
        dur = float(lib.row(lib_eid)["dur"])
        eps[nid] = EpisodeSpec(eid=eid, cid=cid, kind=kind, node=nid, type=nd.type, lib_eid=lib_eid, binds=binds,
                               mode=mode.name, dur=dur)
    edges = []
    rel: dict[str, float] = {}
    for nid in nodes:
        par = [e for e in T.edges if e.b == nid and e.a in nodes]
        if not par:
            rel[nid] = 0.0
            continue
        e = par[0]
        lag = TP.LAGS[e.lag].sample(rng)
        rel[nid] = rel[e.a] + eps[e.a].dur + lag
        edges.append((e.a, nid, e.kind, float(lag)))
    lo = min(rel.values())
    for nid in nodes:
        eps[nid].rel_start = rel[nid] - lo
    span = max(eps[n].rel_start + eps[n].dur for n in nodes)
    return CampaignSpec(cid=cid, kind=kind, template=T.id, nodes=nodes, edges=edges, episodes=eps, span=float(span))


def _like_ip(c: "CampaignSpec", role: str, lib: EpisodeLibrary) -> str | None:
    """Source IP of the first slot bound to ``role`` in the campaign (subnet-mode role IPs)."""
    for n in c.nodes:
        e = c.episodes[n]
        if not e.lib_eid:
            continue
        for slot, r in e.binds.items():
            if r == role:
                ip = lib.row(e.lib_eid)["slots"].get(slot)
                if ip:
                    return ip
    return None


def _role_ip(plan: IPPlan, role: str, cfg: dict, like: str | None) -> str:
    if plan.mode != "subnets":
        return plan.new_role_ip()
    like = like or (cfg.get("role_subnets") or {}).get(role)
    if not like:
        raise ValueError(f"no subnet for role {role} (set role_subnets in the config)")
    return plan.fresh_in_subnet(like)


def decoy_types(allowed: tuple[str, ...]) -> list[str]:
    return sorted({t for tid in allowed for t in TP.TEMPLATES[tid].types(observable_only=True)})


def _d1_binds(etype: str) -> dict[str, str]:
    return {"actor": "C2", "victim": "X"} if etype in ("backdoor", "ransomware") else {"actor": "A", "victim": "X"}


# --------------------------------------------------------------------------- Alg. 7 (sampling)
@dataclass
class IncidentSpec:
    variant: str
    split_role: str
    idx: int
    seed: int
    pool: str
    ds: str
    campaigns: list[CampaignSpec]
    decoys: list[CampaignSpec]
    b0: float = 0.0
    L: float = 0.0
    seed_ton: str | None = None
    log: dict = field(default_factory=dict)


def sample_incident(variant: str, split_role: str, idx: int, cfg: dict, lib: EpisodeLibrary,
                    bglib: BackgroundLibrary, rng: np.random.Generator | None = None,
                    plan: IPPlan | None = None) -> tuple[IncidentSpec, np.random.Generator, IPPlan]:
    cfg = merged_config(cfg)
    ds = cfg.get("dataset", lib.ds)
    seed = incident_seed(cfg.get("seed_variant") or variant, split_role, idx)
    rng = rng if rng is not None else np.random.default_rng(seed)
    plan = plan if plan is not None else IPPlan(rng, mode=cfg.get("ip_plan", "ton"))
    pool = source_pool(split_role, cfg)
    allowed = TP.allowed_templates(variant, split_role, cfg)
    log: dict = {"d2_redrawn": 0, "d2_no_device": 0, "template_resamples": 0, "window_draws": 0, "truncated": []}
    wcfg = cfg["window"]

    # ---- step 1: campaigns + decoy structures (resampled if a capped window cannot hold them)
    n_camp = int(rng.choice(4, p=np.asarray(cfg["n_camp_p"], dtype=float)))
    for attempt in range(int(cfg["max_template_resamples"]) + 1):
        used: set[str] = set()
        camps = []
        for k in range(n_camp):
            c = sample_campaign(rng, TP.TEMPLATES[allowed[int(rng.integers(len(allowed)))]], f"c{k}", "main",
                                pool, lib, used)
            # victim = seed for campaign 0; for k ≥ 1 w.p. victim_is_seed_p (else a neighbour of the seed)
            c.anchor = {"victim_is_seed": bool(k == 0 or rng.random() < float(cfg["victim_is_seed_p"]))}
            camps.append(c)
        n_dec = int(rng.poisson(float(cfg["decoy_rate"])))
        kinds_p = cfg["decoy_kind_p"]
        decoys: list[CampaignSpec] = []
        d2_hosts: set[str] = set()
        for j in range(n_dec):
            kind = _choice_p(rng, ["D1", "D2", "D3"], [kinds_p["D1"], kinds_p["D2"], kinds_p["D3"]])
            if kind == "D2":
                # a D2 shares the infra of a campaign whose victim is the seed: the seed contacts that
                # infra, so the D2 device enters the scope by shared-external expansion (Alg. 1 step 6)
                hosts = [c for c in camps if c.cid not in d2_hosts and c.anchor["victim_is_seed"]
                         and any(TP.TEMPLATES[c.template].node(n).infra and not c.episodes[n].hidden
                                 for n in c.nodes)]
                t2s = [t for t in allowed if TP.TEMPLATES[t].subtree_chain_options()]
                host = None
                if not hosts or not t2s:
                    log["d2_redrawn"] += 1
                    if kinds_p["D1"] + kinds_p["D3"] <= 0:
                        log["d2_dropped"] = log.get("d2_dropped", 0) + 1
                        continue
                    kind = _choice_p(rng, ["D1", "D3"], [kinds_p["D1"], kinds_p["D3"]])
                else:
                    host = hosts[int(rng.integers(len(hosts)))]
                    host_roles = set(host.needs()) & set(INFRA_ROLES)

                    def _shares(T, sub):
                        return any(r in host_roles for n in sub for m in T.node(n).modes for _, r in m.binds)

                    # the parallel infection must reuse ≥ 1 infra role of its host (always true for TON: C2)
                    t2h = [t for t in t2s if any(_shares(TP.TEMPLATES[t], o) for o in TP.TEMPLATES[t].subtree_chain_options())]
                    if not t2h:
                        log["d2_redrawn"] += 1
                        host = None
                        if kinds_p["D1"] + kinds_p["D3"] <= 0:
                            log["d2_dropped"] = log.get("d2_dropped", 0) + 1
                            continue
                        kind = _choice_p(rng, ["D1", "D3"], [kinds_p["D1"], kinds_p["D3"]])
                if kind == "D2" and host is not None:
                    T2 = TP.TEMPLATES[t2h[int(rng.integers(len(t2h)))]]
                    opts = [o for o in T2.subtree_chain_options() if _shares(T2, o)]
                    sub = opts[int(rng.integers(len(opts)))]
                    c2 = sample_campaign(rng, T2, f"d{j}", "D2", pool, lib, used, subset=sub)
                    c2.host_cid = host.cid
                    d2_hosts.add(host.cid)
                    decoys.append(c2)
                    continue
            if kind == "D1":
                d1 = (cfg.get("d1") or {})
                types = [t for t in (d1.get("types", {}).get(pool) if isinstance(d1.get("types"), dict)
                                     else d1.get("types")) or decoy_types(allowed) if t in lib.types(pool)]
                etype = types[int(rng.integers(len(types)))]
                binds = dict(d1["binds"]) if d1.get("binds") else _d1_binds(etype)
                lib_eid = lib.sample(rng, pool, etype, binds, used)
                used.add(lib_eid)
                e = EpisodeSpec(eid=f"d{j}.{etype}", cid=f"d{j}", kind="D1", node="D1", type=etype, lib_eid=lib_eid,
                                binds=binds, dur=float(lib.row(lib_eid)["dur"]))
                decoys.append(CampaignSpec(cid=f"d{j}", kind="D1", template=f"D1:{etype}", nodes=["D1"], edges=[],
                                           episodes={"D1": e}, span=e.dur))
            else:  # D3 (look-alike chosen once the window is known)
                e = EpisodeSpec(eid=f"d{j}.lookalike", cid=f"d{j}", kind="D3", node="D3", type="benign_lookalike",
                                binds={"host": "X"})
                decoys.append(CampaignSpec(cid=f"d{j}", kind="D3", template="D3:lookalike", nodes=["D3"], edges=[],
                                           episodes={"D3": e}, span=0.0))
        spans = [c.span for c in camps] + [d.span for d in decoys if d.kind == "D2"]
        need = max([wcfg["min_s"]] + [s + wcfg["margin_s"] for s in spans])
        if wcfg.get("max_s") is None or need <= float(wcfg["max_s"]):
            break
        log["template_resamples"] += 1
    L = need + float(rng.uniform(0.0, float(wcfg["extra_s"])))
    if wcfg.get("max_s") is not None:
        L = min(L, float(wcfg["max_s"]))
    L = float(math.ceil(L))

    # ---- step 2: window with enough compatible devices
    t_lo, t_hi = bglib.span(pool)
    n_need = 1 + sum(1 for c in camps + decoys if "Y" in c.needs()) + sum(1 for d in decoys if d.kind == "D2")
    best = None
    for draw in range(int(cfg["max_window_draws"])):
        b0 = float(math.floor(rng.uniform(t_lo, max(t_lo, t_hi - L))))
        win = bglib.window(pool, b0, b0 + L)
        comp = compatible_devices(win)
        log["window_draws"] = draw + 1
        if best is None or len(comp) > len(best[2]):
            best = (b0, win, comp)
        if len(comp) >= n_need:
            break
    b0, win, comp = best
    if not comp:
        raise RuntimeError("no window with a compatible device")

    # ---- step 3: seed device, victims, starts
    x = comp[int(rng.integers(len(comp)))]
    nbrs = [h for h in internal_neighbours(win, x) if h in comp]
    taken = {x}
    for k, c in enumerate(camps):
        if c.anchor["victim_is_seed"] or not nbrs:
            v = x
        else:
            v = nbrs[int(rng.integers(len(nbrs)))]
        c.anchor["victim_is_seed"] = bool(v == x)
        c.role_devices["X"] = v
        taken.add(v)
    for c in camps:
        if "Y" in c.needs():
            pool_y = [h for h in comp if h not in taken] or [h for h in comp if h != c.role_devices["X"]]
            if not pool_y:
                raise RuntimeError("no device for the pivot victim Y")
            c.role_devices["Y"] = pool_y[int(rng.integers(len(pool_y)))]
            taken.add(c.role_devices["Y"])
    kept: list[CampaignSpec] = []
    for d in decoys:
        if d.kind == "D2":
            pool_z = [h for h in comp if h not in taken]
            if not pool_z:
                log["d2_no_device"] += 1
                continue
            d.role_devices["X"] = pool_z[int(rng.integers(len(pool_z)))]
            taken.add(d.role_devices["X"])
            if "Y" in d.needs():           # only as a scan target (subtree_chain_options): the D2 bot scans the seed
                d.role_devices["Y"] = x
        kept.append(d)
    decoys = kept
    for c in camps:
        c.start = b0 + float(math.floor(rng.uniform(0.0, max(0.0, L - c.span))))
        for e in c.episodes.values():
            e.start = c.start + e.rel_start
            if not e.hidden and e.start + e.dur > b0 + L:
                e.hidden, e.hidden_reason = True, "truncated"
                log["truncated"].append(e.eid)

    spec = IncidentSpec(variant=variant, split_role=split_role, idx=idx, seed=seed, pool=pool, ds=ds,
                        campaigns=camps, decoys=decoys, b0=b0, L=L, seed_ton=x, log=log)

    # ---- step 3b: role IPs (fixed order: campaigns, then decoys)
    for c in camps:
        for r in sorted(c.needs()):
            if r in DEVICE_ROLES:
                c.roles[r] = plan.device_ip(c.role_devices[r])
            elif r not in c.roles:
                c.roles[r] = _role_ip(plan, r, cfg, _like_ip(c, r, lib))
    by_cid = {c.cid: c for c in camps}

    # ---- step 4: decoy placement
    obs_steps = [e for c in camps for n in c.nodes for e in [c.episodes[n]] if not e.hidden]
    W = float(cfg["W"])
    for d in decoys:
        if d.kind == "D2":
            host = by_cid[d.host_cid]
            for r in sorted(d.needs()):
                if r in DEVICE_ROLES:
                    d.roles[r] = plan.device_ip(d.role_devices[r])
                elif r in INFRA_ROLES and r in host.roles:
                    d.roles[r] = host.roles[r]
                else:
                    d.roles[r] = _role_ip(plan, r, cfg, _like_ip(d, r, lib))
            h_inf = next(host.episodes[n] for n in host.nodes
                         if TP.TEMPLATES[host.template].node(n).infra and not host.episodes[n].hidden)
            d_inf = next(d.episodes[n] for n in d.nodes if TP.TEMPLATES[d.template].node(n).infra)
            target = h_inf.start + float(rng.uniform(-W, W))
            start = target - d_inf.rel_start
            start = max(min(start, b0 + L - d.span), b0)
            d.start = float(math.floor(start))
            for e in d.episodes.values():
                e.start = d.start + e.rel_start
                if not e.hidden and e.start + e.dur > b0 + L:
                    e.hidden, e.hidden_reason = True, "truncated"
                    log["truncated"].append(e.eid)
            d.anchor = {"host_episode": h_inf.eid, "offset_s": float(d_inf.start - h_inf.start)}
            continue
        e = d.episodes[d.nodes[0]]
        if d.kind == "D3":
            la = _pick_lookalike(rng, bglib, pool, b0, b0 + L)
            if la is None:
                e.hidden, e.hidden_reason = True, "no_lookalike"
                spec.log.setdefault("d3_missing", 0)
                spec.log["d3_missing"] = spec.log["d3_missing"] + 1
                continue
            e.la_id = la["la_id"]
            e.dur = float(la["ts_last"] - la["ts"])
            e.node = "D3"
            d.template = f"D3:{la['type']}"
        if obs_steps:
            anc = obs_steps[int(rng.integers(len(obs_steps)))]
            host_c = by_cid[anc.cid]
            vrole = anc.binds.get("victim") or (anc.binds.get("actor") if anc.binds.get("actor") in DEVICE_ROLES
                                                else None)
            vdev = host_c.role_devices.get(vrole) if vrole else None
            vdev = vdev or x
            after = bool(rng.random() < 0.5)
            if d.kind == "D1":
                law = TP.edge_lag_law(anc.type, e.type) if after else TP.edge_lag_law(e.type, anc.type)
            else:
                law = TP.LAGS["default"]
            lag = law.sample(rng)
            start = anc.end + lag if after else anc.start - lag - e.dur
            d.anchor = {"episode": anc.eid, "after": after, "lag_s": float(lag), "lag_law": law.name}
        else:
            vdev = x
            start = float(rng.uniform(b0, b0 + L - e.dur))
            d.anchor = {"episode": None}
        start = min(max(start, b0), b0 + L - e.dur)
        e.start = float(math.floor(start))
        d.start = e.start
        d.span = e.dur
        d.role_devices["X"] = vdev
        d.roles["X"] = plan.device_ip(vdev)
        for r in sorted(d.needs()):
            if r not in d.roles and r not in DEVICE_ROLES and r != "host":
                d.roles[r] = _role_ip(plan, r, cfg, _like_ip(d, r, lib))

    # ---- step 5: evidence loss (one uniform per main/D2 episode, always drawn)
    ell = float(cfg["evidence_loss"])
    for c in camps + [d for d in decoys if d.kind == "D2"]:
        for n in c.nodes:
            e = c.episodes[n]
            u = float(rng.random())
            if not e.hidden and u < ell:
                e.hidden, e.hidden_reason = True, "evidence_loss"
    return spec, rng, plan


def _pick_lookalike(rng, bglib: BackgroundLibrary, pool: str, b0: float, b1: float) -> dict | None:
    la = bglib.lookalikes(pool)
    if la.height == 0:
        return None
    ok = la.filter((pl.col("tile_t1") <= b0) | (pl.col("tile_t0") >= b1))
    if ok.height == 0:
        return None
    return ok.row(int(rng.integers(ok.height)), named=True)


# --------------------------------------------------------------------------- materialisation
def _offset(off: float, cfg: dict, first: float | None = None, last: float | None = None,
            b0: float | None = None, b1: float | None = None, key: str | None = None) -> float:
    """Episode time shift.

    * ts_round set (CF-TON): integral multiples of the rounding step so 1-s source times stay exact (unchanged).
    * time_shift "exact" (default; CF-GOTHAM v1): the real-valued offset that puts the episode's first record
      exactly on its nominal start — root episodes and decoys then start on INTEGER seconds (E15 artifact).
    * time_shift "integer_seconds" (CF-GOTHAM v2): the offset rounded to whole seconds, so every spliced record
      keeps its source record's own sub-second part (microsecond capture times, like the native background);
      nudged by ±1 s when needed to keep the episode's record starts inside the window [b0, b1].
    * time_shift "random_fraction": whole-second offset plus a per-episode U(0, 1) fraction from a separate
      stream keyed by ``key`` (alternative evaluated for the v2 choice; never consumes the incident RNG; it
      perturbs sub-second template lags by up to ±1 s, so it is NOT used).
    * time_shift "campaign_phase" (CF-GOTHAM v2): the exact v1 offset plus ONE U(0, 1) sub-second phase per
      campaign / decoy (``key`` = incident seed | campaign id; separate stream), so every intra-campaign lag is
      preserved exactly while the campaign's sub-second phase relative to the background is uniform; the
      phase is applied as u − 1 instead of u when that keeps the campaign inside the window (same fraction).
    """
    res = cfg.get("ts_round")
    if res:
        return float(round(off / float(res)) * float(res))
    mode = cfg.get("time_shift", "exact")
    if mode == "exact":
        return float(off)
    if mode == "campaign_phase":
        return float(off) + _campaign_phase(key, first, last, b0, b1)
    if mode == "integer_seconds":
        k = float(round(off))
    elif mode == "random_fraction":
        h = blake3.blake3(f"tshift|{key}".encode()).digest()[:8]
        k = float(math.floor(off)) + float(np.random.default_rng(int.from_bytes(h, "little")).random())
    else:
        raise ValueError(f"unknown time_shift {mode!r}")
    if first is not None and b0 is not None and first + k < b0:
        k += float(math.ceil(b0 - (first + k)))
    if last is not None and b1 is not None and last + k > b1 and (first is None or b0 is None or first + k - 1.0 >= b0):
        k -= float(math.ceil((last + k) - b1))
    return k


_PHASE: dict = {}


def _phase_u(key: str) -> float:
    h = blake3.blake3(f"tphase|{key}".encode()).digest()[:8]
    return float(np.random.default_rng(int.from_bytes(h, "little")).random())


def _campaign_phase(key: str | None, first, last, b0, b1) -> float:
    """Phase registered per campaign key by ``set_campaign_phases`` (u or u − 1); 0 if unknown."""
    return float(_PHASE.get(key, 0.0))


def set_campaign_phases(spec: "IncidentSpec", cfg: dict) -> dict:
    """One sub-second phase per campaign / decoy (time_shift "campaign_phase"); u − 1 when the campaign's last
    nominal record start + u would leave the window and its first − 1 + u stays inside."""
    _PHASE.clear()
    if cfg.get("ts_round") or cfg.get("time_shift") != "campaign_phase":
        return {}
    b0, b1 = float(spec.b0), float(spec.b0 + spec.L)
    for c in spec.campaigns + spec.decoys:
        eps = [e for e in c.episodes.values() if not e.hidden and e.start is not None]
        if not eps:
            continue
        u = _phase_u(f"{spec.seed}|{c.cid}")
        lo = min(float(e.start) for e in eps)
        hi = max(float(e.start) + float(e.dur) for e in eps)
        if hi + u > b1 and lo + u - 1.0 >= b0:
            u -= 1.0
        for e in eps:
            _PHASE[f"{spec.seed}|{e.eid}"] = u
    return dict(_PHASE)


def _tag(df: pl.DataFrame, label: str | pl.Expr | None, detailed: str | pl.Expr | None, eid, cid, kind, node,
         ds_tag: str) -> pl.DataFrame:
    base = os.path.basename
    files = df["src_file"].unique().to_list() if df.height else []
    fmap = {f: base(f) if f else "" for f in files}
    out = df.with_columns(
        pl.col("rec_id").alias("_orig_rec_id"),
        pl.col("src_file").replace_strict(fmap, default="", return_dtype=pl.Utf8).alias("_file_base"),
        pl.lit(eid, dtype=pl.Utf8).alias("_eid"), pl.lit(cid, dtype=pl.Utf8).alias("_cid"),
        pl.lit(kind, dtype=pl.Utf8).alias("_kind"), pl.lit(node, dtype=pl.Utf8).alias("_node"))
    if label is not None:
        out = out.with_columns((label if isinstance(label, pl.Expr) else pl.lit(label)).alias("_label"),
                               (detailed if isinstance(detailed, pl.Expr) else pl.lit(detailed)).alias("_detailed"))
    return out.with_columns(pl.format("{}:{}:{}:rec={}", pl.lit(ds_tag), pl.col("_file_base"),
                                      pl.col("src_locator").fill_null(""), pl.col("_orig_rec_id")).alias("src_locator")
                            ).drop("_file_base")


def materialize(spec: IncidentSpec, rng: np.random.Generator, plan: IPPlan, lib: EpisodeLibrary,
                bglib: BackgroundLibrary, cfg: dict) -> dict:
    """Records + quarantined labels/GT of one incident (in memory)."""
    cfg = merged_config(cfg)
    pool, ds = spec.pool, spec.ds
    set_campaign_phases(spec, cfg)
    parts: list[pl.DataFrame] = []
    counts: dict = {"episodes": {}}
    # background window (base map β)
    win = bglib.window(pool, spec.b0, spec.b0 + spec.L)
    for ip in sorted(set(win["src"].to_list()) | set(win["dst"].to_list())):
        plan.map_base(ip)
    bg = apply_ip_map(win, {ip: plan.map_base(ip) for ip in set(win["src"].to_list()) | set(win["dst"].to_list())})
    bg = _tag(bg, "0", "normal", None, None, None, None, ds)
    parts.append(bg)
    counts["background"] = bg.height
    all_c = spec.campaigns + spec.decoys
    for c in all_c:
        for n in c.nodes:
            e = c.episodes[n]
            if e.hidden:
                continue
            if c.kind == "D3":
                la = bglib.lookalikes(pool).filter(pl.col("la_id") == e.la_id).row(0, named=True)
                rec = bglib.by_rec_ids(pool, la["rec_ids"])
                src_ip = la["src"]
                ips = sorted(set(rec["src"].to_list()) | set(rec["dst"].to_list()))
                mu = plan.episode_map(ips, {src_ip: c.roles["X"]})
                rec = apply_ip_map(rec, mu)
                rec = shift_times(rec, _offset(float(e.start) - float(la["ts"]), cfg, float(la["ts"]),
                                               float(la["ts_last"]), spec.b0, spec.b0 + spec.L,
                                               key=f"{spec.seed}|{e.eid}"))
                rec = _tag(rec, "0", "normal", e.eid, c.cid, c.kind, f"D3:{la['type']}", ds)
                e.node = f"D3:{la['type']}"
            else:
                row = lib.row(e.lib_eid)
                rec, lab = lib.records(pool, e.lib_eid)
                slots = row["slots"]
                bind = {}
                for slot, role in e.binds.items():
                    s_ip = slots.get(slot)
                    if s_ip:
                        bind[s_ip] = c.roles[role]
                ips = sorted(set(rec["src"].to_list()) | set(rec["dst"].to_list()))
                mu = plan.episode_map(ips, bind)
                rec = apply_ip_map(rec, mu)
                rec = shift_times(rec, _offset(float(e.start) - float(row["t0"]), cfg, float(row["t0"]),
                                               float(row["t1"]), spec.b0, spec.b0 + spec.L,
                                               key=f"{spec.seed}|{e.eid}"))
                rec = rec.join(lab.select("rec_id", pl.col("label").alias("_label"),
                                          pl.col("detailed_label").alias("_detailed")), on="rec_id", how="left")
                node = e.node if c.kind in ("main", "D2") else f"D1:{e.type}"
                rec = _tag(rec, None, None, e.eid, c.cid, c.kind, node, ds)
            parts.append(rec)
            counts["episodes"][e.eid] = {"n_source": rec.height}
    df = pl.concat([p.select(sorted(p.columns)) for p in parts], how="diagonal_relaxed")
    vis = visible_mask(df)
    dropped = df.filter(~vis)
    df = df.filter(vis)
    counts["dropped_unobservable"] = {"total": dropped.height,
                                      "background": int(dropped["_eid"].is_null().sum()) if dropped.height else 0}
    for eid, n in (dropped.filter(pl.col("_eid").is_not_null()).group_by("_eid").len().iter_rows()
                   if dropped.height else []):
        counts["episodes"][eid]["n_dropped_unobservable"] = int(n)
    emitted = dict(df.filter(pl.col("_eid").is_not_null()).group_by("_eid").len().iter_rows()) if df.height else {}
    for eid, d in counts["episodes"].items():
        d["n_emitted"] = int(emitted.get(eid, 0))
    # declare episodes left without any observable record hidden
    for c in all_c:
        for n in c.nodes:
            e = c.episodes[n]
            if not e.hidden and counts["episodes"].get(e.eid, {}).get("n_emitted", 0) == 0:
                e.hidden, e.hidden_reason = True, "unobservable"
    # times, order, rec_id
    if cfg.get("ts_round"):
        df = round_times(df, float(cfg["ts_round"]))
    df = df.with_columns(tiehash_series(df).alias("_tiehash"),
                         pl.Series("_rand", rng.random(df.height), dtype=pl.Float64))
    df = df.sort(["ts", "_tiehash", "te", "obytes", "rbytes", "opkts", "rpkts", "conn_state", "service", "_rand"],
                 nulls_last=True)
    df = df.with_columns(pl.Series("rec_id", rng.permutation(df.height).astype(np.int64), dtype=pl.Int64))
    records = conform(df, RECORD_SCHEMA)
    assert_label_free(records.columns)
    labels = df.select(pl.col("rec_id"), pl.col("_label").alias("label"), pl.col("_detailed").alias("detailed_label"),
                       pl.col("_eid").alias("episode_id"), pl.col("_cid").alias("campaign_id"),
                       pl.when(pl.col("_kind").is_in(["D1", "D2", "D3"])).then(pl.col("_kind"))
                       .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("decoy_kind"),
                       pl.col("_node").alias("template_node"))
    labels = conform(labels, LABEL_OUT_SCHEMA).sort("rec_id")
    counts["n_records"] = records.height
    return {"records": records, "labels": labels, "counts": counts}


def gt_json(spec: IncidentSpec) -> dict:
    camps, eps, trans, hidden = {}, {}, [], []
    for c in spec.campaigns + spec.decoys:
        camps[c.cid] = {"template": c.template, "kind": c.kind}
        for n in c.nodes:
            e = c.episodes[n]
            node = e.node if c.kind in ("main", "D2", "D3") else f"D1:{e.type}"
            eps[e.eid] = {"campaign_id": c.cid, "type": e.type, "template_node": node, "hidden": bool(e.hidden)}
            if e.hidden:
                hidden.append(e.eid)
        for a, b, _kind, _lag in c.edges:
            trans.append([c.episodes[a].eid, c.episodes[b].eid])
    return {"campaigns": camps, "episodes": eps, "transitions": trans, "hidden": sorted(hidden)}


def meta_json(spec: IncidentSpec, plan: IPPlan, cfg: dict) -> dict:
    cfg = merged_config(cfg)
    return {"inc_id": inc_id(spec.variant, spec.split_role, spec.idx), "variant": spec.variant, "pool": spec.pool,
            "split_role": spec.split_role, "seed_device": plan.device_ip(spec.seed_ton),
            "t0": float(spec.b0), "t1": float(spec.b0 + spec.L),
            "scope_rule": cfg.get("scope_rule", "alg1_step6:seed+N1+shared_external;F_infra=3;U_max=1000"),
            "internal_prefixes": plan.internal_prefixes, "generator_seed": int(spec.seed)}


def gen_json(spec: IncidentSpec, plan: IPPlan, counts: dict, cfg: dict) -> dict:
    def camp(c: CampaignSpec) -> dict:
        return {"cid": c.cid, "kind": c.kind, "template": c.template, "host_cid": c.host_cid, "anchor": c.anchor,
                "start": c.start, "span_s": c.span, "roles": c.roles, "role_devices_src": c.role_devices,
                "edges": [{"a": c.episodes[a].eid, "b": c.episodes[b].eid, "kind": k, "lag_s": lag}
                          for a, b, k, lag in c.edges],
                "episodes": [{k: v for k, v in asdict(c.episodes[n]).items()} for n in c.nodes]}
    return {"inc_id": inc_id(spec.variant, spec.split_role, spec.idx), "variant": spec.variant,
            "split_role": spec.split_role, "idx": spec.idx, "seed": int(spec.seed), "pool": spec.pool,
            "dataset": spec.ds, "window": {"b0": spec.b0, "L_s": spec.L}, "seed_device_src": spec.seed_ton,
            "seed_device": plan.device_ip(spec.seed_ton), "internal_prefixes": plan.internal_prefixes,
            "campaigns": [camp(c) for c in spec.campaigns], "decoys": [camp(d) for d in spec.decoys],
            "log": spec.log, "counts": counts,
            "config": {k: v for k, v in merged_config(cfg).items() if k not in ("templates",)}}


# --------------------------------------------------------------------------- I/O
def inc_id(variant: str, split_role: str, idx: int) -> str:
    return f"{variant}/{split_role}/{idx:06d}"


def incident_dir(variant: str, split_role: str, idx: int, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else interim(variant, "incidents")
    return base / split_role / f"{idx:06d}"


def generate_incident(variant: str, split_role: str, idx: int, cfg: dict, lib: EpisodeLibrary,
                      bglib: BackgroundLibrary) -> dict:
    """Sample + materialise one incident (in memory): records, labels, meta, gt, gen."""
    spec, rng, plan = sample_incident(variant, split_role, idx, cfg, lib, bglib)
    out = materialize(spec, rng, plan, lib, bglib, cfg)
    out["meta"] = meta_json(spec, plan, cfg)
    out["gt"] = gt_json(spec)
    out["gen"] = gen_json(spec, plan, out["counts"], cfg)
    out["spec"] = spec
    return out


def write_incident(out: dict, d: Path) -> Path:
    """Atomic directory write: build in <d>.tmp-<pid>, then rename."""
    d = Path(d)
    tmp = d.with_name(f"{d.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    atomic_write_parquet(out["records"], tmp / "records.parquet")
    atomic_write_parquet(out["labels"], tmp / "labels.parquet")
    atomic_write_json(tmp / "meta.json", out["meta"])
    atomic_write_json(tmp / "gt.json", out["gt"])
    atomic_write_json(tmp / "gen.json", out["gen"])
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)
    return d


def incident_summary(out: dict) -> dict:
    spec: IncidentSpec = out["spec"]
    gt = out["gt"]
    lab = out["labels"]
    n_main = sum(1 for c in spec.campaigns)
    kinds = [d.kind for d in spec.decoys]
    hid = set(gt["hidden"])
    T = gt["transitions"]
    return {"inc_id": out["meta"]["inc_id"], "split_role": spec.split_role, "n_records": out["records"].height,
            "n_background": out["counts"]["background"],
            "n_episode_records": int(lab["episode_id"].is_not_null().sum()),
            "n_campaigns": n_main, "templates": [c.template for c in spec.campaigns],
            "n_decoys": len(kinds), "n_D1": kinds.count("D1"), "n_D2": kinds.count("D2"), "n_D3": kinds.count("D3"),
            "n_T": len(T), "n_T_hidden_endpoint": sum(1 for a, b in T if a in hid or b in hid),
            "n_hidden": len(hid), "L_s": spec.L, "lib_eids": sorted({e.lib_eid for c in spec.campaigns + spec.decoys
                                                                      for e in c.episodes.values() if e.lib_eid}),
            "la_ids": sorted({e.la_id for d in spec.decoys for e in d.episodes.values() if e.la_id}),
            "window": [spec.b0, spec.b0 + spec.L], "log": spec.log}


__all__ = ["DEFAULTS", "incident_seed", "source_pool", "merged_config", "EpisodeSpec", "CampaignSpec", "IncidentSpec",
           "sample_campaign", "sample_incident", "materialize", "gt_json", "meta_json", "gen_json", "inc_id",
           "incident_dir", "generate_incident", "write_incident", "incident_summary", "decoy_types",
           "LABEL_OUT_SCHEMA"]
