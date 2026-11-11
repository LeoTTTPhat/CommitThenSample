"""Shared helpers for the core statistical experiments E1-E3 (owner: P1-E; DESIGN §5.4-5.6, §6.1-6.6).

Everything here reads the derived results of other stages (streams, models, designs) and never changes them.

* Stream discovery: test streams from ``results/streams/index.json``; LP streams whose prevalence target was
  capped (q = 1) are identical or near-identical to the next lower target, so streams are grouped by
  (kind, fold, realised-prevalence bin) and only the lowest target of each group is evaluated
  (``distinct_streams``; the dropped ids are reported).
* Designs: π files of P1-D (``results/designs/<stream>/<design>_<rho>.npy``, payload order) and their JSON;
  ``static_spec`` / ``adaptive_spec`` / ``bm_spec`` rebuild the MC engine's ``DesignSpec`` from them (the engine's
  static pass reproduces the saved π exactly; checked by ``check_static_reproduces``).
* Estimands (DESIGN §5.4 evaluation rule): every query of the stream with >= 50 eligible units in the committed
  metadata (b > 0), whole-stream window; strata by Q in {0}, {1-20}, {> 20}; classes D (detectors),
  S (Suricata families), L (oracle-analyst labels, secondary).
* LCS grid centre: V_ref = training-fold per-epoch V of the query under the design family x 2,880
  (``models/<fold>/<seed>/grid.json``). Queries without training hits (V = 0) get the pre-specified fallback
  V_ref = b̄²/φ0 (one floor-level hit); this never uses the test stream (validity does not depend on V_ref).
* E1: incident sets (content units, metadata-defined presence sets), analytic recall / TTE / benign bytes /
  natural-estimator bias and RMSE for static and deterministic policies, and an exact full-unit simulator of
  the adaptive arm (boost + realised-cost controller) that draws a uniform for EVERY payload unit.
* E3: analytic HT variance Σ f²(1-π)/π per query (P1-C's corrected formula), PPI variance, inflation helpers.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..utils.io import RESULTS
from .montecarlo import DesignSpec, Estimand, stable_id
from .tables import StreamTables

__all__ = ["STREAMS", "DESIGNS", "MODELS", "SUMMARY", "Q_STRATA", "K_STRATA", "q_stratum", "k_stratum",
           "prevalence_bin", "query_class", "load_index", "test_streams", "distinct_streams", "load_tables",
           "design_meta", "design_pi", "calibration_of", "lamref_entry", "grid_entry", "static_spec", "adaptive_spec",
           "bm_spec", "check_static_reproduces", "evaluated_queries", "v_ref_value", "build_estimands",
           "pay_index", "hiding_capacity_rows", "incident_sets", "recall_from_pi", "recall_from_masks",
           "tte_from_pi", "tte_from_mask", "benign_bytes", "natural_estimator_stats", "AdaptiveFullSim",
           "ht_var_by_query", "ppi_var_by_query", "spend_stats", "json_default"]

STREAMS = RESULTS / "streams"
DESIGNS = RESULTS / "designs"
MODELS = RESULTS / "models"
SUMMARY = RESULTS / "summary"

Q_STRATA = ("Q=0", "Q1-20", "Q>20")
K_STRATA = ("k<=20", "k21-100", "k>100")
T_EPOCH_S = 30.0


def json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    raise TypeError(type(o))


# ----------------------------------------------------------------------------------------------- #
# strata and labels
# ----------------------------------------------------------------------------------------------- #
def q_stratum(Q: float) -> str:
    """Q in {0}, {1-20}, {> 20} (DESIGN §5.4)."""
    if Q <= 0:
        return "Q=0"
    return "Q1-20" if Q <= 20 else "Q>20"


def k_stratum(k: int) -> str:
    """Incident size strata of E1 (DESIGN §5.5): k <= 20, 21-100, > 100 content-bearing units."""
    if k <= 20:
        return "k<=20"
    return "k21-100" if k <= 100 else "k>100"


def prevalence_bin(p: float | None) -> str:
    """Realised-prevalence bin of an LP stream: '0.1%' (< 0.3%), '1%' ([0.3%, 2.5%)), '5%' (>= 2.5%)."""
    if p is None or not np.isfinite(p):
        return "nat"
    if p < 0.003:
        return "0.1%"
    return "1%" if p < 0.025 else "5%"


def query_class(name: str) -> str:
    """D = protocol detectors, S = Suricata families, L = oracle-analyst labels (secondary)."""
    c = str(name)[:1]
    return c if c in ("D", "S", "L") else "M"


# ----------------------------------------------------------------------------------------------- #
# stream discovery
# ----------------------------------------------------------------------------------------------- #
def load_index() -> list[dict]:
    return json.loads((STREAMS / "index.json").read_text())["streams"]


def _fold_short(fold) -> str:
    if fold is None:
        return "all"
    return str(fold).replace("CTU-IoT-Malware-Capture-", "")


def test_streams(kinds=("lp_iot23", "lp_edge", "lp_gotham", "natural")) -> list[dict]:
    """Test streams of the given kinds with fold (short), target p, realised prevalence and its bin."""
    out = []
    for s in load_index():
        if s["kind"] not in kinds:
            continue
        r = dict(s)
        r["fold_short"] = _fold_short(s.get("fold"))
        r["bin"] = prevalence_bin(s.get("realised_prevalence")) if s["kind"] != "natural" else "nat"
        if s["kind"] == "lp_gotham" and s.get("p") is None:
            r["bin"] = "nat"
        out.append(r)
    return out


def distinct_streams(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the lowest prevalence target per (kind, fold, realised-prevalence bin).

    Capped LP targets (q = 1) reproduce the lower target's stream exactly (1-1, 34-1, 48-1 at 5%) or nearly
    (3-1, 49-1 at 5%: 1.03% / 1.18% realised): evaluating both would count the same data twice.
    Returns (kept, dropped) with dropped rows carrying 'duplicate_of'."""
    groups: dict = {}
    for r in rows:
        key = (r["kind"], r["fold_short"], r["bin"])
        groups.setdefault(key, []).append(r)
    kept, dropped = [], []
    for key, g in groups.items():
        g = sorted(g, key=lambda r: (r.get("p") if r.get("p") is not None else -1.0, r["stream_id"]))
        kept.append(g[0])
        for r in g[1:]:
            dropped.append(dict(r, duplicate_of=g[0]["stream_id"]))
    kept.sort(key=lambda r: r["stream_id"])
    return kept, dropped


@lru_cache(maxsize=8)
def load_tables(sid: str) -> StreamTables:
    return StreamTables.load(STREAMS / sid)


def pay_index(T: StreamTables) -> np.ndarray:
    return np.nonzero(np.asarray(T.pay, dtype=bool))[0]


# ----------------------------------------------------------------------------------------------- #
# designs and calibration files (P1-D outputs; read only)
# ----------------------------------------------------------------------------------------------- #
@lru_cache(maxsize=256)
def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def design_meta(sid: str, name: str, rho: float) -> dict:
    return json.loads((DESIGNS / sid / f"{name}_{rho:g}.json").read_text())


def design_pi(sid: str, name: str, rho: float, suffix: str = "") -> np.ndarray:
    """π over the payload units (payload order), or an auxiliary array (suffix e.g. '.mask_any')."""
    return np.load(DESIGNS / sid / f"{name}_{rho:g}{suffix}.npy")


def calibration_of(meta: dict) -> tuple[str, str, str]:
    """(fold, seed, pkey) of a design's training calibration ('models/<fold>/<seed>/<pkey>')."""
    _, fold, seed, pkey = meta["calibration"].split("/")
    return fold, seed, pkey


def lamref_entry(fold: str, seed: str, pkey: str, rho: float, dkey: str) -> dict:
    return _read_json(str(MODELS / fold / str(seed) / "lamref.json"))[pkey][f"{rho:g}"][dkey]


def grid_entry(fold: str, seed: str, pkey: str, rho: float, dkey: str) -> dict:
    return _read_json(str(MODELS / fold / str(seed) / "grid.json"))[pkey][f"{rho:g}"][dkey]


def static_spec(meta: dict, gamma: float = 0.3) -> DesignSpec:
    """DesignSpec of a controller-driven static design ('as deployed'; expected-cost controller, no boost)."""
    sp = meta["spec"]
    return DesignSpec(name=meta["design"], design_id=int(sp["design_id"]), phi0=float(sp["phi0"]), Bbar=float(sp["Bbar"]),
                      lam_ref=tuple(float(x) for x in sp["lam_ref"]), kind=sp["kind"], util=sp["util"],
                      tau_mix=float(sp["tau_mix"]), controller="expected", gamma=gamma, rho=float(meta["rho"]))


def adaptive_spec(meta: dict, lam: dict, gamma: float = 0.3) -> DesignSpec:
    """Adaptive arm of a static design: realised-cost controller + boost, λ_ref^A from the boost replay on the
    training streams ('lamref_adaptive'), trigger queries = the fold's boost set (benign firing <= 1e-3)."""
    base = static_spec(meta, gamma)
    if lam.get("lamref_adaptive") is None or not lam.get("boost_queries"):
        raise ValueError(f"{meta['design']}: no adaptive calibration")
    info = lam.get("adaptive_info", {}) or {}
    return replace(base, name=base.name + "_adaptive", design_id=stable_id(base.name + "|adaptive"),
                   controller="realised", boost=True, lam_ref=tuple(float(x) for x in lam["lamref_adaptive"]),
                   boost_queries=tuple(lam["boost_queries"]), kappa_b=float(info.get("kappa_b", 5.0)),
                   W_b=int(info.get("W_b", 20)))


def bm_spec(meta_bm: dict, gamma: float = 0.3) -> DesignSpec:
    """Budget-matched static design: fixed λ for the stream (controller 'none')."""
    sp = meta_bm["spec"]
    name = meta_bm["design"]
    return DesignSpec(name=name, design_id=stable_id(name), phi0=float(sp["phi0"]), Bbar=float(sp["Bbar"]),
                      lam_ref=tuple(float(x) for x in sp["lam"]), kind=sp["kind"], util=sp["util"],
                      tau_mix=float(sp["tau_mix"]), controller="none", quantise_lam=False, gamma=gamma,
                      rho=float(meta_bm["rho"]))


def check_static_reproduces(T: StreamTables, spec: DesignSpec, pi_saved: np.ndarray) -> float:
    """max |π_engine − π_saved| over payload units (engine = sim.montecarlo.static_pass)."""
    from .montecarlo import static_pass
    st = static_pass(T, replace(spec, boost=False, flood_guard=False,
                                controller="expected" if spec.controller == "realised" else spec.controller))
    return float(np.max(np.abs(st["pi_pay"] - np.asarray(pi_saved)))) if len(pi_saved) else 0.0


# ----------------------------------------------------------------------------------------------- #
# estimands
# ----------------------------------------------------------------------------------------------- #
def evaluated_queries(T: StreamTables, min_elig: int = 50) -> list[dict]:
    """Queries of the stream with >= min_elig eligible units (b > 0) in the committed metadata (Z- and
    f-independent rule, DESIGN §5.4). Rows: name, kind, class, n_elig, Q, n_hits, stratum, bbar."""
    rows = []
    for q in range(T.n_queries):
        b = np.asarray(T.b_col(q))
        n_el = int(np.count_nonzero(b > 0))
        if n_el < min_elig:
            continue
        u, f = T.hits(q)
        Q = float(np.sum(f))
        nm = T.query_names[q]
        rows.append({"estimand": nm, "kind": T.query_kinds[q], "class": query_class(nm), "n_elig": n_el,
                     "Q": Q, "n_hits": int(len(u)), "stratum": q_stratum(Q), "bbar": float(T.bbar(q))})
    return rows


def v_ref_value(grid: dict, qname: str, phi0: float, bbar: float = 1.0, epochs: int = 2880) -> tuple[float, str]:
    """(V_ref, source): training per-epoch V x epochs; fallback b̄²/φ0 (one floor-level hit) without training hits."""
    v = (grid.get("V_per_epoch") or {}).get(qname)
    if v is not None and np.isfinite(v) and v > 0:
        return float(v) * epochs, "training"
    bb = bbar if (bbar is not None and np.isfinite(bbar) and bbar > 0) else 1.0
    return float(bb * bb / phi0), "fallback_one_floor_hit"


def build_estimands(T: StreamTables, grid: dict, phi0: float, min_elig: int = 50, family_size: int = 1,
                    epochs: int = 2880) -> tuple[list[Estimand], list[dict]]:
    """Whole-stream estimands of the evaluated queries with the pre-registered V_ref (see module doc)."""
    rows = evaluated_queries(T, min_elig)
    ests = []
    for r in rows:
        V, src = v_ref_value(grid, r["estimand"], phi0, r["bbar"] if r["kind"] == "count" else 1.0, epochs)
        r["V_ref"] = V
        r["V_ref_source"] = src
        ests.append(Estimand(name=r["estimand"], q=r["estimand"], V_ref=V, family_size=int(family_size)))
    return ests, rows


def hiding_capacity_rows(T: StreamTables, pi_pay: np.ndarray, names, alpha_side: float = 0.025) -> dict:
    """Hiding capacity H_{α/2} (Thm. 1) of each indicator query's eligible payload units under π (payload order)."""
    from ..estimation.capacity import hiding_capacity
    pidx = pay_index(T)
    out = {}
    for nm in names:
        q = T.query_index(nm)
        if T.query_kinds[q] != "indicator":
            continue
        b = np.asarray(T.b_col(q))[pidx]
        m = b > 0
        if not m.any():
            continue
        out[nm] = float(hiding_capacity(np.asarray(pi_pay)[m], None, beta=alpha_side)["H"])
    return out


# ----------------------------------------------------------------------------------------------- #
# E1: incidents
# ----------------------------------------------------------------------------------------------- #
def incident_sets(T: StreamTables) -> list[dict]:
    """Per incident (T.incident >= 0 on its content-bearing units): payload positions of the content units,
    k, devices, first/last epoch, and the metadata-defined presence set A = payload units of the incident's
    device(s) inside [first epoch, last epoch] (DESIGN §3.9 'metadata-defined incident set').
    Positions index the payload order (np.nonzero(pay))."""
    pay = np.asarray(T.pay, dtype=bool)
    pidx = np.nonzero(pay)[0]
    upos = np.full(T.n_units, -1, dtype=np.int64)
    upos[pidx] = np.arange(len(pidx))
    inc = np.asarray(T.incident)
    content = np.asarray(T.content, dtype=bool) if T.content is not None else inc >= 0
    ep_pay = np.asarray(T.epoch)[pidx]
    dev_pay = np.asarray(T.device)[pidx]
    eptr = np.searchsorted(ep_pay, np.arange(T.n_epochs + 1))
    sel = np.nonzero((inc >= 0) & content & pay)[0]
    ids, inv = np.unique(inc[sel], return_inverse=True)
    meta_inc = {int(d["incident"]): d for d in (T.extra or {}).get("incidents", [])}
    out = []
    for j, iid in enumerate(ids.tolist()):
        units = sel[inv == j]
        cpos = upos[units]
        devs = np.unique(dev_pay[cpos])
        e0, e1 = int(ep_pay[cpos].min()), int(ep_pay[cpos].max())
        a, b = eptr[e0], eptr[e1 + 1]
        win = np.arange(a, b)
        ppos = win[np.isin(dev_pay[a:b], devs)]
        m = meta_inc.get(int(iid), {})
        out.append({"incident": int(iid), "k": int(len(cpos)), "content_pos": cpos, "presence_pos": ppos,
                    "devices": devs.tolist(), "first_epoch": e0, "last_epoch": e1, "family": m.get("family"),
                    "stratum": k_stratum(len(cpos))})
    return out


def _log_miss(pi, pos):
    p = np.clip(np.asarray(pi, dtype=float)[pos], 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return float(np.log1p(-p).sum())


def recall_from_pi(pi_pay: np.ndarray, incs: list[dict], pi_presence: np.ndarray | None = None) -> list[dict]:
    """Analytic per-incident recall for an independent-Bernoulli design (CtS, B6, B13): content recall
    1 − Π_content(1 − π); presence recall 1 − Π_A(1 − π_presence) (default π_presence = π)."""
    pp = pi_pay if pi_presence is None else pi_presence
    return [{"incident": d["incident"], "content_recall": 1.0 - math.exp(_log_miss(pi_pay, d["content_pos"])),
             "presence_recall": 1.0 - math.exp(_log_miss(pp, d["presence_pos"]))} for d in incs]


def recall_from_masks(mask_full: np.ndarray, mask_any: np.ndarray, incs: list[dict]) -> list[dict]:
    """Deterministic policies: content recall = some content unit retained whole; presence = any byte of a
    unit of A retained (mask_any: e.g. the boundary unit of B2, the cut-off unit of B3)."""
    mf = np.asarray(mask_full, dtype=bool)
    ma = np.asarray(mask_any, dtype=bool)
    return [{"incident": d["incident"], "content_recall": float(mf[d["content_pos"]].any()),
             "presence_recall": float(ma[d["presence_pos"]].any() or mf[d["content_pos"]].any())} for d in incs]


def tte_from_pi(pi_pay: np.ndarray, ep_pay: np.ndarray, incs: list[dict], q: float = 0.5) -> list[float]:
    """Median time to evidence per incident (independent design), in epochs after the incident's first epoch:
    smallest n with P(some content unit of epochs <= first + n retained) >= q; inf if never."""
    out = []
    for d in incs:
        pos = d["content_pos"]
        e = np.asarray(ep_pay)[pos] - d["first_epoch"]
        p = np.clip(np.asarray(pi_pay, dtype=float)[pos], 0, 1)
        with np.errstate(divide="ignore"):
            lm = np.bincount(e, weights=np.log1p(-p), minlength=int(e.max()) + 1)
        surv = np.exp(np.cumsum(lm))
        hit = np.nonzero(1.0 - surv >= q - 1e-15)[0]
        out.append(float(hit[0]) if len(hit) else math.inf)
    return out


def tte_from_mask(mask_full: np.ndarray, ep_pay: np.ndarray, incs: list[dict]) -> list[float]:
    mf = np.asarray(mask_full, dtype=bool)
    out = []
    for d in incs:
        pos = d["content_pos"]
        r = pos[mf[pos]]
        out.append(float(np.asarray(ep_pay)[r].min() - d["first_epoch"]) if len(r) else math.inf)
    return out


def benign_bytes(T: StreamTables, kept_bytes_pay: np.ndarray) -> dict:
    """Expected (or realised) retained bytes split by label: benign (label 0), malicious (> 0), unknown (< 0)."""
    pidx = pay_index(T)
    lab = np.asarray(T.label)[pidx] if T.label is not None else np.zeros(len(pidx))
    kb = np.asarray(kept_bytes_pay, dtype=float)
    return {"benign": float(kb[lab == 0].sum()), "malicious": float(kb[lab > 0].sum()),
            "unknown": float(kb[lab < 0].sum()), "total": float(kb.sum())}


def natural_estimator_stats(T: StreamTables, names, policy: str, pi_pay=None, mask_full=None,
                            scale_fraction: bool = False, hit_p: dict | None = None) -> list[dict]:
    """Bias and RMSE of a policy's natural estimator per query (analytic, over the policy's randomness).

    policy 'ht'  : Horvitz-Thompson Σ Z f/π (CtS designs, B8, B10, B13): bias 0, Var = Σ f²(1−π)/π.
    policy 'sum' : sum of retained f (B2, B3, B4, B6): mean Σ π f, Var = Σ f² π(1−π) (0 for masks).
    scale_fraction (B4 variant): sum of retained f ÷ retained fraction of eligible units (deterministic).
    Units without π (mask policies) use π = mask. ``hit_p[q]`` (rev1, byte-level B3): per-hit retention
    probability aligned with T.hits(q), overriding π of the hit's unit (a hit is kept iff its packet is)."""
    pidx = pay_index(T)
    upos = np.full(T.n_units, -1, dtype=np.int64)
    upos[pidx] = np.arange(len(pidx))
    if pi_pay is None:
        pi_pay = np.asarray(mask_full, dtype=float)
    pi_pay = np.asarray(pi_pay, dtype=float)
    out = []
    for nm in names:
        q = T.query_index(nm)
        u, f = T.hits(q)
        pos = upos[u]
        if np.any(pos < 0):
            raise ValueError("hit on a zero-payload unit")
        p = pi_pay[pos] if hit_p is None or nm not in hit_p else np.asarray(hit_p[nm], dtype=float)
        Q = float(f.sum())
        if policy == "ht":
            if np.any(p <= 0):
                var, bias = math.inf, 0.0
            else:
                var, bias = float(np.sum(f * f * (1 - p) / p)), 0.0
        elif policy == "sum":
            mean = float(np.sum(p * f))
            var = float(np.sum(f * f * p * (1 - p)))
            bias = mean - Q
            if scale_fraction:
                b = np.asarray(T.b_col(q))[pidx]
                el = b > 0
                frac = float(np.sum(pi_pay[el])) / max(int(el.sum()), 1)
                if frac > 0:
                    bias = mean / frac - Q
                    var = var / frac ** 2
                else:
                    bias = -Q
        else:
            raise ValueError(policy)
        out.append({"estimand": nm, "Q": Q, "bias": bias, "var": var, "rmse": math.sqrt(bias * bias + var)})
    return out


def spend_stats(T: StreamTables, kept_bytes_pay: np.ndarray, Bbar: float) -> dict:
    """spend/target and max |D|/B̄ of a (expected or realised) per-unit retained-bytes vector."""
    pidx = pay_index(T)
    ep = np.asarray(T.epoch)[pidx]
    E = T.n_epochs
    cost = np.bincount(ep, weights=np.asarray(kept_bytes_pay, dtype=float), minlength=E)
    D = np.cumsum(Bbar - cost)
    return {"spend_over_target": float(cost.sum() / (Bbar * E)), "max_abs_D_over_Bbar": float(np.max(np.abs(D)) / Bbar)}


# ----------------------------------------------------------------------------------------------- #
# E1 adaptive arm: exact full-unit simulation (every payload unit gets a uniform)
# ----------------------------------------------------------------------------------------------- #
class AdaptiveFullSim:
    """Adaptive CtS design (realised-cost controller + boost) simulated with a uniform for EVERY payload unit,
    so that per-incident retention, time to evidence, retained bytes by label and HT estimates are exact
    per rep (the MC engine of P1-C draws uniforms for hit units only and approximates the non-hit cost).
    Same components as the engine (HorizonController, BoostState, design_pi with committed λ, â, t16, τ^φ).
    Uniforms: Generator(Philox(key=SeedSequence([stream_id, design_id, rep]))) with counter word 2 = 7
    (a purpose not used by the engine), one block per epoch: counter = [0, e, 7, 0]."""

    WORD2 = 7

    def __init__(self, T: StreamTables, design: DesignSpec, queries=(), incs=None):
        from ..design.probs import committed_floor, committed_mix, quantise_utility, sqrt_eff
        self.T = T
        self.d = design
        pidx = pay_index(T)
        self.pidx = pidx
        self.ep = np.asarray(T.epoch)[pidx].astype(np.int64)
        self.pptr = np.searchsorted(self.ep, np.arange(T.n_epochs + 1)).astype(np.int64)
        self.s = np.asarray(T.s, dtype=float)[pidx]
        self.sq = sqrt_eff(self.s)
        self.a = quantise_utility(np.asarray(T.util(design.util), dtype=float)[pidx]) if design.util else np.ones(len(pidx))
        self.tmix = committed_mix(design.tau_mix)
        if design.floor_mode != "constant":
            raise ValueError("full-unit simulator: constant floor only")
        self.phi = np.full(len(pidx), committed_floor(design.phi0))
        self.dev = np.asarray(T.device)[pidx].astype(np.int64)
        lab = np.asarray(T.label)[pidx] if T.label is not None else np.zeros(len(pidx))
        self.lab = lab
        upos = np.full(T.n_units, -1, dtype=np.int64)
        upos[pidx] = np.arange(len(pidx))
        self.upos = upos
        trig = np.zeros(len(pidx), bool)
        if design.boost:
            qs = range(T.n_queries) if design.boost_queries is None else [T.query_index(q) for q in design.boost_queries]
            for q in qs:
                trig[upos[T.hits(q)[0]]] = True
        self.trig = trig
        self.queries = list(queries)
        self.fq = []
        for nm in self.queries:
            u, f = T.hits(T.query_index(nm))
            self.fq.append((upos[u], np.asarray(f, dtype=float)))
        self.incs = incs or []

    def run(self, reps, stream_id: int, chunk: int = 50) -> dict:
        reps = np.asarray(reps, dtype=np.int64)
        outs = [self._chunk(reps[i:i + chunk], stream_id) for i in range(0, len(reps), chunk)]
        return {k: np.concatenate([o[k] for o in outs], axis=0) for k in outs[0]}

    def _chunk(self, reps, stream_id):
        from ..design.boost import BoostState
        from ..design.controller import HorizonController
        from ..design.probs import design_pi, n_lams
        from .montecarlo import philox_epoch_uniforms, rep_keys
        d = self.d
        R = len(reps)
        E = self.T.n_epochs
        n_e = np.diff(self.pptr)
        keys = rep_keys(stream_id, d.design_id, reps)
        C = n_lams(d.kind)
        realised = d.controller == "realised"
        ctrl = HorizonController(np.asarray(d.lam_ref, float), d.Bbar, d.kappa, d.Wc, d.L,
                                 "realised" if realised else "expected", shape=(R,), quantise=d.quantise_lam)
        boost = BoostState(self.T.n_devices, (R,), d.kappa_b, d.W_b) if d.boost else None
        npay = len(self.s)
        Zall = np.zeros((R, npay), dtype=bool)
        pi_all = np.zeros((R, npay), dtype=np.float64)
        boosted = np.zeros(R)
        for e in range(E):
            a0, a1 = self.pptr[e], self.pptr[e + 1]
            if a1 == a0:
                ctrl.update(np.zeros(R))
                if boost is not None:
                    boost.begin_epoch(e)
                    boost.record_units(np.zeros((R, 0), bool), np.zeros(0, bool), np.zeros(0, np.int64))
                continue
            cnt = np.zeros(E, dtype=np.int64)
            cnt[e] = a1 - a0                     # counter [0, e, WORD2, 0] (only epoch e's blocks are computed)
            U = philox_epoch_uniforms(keys, cnt, word2=self.WORD2)
            lam = ctrl.lam()
            bm = None
            if boost is not None:
                boost.begin_epoch(e)
                bm = boost.multiplier(self.dev[a0:a1])
                boosted += (bm > 1).mean(axis=1)
            lams = [np.asarray(lam, float)[..., c:c + 1] for c in range(C)]
            p = design_pi(d.kind, lams, self.a[a0:a1][None, :], self.s[a0:a1][None, :], self.phi[a0:a1][None, :],
                          self.tmix, boost=bm, sqrt_s=self.sq[a0:a1][None, :])
            p = np.broadcast_to(p, (R, a1 - a0))
            Z = U < p
            Zall[:, a0:a1] = Z
            pi_all[:, a0:a1] = p
            cost = Z.astype(float) @ self.s[a0:a1] if realised else p @ self.s[a0:a1]
            ctrl.update(cost)
            if boost is not None:
                tr = self.trig[a0:a1]
                boost.record_units(Z[:, tr], np.ones(int(tr.sum()), bool), self.dev[a0:a1][tr])
        return self._summarise(Zall, pi_all, ctrl, boosted / max(E, 1))

    def _summarise(self, Zall, pi_all, ctrl, boosted):
        R = Zall.shape[0]
        kept = Zall * self.s[None, :]
        out = {"spend_ratio": np.asarray(ctrl.spend_ratio(), float).reshape(R),
               "max_abs_D_over_B": np.asarray(ctrl.max_abs_D, float).reshape(R) / self.d.Bbar,
               "boosted_frac": boosted,
               "benign_bytes": kept[:, self.lab == 0].sum(axis=1),
               "malicious_bytes": kept[:, self.lab > 0].sum(axis=1),
               "total_bytes": kept.sum(axis=1)}
        nI = len(self.incs)
        cont = np.zeros((R, nI), bool)
        pres = np.zeros((R, nI), bool)
        tte = np.full((R, nI), np.inf)
        for j, dd in enumerate(self.incs):
            zc = Zall[:, dd["content_pos"]]
            cont[:, j] = zc.any(axis=1)
            pres[:, j] = Zall[:, dd["presence_pos"]].any(axis=1) | cont[:, j]
            e = self.ep[dd["content_pos"]] - dd["first_epoch"]
            first = np.where(zc, e[None, :], np.iinfo(np.int64).max).min(axis=1)
            tte[:, j] = np.where(first < np.iinfo(np.int64).max, first, np.inf)
        out["content"] = cont
        out["presence"] = pres
        out["tte"] = tte
        Qh = np.zeros((R, len(self.queries)))
        for k, (pos, f) in enumerate(self.fq):
            if len(pos):
                z = Zall[:, pos]
                p = pi_all[:, pos].astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    Qh[:, k] = np.where(z, f[None, :] / np.where(z, p, 1.0), 0.0).sum(axis=1)
        out["Qhat"] = Qh
        return out


# ----------------------------------------------------------------------------------------------- #
# E3: analytic variances
# ----------------------------------------------------------------------------------------------- #
def ht_var_by_query(T: StreamTables, pi_pay: np.ndarray, names) -> dict:
    """Var(Q̂_q) = Σ f²(1−π)/π over the hit units (P1-C's corrected formula; DESIGN's /π² is wrong)."""
    pidx = pay_index(T)
    upos = np.full(T.n_units, -1, dtype=np.int64)
    upos[pidx] = np.arange(len(pidx))
    pi_pay = np.asarray(pi_pay, dtype=float)
    out = {}
    for nm in names:
        u, f = T.hits(T.query_index(nm))
        p = pi_pay[upos[u]]
        if np.any(p <= 0):
            out[nm] = math.inf
            continue
        out[nm] = float(np.sum(f * f * (1.0 - p) / p))
    return out


def ppi_var_by_query(T: StreamTables, pi_pay: np.ndarray, ghat_pay: dict) -> dict:
    """Var of the PPI difference estimator Σ (f − ĝ)²(1−π)/π over ALL payload units (ĝ > 0 units matter)."""
    pidx = pay_index(T)
    upos = np.full(T.n_units, -1, dtype=np.int64)
    upos[pidx] = np.arange(len(pidx))
    pi_pay = np.asarray(pi_pay, dtype=float)
    out = {}
    for nm, g in ghat_pay.items():
        u, f = T.hits(T.query_index(nm))
        fp = np.zeros(len(pidx))
        fp[upos[u]] = f
        dlt = fp - np.asarray(g, dtype=float)
        m = dlt != 0
        if np.any(pi_pay[m] <= 0):
            out[nm] = math.inf
            continue
        out[nm] = float(np.sum(dlt[m] ** 2 * (1.0 - pi_pay[m]) / pi_pay[m]))
    return out


def spec_dict(spec: DesignSpec) -> dict:
    return asdict(spec)
