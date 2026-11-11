"""The ONE GT labelling function for all sources (DESIGN §2.3) + transition statistics.

For a candidate (u, v) with t^s_u < t^s_v, Y_uv ∈ {1, 0, AMB(-1)}:
  1. ep(u) = ⊥ or ep(v) = ⊥, or camp(u) ≠ camp(v)                     ⇒ 0
     (MIXED units — uncertain episode — are not ⊥: their campaign is used when known)
  -  u or v MIXED                                                       ⇒ AMB
  2. ep(u) = ep(v)                                                      ⇒ AMB
  -  (ep(u), ep(v)) ∈ gt.amb_pairs (source-specific, e.g. Gotham cracker) ⇒ AMB
  3. ep(u) → ep(v) ∈ T*, or ep(u) ⇝ ep(v) through hidden intermediates only (bridge):
       1 iff u = enab(ep(u), ep(v)) and v = first(ep(v)), else AMB
       enab(A, B) = RELATION-AWARE (coordinator decision, 18 Oct 19:52): the last unit of A starting
       strictly before first(B) that shares an entity with first(B) — the pair would satisfy ≥ 1 of
       PIVOT/REVERSE/SAME_ACTOR/SAME_VICTIM/SHARED_EXT under Alg. 2's entity definitions
       (candidates.generate.shares_entity; no time window); if no such unit exists, the DESIGN rule:
       the last unit of A starting strictly before first(B) (order (t^s, unit_id)). Bridges use the
       same enab. Without the entity columns (minimal unit frames) the DESIGN rule is used.
  4. ep(u) ⇝ ep(v) through ≥ 1 observed intermediate                     ⇒ AMB
  5. otherwise (same campaign, no directed path)                         ⇒ 0
     (⇒ AMB instead for campaigns in gt.amb_unrelated_campaigns: IoT-23's "malicious pairs not
      covered by R1/R2 are AMB")
An episode is *hidden* in an incident if it has no unit there (declared hidden, or all its
records fell outside scope / U_max); T*_hid = transitions with a hidden endpoint.

``read_quarantined`` is the single function through which gt/* reads quarantined label/GT
files (P2-A's io.labels.read_labels_QUARANTINED when available).
"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import polars as pl

from certichain.io.schema import PAIR_LABEL_SCHEMA

AMB = -1
PROJECT = Path(__file__).resolve().parents[3]          # P2_CertiChain/


@dataclass
class EpisodeGT:
    unit_episode: dict[int, str]                       # unit_id → episode id (absent ⇒ ⊥ unless MIXED)
    unit_campaign: dict[int, str | None]               # unit_id → campaign (episode and MIXED units)
    transitions: set[tuple[str, str]]                  # T*: direct transitions over episodes
    hidden: set[str] = field(default_factory=set)      # episodes declared hidden (no network evidence)
    episode_campaign: dict[str, str] = field(default_factory=dict)
    mixed_units: set[int] = field(default_factory=set)
    amb_pairs: set[tuple[str, str]] = field(default_factory=set)
    amb_unrelated_campaigns: set[str] = field(default_factory=set)
    meta: dict = field(default_factory=dict)

    def campaign_of(self, uid: int) -> str | None:
        c = self.unit_campaign.get(uid)
        if c is None and uid in self.unit_episode:
            c = self.episode_campaign.get(self.unit_episode[uid])
        return c

    def to_json(self) -> dict:
        return {"transitions": sorted([list(t) for t in self.transitions]), "hidden": sorted(self.hidden),
                "episode_campaign": dict(sorted(self.episode_campaign.items())),
                "amb_pairs": sorted([list(t) for t in self.amb_pairs]),
                "amb_unrelated_campaigns": sorted(self.amb_unrelated_campaigns), "meta": self.meta}

    def units_frame(self) -> pl.DataFrame:
        ids = sorted(set(self.unit_episode) | set(self.mixed_units) | set(self.unit_campaign))
        return pl.DataFrame({"unit_id": ids,
                             "episode": [self.unit_episode.get(i) for i in ids],
                             "campaign": [self.campaign_of(i) for i in ids],
                             "mixed": [i in self.mixed_units for i in ids]},
                            schema={"unit_id": pl.Int64, "episode": pl.Utf8, "campaign": pl.Utf8, "mixed": pl.Boolean})


# --------------------------------------------------------------------------- episode index
ENTITY_COLS = ("unit_id", "ts", "te", "src", "svc_class", "type", "dsts", "infra", "n_dst", "dns_names",
               "http_hosts", "tls_snis")


def _entity_index(units: pl.DataFrame, scope):
    if not all(c in units.columns for c in ENTITY_COLS):
        return None
    from certichain.candidates.generate import entity_index
    from certichain.units.scope import ScopeResult
    sc = scope if scope is not None else ScopeResult(seeds=(), infra={}, infra_contacts={}, hosts={})
    return entity_index(units, sc)


class _Index:
    def __init__(self, units: pl.DataFrame, gt: EpisodeGT, scope=None):
        self._ent_src = (units, scope)
        self._ent = None
        self._enab: dict[tuple[str, str], tuple[int | None, str | None]] = {}
        ts = dict(zip(units["unit_id"].to_list(), units["ts"].to_list()))
        self.ts = ts
        lists: dict[str, list[tuple[float, int]]] = {}
        for uid, ep in gt.unit_episode.items():
            if uid in ts and uid not in gt.mixed_units:
                lists.setdefault(ep, []).append((ts[uid], uid))
        for v in lists.values():
            v.sort()
        self.lists = lists
        self.observed = set(lists)
        nodes = set(gt.hidden) | set(gt.episode_campaign) | {a for a, _ in gt.transitions} | {b for _, b in gt.transitions}
        self.hidden = (nodes | set(gt.hidden)) - self.observed
        succ: dict[str, set[str]] = {}
        for a, b in gt.transitions:
            succ.setdefault(a, set()).add(b)
        self.succ = succ
        self.T = set(gt.transitions)
        self._bridge: dict[str, set[str]] = {}
        self._reach: dict[str, set[str]] = {}

    def first(self, ep: str) -> int | None:
        lst = self.lists.get(ep)
        return lst[0][1] if lst else None

    @property
    def ent(self):
        if self._ent is None and self._ent_src is not None:
            self._ent = _entity_index(*self._ent_src)
            self._ent_src = None
        return self._ent

    def enab_rule(self, a: str, b: str) -> tuple[int | None, str | None]:
        """(enab(A,B), rule) with rule ∈ {"relation", "fallback", None}."""
        key = (a, b)
        if key not in self._enab:
            la, fb = self.lists.get(a), self.first(b)
            out: tuple[int | None, str | None] = (None, None)
            if la and fb is not None:
                t_b = self.ts[fb]
                i = bisect.bisect_left(la, (t_b, -1))      # units of A with ts < t_b: la[:i]
                if i > 0:
                    out = (la[i - 1][1], "fallback")
                    if self.ent is not None:
                        from certichain.candidates.generate import shares_entity
                        for k in range(i - 1, -1, -1):
                            if shares_entity(self.ent, la[k][1], fb):
                                out = (la[k][1], "relation")
                                break
            self._enab[key] = out
        return self._enab[key]

    def enab(self, a: str, b: str) -> int | None:
        return self.enab_rule(a, b)[0]

    def bridges(self, a: str) -> set[str]:
        """Observed episodes reachable from a through ≥ 1 hidden intermediate, all hidden."""
        if a not in self._bridge:
            out: set[str] = set()
            stack = [n for n in self.succ.get(a, ()) if n in self.hidden]
            seen = set(stack)
            while stack:
                n = stack.pop()
                for m in self.succ.get(n, ()):
                    if m in self.hidden:
                        if m not in seen:
                            seen.add(m)
                            stack.append(m)
                    elif m != a:
                        out.add(m)
            self._bridge[a] = out
        return self._bridge[a]

    def reach(self, a: str) -> set[str]:
        if a not in self._reach:
            seen: set[str] = set()
            stack = list(self.succ.get(a, ()))
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                stack.extend(self.succ.get(n, ()))
            self._reach[a] = seen
        return self._reach[a]


REASONS = ("0:background", "0:cross_campaign", "AMB:mixed", "AMB:same_episode", "AMB:explicit", "1:direct",
           "1:bridge", "AMB:not_enabling_pair", "AMB:transitive", "0:no_path", "AMB:unrelated_in_campaign")


def label_pair_reason(u: int, v: int, gt: EpisodeGT, ix: _Index) -> tuple[int, bool, str]:
    eu, ev = gt.unit_episode.get(u), gt.unit_episode.get(v)
    mu, mv = u in gt.mixed_units, v in gt.mixed_units
    if (eu is None and not mu) or (ev is None and not mv):
        return 0, False, "0:background"                             # rule 1 (⊥)
    cu, cv = gt.campaign_of(u), gt.campaign_of(v)
    if cu is not None and cv is not None and cu != cv:
        return 0, False, "0:cross_campaign"                         # rule 1 (campaigns)
    if mu or mv or cu is None or cv is None:
        return AMB, False, "AMB:mixed"                              # MIXED / unknown campaign
    if eu == ev:
        return AMB, False, "AMB:same_episode"                       # rule 2
    if (eu, ev) in gt.amb_pairs:
        return AMB, False, "AMB:explicit"
    direct = (eu, ev) in ix.T
    if direct or ev in ix.bridges(eu):                               # rule 3
        if u == ix.enab(eu, ev) and v == ix.first(ev):
            return 1, not direct, ("1:direct" if direct else "1:bridge")
        return AMB, False, "AMB:not_enabling_pair"
    if ev in ix.reach(eu):
        return AMB, False, "AMB:transitive"                         # rule 4
    if cu in gt.amb_unrelated_campaigns:
        return AMB, False, "AMB:unrelated_in_campaign"
    return 0, False, "0:no_path"                                    # rule 5


def label_pair(u: int, v: int, gt: EpisodeGT, ix: _Index) -> tuple[int, bool]:
    y, b, _ = label_pair_reason(u, v, gt, ix)
    return y, b


def label_pairs(cands: pl.DataFrame, units: pl.DataFrame, gt: EpisodeGT, scope=None,
                return_reasons: bool = False):
    """PAIR_LABEL_SCHEMA frame (cand_id, y, bridge) for every candidate (``scope``: the incident's
    ScopeResult, for the INFRA collapse in the relation-aware enab; None ⇒ no INFRA). With
    ``return_reasons`` also a dict reason → count (which rule decided each label)."""
    ix = _Index(units, gt, scope)
    ys, bs, rs = [], [], {}
    for u, v in zip(cands["u"].to_list(), cands["v"].to_list()):
        y, b, r = label_pair_reason(int(u), int(v), gt, ix)
        ys.append(y)
        bs.append(b)
        rs[r] = rs.get(r, 0) + 1
    df = pl.DataFrame({"cand_id": cands["cand_id"], "y": pl.Series(ys, dtype=pl.Int8),
                       "bridge": pl.Series(bs, dtype=pl.Boolean)}, schema=PAIR_LABEL_SCHEMA)
    return (df, rs) if return_reasons else df


TRANSITION_SCHEMA = {"src_ep": pl.Utf8, "dst_ep": pl.Utf8, "kind": pl.Utf8, "enab_u": pl.Int64,
                     "first_v": pl.Int64, "in_C": pl.Boolean, "cand_id": pl.Int64, "enab_rule": pl.Utf8}


def transition_table(cands: pl.DataFrame, units: pl.DataFrame, gt: EpisodeGT, scope=None) -> pl.DataFrame:
    """One row per T* transition (kind obs | hid) and per bridge (kind bridge), with its enabling
    unit pair (enab(A,B), first(B)), the enab rule used, and whether that pair is a candidate."""
    ix = _Index(units, gt, scope)
    cmap = {(int(u), int(v)): int(c) for c, u, v in zip(cands["cand_id"].to_list(), cands["u"].to_list(),
                                                        cands["v"].to_list())} if cands.height else {}
    rows = []
    for a, b in sorted(gt.transitions):
        if a in ix.hidden or b in ix.hidden or a not in ix.observed or b not in ix.observed:
            rows.append((a, b, "hid", None, None, False, None, None))
            continue
        (e, rule), f = ix.enab_rule(a, b), ix.first(b)
        cid = cmap.get((e, f)) if e is not None else None
        rows.append((a, b, "obs", e, f, cid is not None, cid, rule))
    for a in sorted(ix.observed):
        for b in sorted(ix.bridges(a)):
            if (a, b) in ix.T or (a, b) in gt.amb_pairs:
                continue
            (e, rule), f = ix.enab_rule(a, b), ix.first(b)
            cid = cmap.get((e, f)) if e is not None else None
            rows.append((a, b, "bridge", e, f, cid is not None, cid, rule))
    if not rows:
        return pl.DataFrame(schema=TRANSITION_SCHEMA)
    return pl.DataFrame(rows, schema=TRANSITION_SCHEMA, orient="row")


def transition_stats(cands: pl.DataFrame, units: pl.DataFrame, gt: EpisodeGT,
                     labels: pl.DataFrame | None = None, scope=None) -> dict:
    """|T*|, |T*_obs|, |T*_hid|, r_{C,T}, ρ_T, bridges, ceiling and label counts (§2.3)."""
    tt = transition_table(cands, units, gt, scope)
    obs = tt.filter(pl.col("kind") == "obs")
    hid = tt.filter(pl.col("kind") == "hid")
    br = tt.filter(pl.col("kind") == "bridge")
    n_T, n_obs, n_hid = obs.height + hid.height, obs.height, hid.height
    n_obs_in_C = int(obs["in_C"].sum()) if n_obs else 0
    r = n_obs_in_C / n_obs if n_obs else None
    rho = n_hid / n_T if n_T else None
    out = {"n_T": n_T, "n_T_obs": n_obs, "n_T_hid": n_hid, "n_T_obs_in_C": n_obs_in_C,
           "n_T_obs_no_enab": int(obs["enab_u"].is_null().sum()) if n_obs else 0,
           "r_CT": r, "rho_T": rho,
           "ceiling": ((1 - rho) * r) if (r is not None and rho is not None) else None,
           "n_bridges": br.height, "n_bridges_in_C": int(br["in_C"].sum()) if br.height else 0,
           "n_E_link": int(obs["enab_u"].is_not_null().sum() if n_obs else 0) + int(br["enab_u"].is_not_null().sum() if br.height else 0),
           "n_T_obs_enab_relation": int((obs["enab_rule"] == "relation").sum()) if n_obs else 0,
           "n_T_obs_enab_fallback": int((obs["enab_rule"] == "fallback").sum()) if n_obs else 0,
           "n_episodes_observed": len({e for u, e in gt.unit_episode.items()
                                       if u in set(units["unit_id"].to_list()) and u not in gt.mixed_units}),
           "n_candidates": cands.height}
    if labels is not None and labels.height:
        y = labels["y"]
        out.update({"n_pos": int((y == 1).sum()), "n_null": int((y == 0).sum()), "n_amb": int((y == AMB).sum()),
                    "n_pos_bridge": int((labels["bridge"] & (y == 1)).sum())})
    return out


# --------------------------------------------------------------------------- quarantined I/O
def _iot23_label_path(scenario: str, window_id: str) -> Path:
    return PROJECT / "results" / "interim" / "iot23" / "labels" / scenario / f"{window_id}.parquet"


def read_quarantined(dataset: str, *, what: str = "labels", scenario: str | None = None,
                     window_id: str | None = None, incident_dir: str | Path | None = None,
                     rec_ids: Iterable[int] | None = None, path: str | Path | None = None):
    """THE single entry point through which gt/* reads quarantined labels / GT files.

    dataset 'iot23'      what='labels'  → LABEL_SCHEMA frame of one window
    dataset 'gotham'     what='labels'  → LABEL_SCHEMA frame (``path`` or scenario/window)
                         what='reports' → decoded report payload table (or None if absent)
    dataset 'chainforge' what='labels'  → incident_dir/labels.parquet
                         what='gt'      → incident_dir/gt.json (dict)
    """
    ds = dataset.lower()
    ids = None if rec_ids is None else pl.Series(list(rec_ids), dtype=pl.Int64)
    if what == "gt":                                   # ChainForge gt.json (quarantined)
        return json.loads((Path(incident_dir) / "gt.json").read_text())
    if what == "reports":                              # Gotham decoded report payloads
        p = Path(path) if path else PROJECT / "results" / "interim" / "gotham" / "reports.parquet"
        return pl.read_parquet(p) if p.exists() else None
    if ds.startswith("chainforge") or ds.startswith("cf_"):
        # extended label table (episode_id, campaign_id, decoy_kind, template_node): read as is,
        # LABEL_SCHEMA conformance would drop the GT columns
        df = pl.read_parquet(Path(incident_dir) / "labels.parquet")
        return df if ids is None else df.filter(pl.col("rec_id").is_in(ids))
    if path is not None:
        p = Path(path)
    elif ds == "iot23":
        p = _iot23_label_path(scenario, window_id)
    else:
        p = PROJECT / "results" / "interim" / ds / "labels" / scenario / f"{window_id}.parquet"
    try:  # P2-A's quarantined reader (LABEL_SCHEMA tables)
        from certichain.io.labels import read_labels_QUARANTINED
    except ImportError:  # pragma: no cover - fallback while io.labels did not exist
        df = pl.read_parquet(p)
        return df if ids is None else df.filter(pl.col("rec_id").is_in(ids))
    return read_labels_QUARANTINED(p, rec_ids=None if ids is None else ids.to_list())


def unit_majority(units: pl.DataFrame, rec_labels: pl.DataFrame, col: str, min_share: float = 0.5,
                  denominator: str = "all") -> pl.DataFrame:
    """Per unit: the value of ``col`` held by ≥ min_share of the unit's records.

    denominator='all'      → share over all records of the unit (null = background counts);
    denominator='labelled' → share over records with a non-null value.
    Returns unit_id, value (None = ⊥ / no labelled record), mixed (bool), n_lab, n."""
    from certichain.units.sessionize import unit_record_map

    m = unit_record_map(units).join(rec_labels.select("rec_id", col), on="rec_id", how="left")
    n = m.group_by("unit_id").agg(pl.len().alias("n"), pl.col(col).is_not_null().sum().alias("n_lab"))
    cnt = (m.filter(pl.col(col).is_not_null()).group_by(["unit_id", col]).agg(pl.len().alias("c"))
           .sort(["unit_id", "c", col], descending=[False, True, False]))
    top = cnt.group_by("unit_id", maintain_order=True).agg(pl.col(col).first().alias("value"),
                                                           pl.col("c").first().alias("c1"),
                                                           pl.col("c").slice(1, 1).first().alias("c2"))
    out = n.join(top, on="unit_id", how="left")
    den = pl.col("n") if denominator == "all" else pl.col("n_lab")
    ok = (pl.col("c1").cast(pl.Float64) >= min_share * den.cast(pl.Float64)) & (
        pl.col("c2").is_null() | (pl.col("c2") < pl.col("c1")))
    out = out.with_columns(
        pl.when(pl.col("n_lab") == 0).then(pl.lit(None, dtype=pl.Utf8))
        .when(ok).then(pl.col("value").cast(pl.Utf8)).otherwise(pl.lit(None, dtype=pl.Utf8)).alias("value"),
        ((pl.col("n_lab") > 0) & ~ok.fill_null(False)).alias("mixed"))
    return out.select("unit_id", "value", "mixed", "n_lab", "n").sort("unit_id")
