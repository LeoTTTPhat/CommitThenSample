"""Alg. 2 (DESIGN §3.3): label-free candidate links C ⊆ U × U.

Relations (unit u earlier, v later):
  PIVOT       ∃ h ∈ dst_u with h = src_v
  REVERSE     src_v ∈ dst_u and src_u ∈ dst_v
  SAME_ACTOR  src_u = src_v
  SAME_VICTIM dst_u ∩ dst_v ≠ ∅
  SHARED_EXT  a common external endpoint (src or dst) or INFRA host, DNS name, HTTP host or TLS SNI
INFRA hosts (and multicast/broadcast addresses) are collapsed: they never witness PIVOT,
REVERSE or SAME_VICTIM, only SHARED_EXT (as "SHARED_EXT-via-INFRA"); SAME_ACTOR is literal.
DNS names, HTTP hosts and SNIs share one lower-cased name namespace.

1  per relation: entity keys per unit; join on key; keep t^s_u < t^s_v ≤ t^e_u + W
2  union over relations → rel_mask
3  EXCLUDE self-continuation (same src ∧ same svc_class ∧ same type) → C_cont
4  prune_generic: drop pairs whose relations ⊆ {SAME_ACTOR, SHARED_EXT-via-INFRA only} and whose
   two types are benign-protocol types
5  per v and relation: keep the P predecessors with the smallest lag (tie: unit-id hash)
6  if |C| > N_max: drop largest-lag candidates first; logged
lag ≡ lag_gap = t^s_v − t^e_u (negative when u is still active at t^s_v): the most recently
active predecessor ranks first, so long-lived C2 / scan units are not crowded out by short ones.

G1 fallback levers (default off; dev pool only): lever_P5 (P → 5), lever_victim_seed
(SAME_VICTIM-only pairs kept only if u or v touches a seed), lever_prune_all (the benign-type
prune applied whatever the relations).

Shared entity (CANDIDATE_SCHEMA): witness of the highest-priority relation
(PIVOT, REVERSE, SAME_VICTIM, SHARED_EXT external/name, SAME_ACTOR, SHARED_EXT INFRA),
lowest degree first; kind ∈ {internal, external, infra}; degree = number of distinct peers
of a host over the incident's units (fan-out sources: at least their n_dst), number of
distinct unit sources for a name. shared_is_seed = some host witness of the pair is a seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import polars as pl

from certichain.io.schema import (CANDIDATE_SCHEMA, REL_NAMES, REL_PIVOT, REL_REVERSE, REL_SAME_ACTOR,
                                  REL_SAME_VICTIM, REL_SHARED_EXT, conform)
from certichain.units.sessionize import classify_ips, id_hash
from certichain.units.unit_types import BENIGN_PROTOCOL_TYPES

REL_BITS = {v: k for k, v in REL_NAMES.items()}
_PRIO = {"PIVOT": 1, "REVERSE": 2, "SAME_VICTIM": 3, "SHARED_EXT_OTHER": 4, "SAME_ACTOR": 5, "SHARED_EXT_INFRA": 6}

CONT_SCHEMA = {"u": pl.Int64, "v": pl.Int64, "rel_mask": pl.Int32, "lag_start": pl.Float64, "lag_gap": pl.Float64}


@dataclass(frozen=True)
class CandParams:
    W: float = 1800.0
    P: int = 10
    N_max: int = 30_000
    prune_generic: bool = True
    relations: tuple[str, ...] = ("PIVOT", "REVERSE", "SAME_ACTOR", "SAME_VICTIM", "SHARED_EXT")
    lever_P5: bool = False
    lever_victim_seed: bool = False
    lever_prune_all: bool = False
    max_join_rows: int = 20_000_000

    @property
    def P_eff(self) -> int:
        return min(self.P, 5) if self.lever_P5 else self.P

    @classmethod
    def from_config(cls, cfg: dict | None) -> "CandParams":
        cfg = dict(cfg or {})
        lev = cfg.pop("g1_levers", {}) or {}
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if k == "relations" else v) for k, v in cfg.items() if k in names}
        kw.update({"lever_P5": bool(lev.get("P5", False)),
                   "lever_victim_seed": bool(lev.get("victim_seed_only", False)),
                   "lever_prune_all": bool(lev.get("prune_benign_all", False))})
        return cls(**kw)


@dataclass
class CandidateResult:
    cands: pl.DataFrame                 # CANDIDATE_SCHEMA
    cont: pl.DataFrame                  # self-continuation pairs C_cont (CONT_SCHEMA)
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers
def _join_window(left: pl.DataFrame, right: pl.DataFrame, key: str, W: float, max_rows: int) -> pl.DataFrame:
    """Join left (u, key, ts_u, te_u, ...) with right (v, key, ts_v, ...) on key, keeping
    t^s_u < t^s_v ≤ t^e_u + W. Chunks the right side if the join would be too large."""
    if left.height == 0 or right.height == 0:
        return pl.DataFrame()
    cnt = left.group_by(key).agg(pl.len().alias("_c"))
    r = right.join(cnt, on=key, how="inner")
    if r.height == 0:
        return pl.DataFrame()
    total = int(r["_c"].sum())

    def one(rr: pl.DataFrame) -> pl.DataFrame:
        j = left.join(rr.drop("_c"), on=key, how="inner")
        return j.filter((pl.col("ts_u") < pl.col("ts_v")) & (pl.col("ts_v") <= pl.col("te_u") + W))

    if total <= max_rows:
        return one(r)
    r = r.sort("v").with_columns(pl.col("_c").cum_sum().alias("_cs"))
    r = r.with_columns((pl.col("_cs") // max_rows).alias("_chunk"))
    outs = [one(part.drop("_cs", "_chunk")) for part in r.partition_by("_chunk", maintain_order=True)]
    return pl.concat([o for o in outs if o.height], how="vertical") if any(o.height for o in outs) else outs[0]


def _empty_result(info: dict) -> CandidateResult:
    return CandidateResult(cands=pl.DataFrame(schema=CANDIDATE_SCHEMA), cont=pl.DataFrame(schema=CONT_SCHEMA),
                           info=info)


# --------------------------------------------------------------------------- entities (shared with gt/semantics)
def entity_tables(units: pl.DataFrame, scope) -> dict:
    """The relation entities of every unit (the ONE definition, used by Alg. 2 and by the GT's
    relation-aware enab): D = (unit_id, src, h, internal, unicast, ok) for each stored dst h;
    Sx = (unit_id, h = src, internal, unicast, ok); ok = unicast and not INFRA (may witness
    PIVOT/REVERSE/SAME_VICTIM); names = (unit_id, src, name); SHARED_EXT keys:
    ext = external unicast non-INFRA endpoints, inf = INFRA tokens, nm = names."""
    infra_ips = list(dict(scope.infra))
    U = units.select("unit_id", "ts", "te", "src", "svc_class", "type", "dsts", "infra", "n_dst",
                     "dns_names", "http_hosts", "tls_snis")
    ips = set(U["src"].to_list()) | set(U.select(pl.col("dsts").explode().drop_nulls())["dsts"].to_list())
    ipt = classify_ips(ips, scope.internal_prefixes).select("ip", "internal", "unicast")
    D = (U.select("unit_id", "src", pl.col("dsts").alias("h")).explode("h").drop_nulls("h")
         .join(ipt.rename({"ip": "h"}), on="h", how="left"))
    D = D.with_columns((pl.col("unicast").fill_null(False) & ~pl.col("h").is_in(infra_ips)).alias("ok"))
    Sx = (U.select("unit_id", pl.col("src").alias("h"))
          .join(ipt.rename({"ip": "h"}), on="h", how="left"))
    Sx = Sx.with_columns((pl.col("unicast").fill_null(False) & ~pl.col("h").is_in(infra_ips)).alias("ok"))
    names = pl.concat([U.select("unit_id", "src", pl.col(c).alias("name")).explode("name")
                       for c in ("dns_names", "http_hosts", "tls_snis")]).drop_nulls("name").unique(["unit_id", "name"])
    ep = pl.concat([D.select("unit_id", "h", "internal", "unicast"), Sx.select("unit_id", "h", "internal", "unicast")])
    ext = (ep.filter(~pl.col("internal").fill_null(False) & pl.col("unicast").fill_null(False)
                     & ~pl.col("h").is_in(infra_ips)).select("unit_id", "h").unique())
    inf = U.select("unit_id", pl.col("infra").alias("tok")).explode("tok").drop_nulls("tok").unique()
    return {"U": U, "ipt": ipt, "D": D, "Sx": Sx, "names": names, "ext": ext, "inf": inf, "infra_ips": infra_ips}


@dataclass
class EntityIndex:
    src: dict            # unit_id → src
    src_h: dict          # unit_id → src if it may witness host relations (unicast, not INFRA) else None
    dst_h: dict          # unit_id → frozenset of stored dsts that may witness host relations
    ext: dict            # unit_id → frozenset of SHARED_EXT keys ("ip:…", "infra:…", "name:…")


def entity_index(units: pl.DataFrame, scope) -> EntityIndex:
    """Per-unit entity sets from ``entity_tables`` (Python sets, for pair look-ups)."""
    t = entity_tables(units, scope)
    ids = t["U"]["unit_id"].to_list()
    src = dict(zip(ids, t["U"]["src"].to_list()))
    src_h = {u: (h if ok else None) for u, h, ok in t["Sx"].select("unit_id", "h", "ok").iter_rows()}
    dst_h: dict = {u: set() for u in ids}
    for u, h in t["D"].filter(pl.col("ok")).select("unit_id", "h").iter_rows():
        dst_h[u].add(h)
    ext: dict = {u: set() for u in ids}
    for u, h in t["ext"].iter_rows():
        ext[u].add("ip:" + h)
    for u, tok in t["inf"].iter_rows():
        ext[u].add("infra:" + tok)
    for u, _, n in t["names"].iter_rows():
        ext[u].add("name:" + n)
    return EntityIndex(src=src, src_h=src_h, dst_h={u: frozenset(v) for u, v in dst_h.items()},
                       ext={u: frozenset(v) for u, v in ext.items()})


def shares_entity(ix: EntityIndex, u: int, v: int) -> bool:
    """True iff (u earlier, v later) would satisfy ≥ 1 of PIVOT / REVERSE / SAME_ACTOR / SAME_VICTIM /
    SHARED_EXT under Alg. 2's entity definitions (no time window; REVERSE implies PIVOT)."""
    if u not in ix.src or v not in ix.src:
        return False
    sv = ix.src_h.get(v)
    if sv is not None and sv in ix.dst_h[u]:
        return True                                   # PIVOT (and REVERSE)
    if ix.src[u] == ix.src[v]:
        return True                                   # SAME_ACTOR
    if ix.dst_h[u] & ix.dst_h[v]:
        return True                                   # SAME_VICTIM
    return bool(ix.ext[u] & ix.ext[v])                # SHARED_EXT


# --------------------------------------------------------------------------- main
def generate_candidates(units: pl.DataFrame, scope, params: CandParams = CandParams(),
                        side: pl.DataFrame | None = None) -> CandidateResult:
    """Alg. 2 on one incident's units (UNIT_SCHEMA) with its ScopeResult (INFRA, seeds)."""
    info: dict = {"n_units": units.height, "W": params.W, "P": params.P_eff, "N_max": params.N_max,
                  "prune_generic": params.prune_generic, "levers": {"P5": params.lever_P5,
                  "victim_seed_only": params.lever_victim_seed, "prune_benign_all": params.lever_prune_all}}
    if units.height < 2:
        info.update({"n_union": 0, "n_cont": 0, "n_pruned": 0, "n_final": 0, "N_max_hit": False})
        return _empty_result(info)
    W = float(params.W)
    seeds = set(scope.seeds)
    infra = dict(scope.infra)
    infra_ips = list(infra)
    seed_l = list(seeds)

    ET = entity_tables(units, scope)
    U, ipt = ET["U"], ET["ipt"]
    T = U.select(pl.col("unit_id"), pl.col("ts"), pl.col("te"))
    Tu = T.rename({"unit_id": "u", "ts": "ts_u", "te": "te_u"})
    Tv = T.select(pl.col("unit_id").alias("v"), pl.col("ts").alias("ts_v"))

    # ---- entity tables (shared definition: entity_tables)
    D, Sx = ET["D"], ET["Sx"]

    # degrees
    e = pl.concat([D.select(pl.col("src").alias("ip"), pl.col("h").alias("peer")),
                   D.select(pl.col("h").alias("ip"), pl.col("src").alias("peer"))])
    deg = e.group_by("ip").agg(pl.col("peer").n_unique().alias("deg"))
    fan = U.group_by("src").agg(pl.col("n_dst").max().alias("nd")).rename({"src": "ip"})
    deg = (deg.join(fan, on="ip", how="full", coalesce=True)
           .select("ip", pl.max_horizontal(pl.col("deg").fill_null(0), pl.col("nd").fill_null(0)).alias("deg")))
    deg_map = dict(zip(deg["ip"].to_list(), deg["deg"].to_list()))
    names = ET["names"]
    name_deg = names.group_by("name").agg(pl.col("src").n_unique().alias("deg"))
    ip_kind = {ip: ("internal" if it else "external") for ip, it in zip(ipt["ip"].to_list(), ipt["internal"].to_list())}

    def host_kind(col: str) -> pl.Expr:
        return (pl.when(pl.col(col).is_in(infra_ips)).then(pl.lit("infra"))
                .otherwise(pl.col(col).replace_strict(ip_kind, default="external", return_dtype=pl.Utf8)))

    def host_deg(col: str) -> pl.Expr:
        return pl.col(col).replace_strict(deg_map, default=0, return_dtype=pl.Int64)

    rel_rows: list[pl.DataFrame] = []
    raw_counts: dict[str, int] = {}
    want = set(params.relations)
    witness_cols = ["u", "v", "bit", "wprio", "wkey", "wkind", "wdeg", "wseed", "ext_infra"]

    def emit(df: pl.DataFrame, rel: str, prio_name: str, key_col: str, kind: pl.Expr, degx: pl.Expr,
             seed_expr: pl.Expr, ext_infra: pl.Expr) -> None:
        if df.height == 0:
            raw_counts[rel] = raw_counts.get(rel, 0)
            return
        out = df.select(pl.col("u"), pl.col("v"), pl.lit(REL_BITS[rel]).cast(pl.Int32).alias("bit"),
                        pl.lit(_PRIO[prio_name]).cast(pl.Int32).alias("wprio"),
                        pl.col(key_col).cast(pl.Utf8).alias("wkey"), kind.alias("wkind"),
                        degx.cast(pl.Int64).alias("wdeg"), seed_expr.alias("wseed"), ext_infra.alias("ext_infra"))
        raw_counts[rel] = raw_counts.get(rel, 0) + out.select(["u", "v"]).unique().height
        rel_rows.append(out.select(witness_cols))

    # ---- PIVOT / REVERSE: h ∈ dst_u, h = src_v
    piv = pl.DataFrame()
    if want & {"PIVOT", "REVERSE"}:
        left = D.filter(pl.col("ok")).select(pl.col("unit_id").alias("u"), "h").join(Tu, on="u")
        right = Sx.filter(pl.col("ok")).select(pl.col("unit_id").alias("v"), "h").join(Tv, on="v")
        piv = _join_window(left, right, "h", W, params.max_join_rows)
        if piv.height:
            piv = piv.unique(["u", "v"])  # h = src_v is unique per v
            if "PIVOT" in want:
                emit(piv, "PIVOT", "PIVOT", "h", host_kind("h"), host_deg("h"), pl.col("h").is_in(seed_l),
                     pl.lit(False))
            if "REVERSE" in want:
                srcu = U.select(pl.col("unit_id").alias("u"), pl.col("src").alias("src_u"))
                dv = D.filter(pl.col("ok")).select(pl.col("unit_id").alias("v"), pl.col("h").alias("src_u")).unique()
                rev = piv.join(srcu, on="u").join(dv, on=["v", "src_u"], how="semi")
                emit(rev, "REVERSE", "REVERSE", "h", host_kind("h"), host_deg("h"), pl.col("h").is_in(seed_l),
                     pl.lit(False))

    # ---- SAME_ACTOR
    if "SAME_ACTOR" in want:
        left = U.select(pl.col("unit_id").alias("u"), pl.col("src").alias("h")).join(Tu, on="u")
        right = U.select(pl.col("unit_id").alias("v"), pl.col("src").alias("h")).join(Tv, on="v")
        sa = _join_window(left, right, "h", W, params.max_join_rows)
        emit(sa, "SAME_ACTOR", "SAME_ACTOR", "h", host_kind("h"), host_deg("h"), pl.col("h").is_in(seed_l),
             pl.lit(False))

    # ---- SAME_VICTIM
    if "SAME_VICTIM" in want:
        dd = D.filter(pl.col("ok")).select("unit_id", "h").unique()
        left = dd.rename({"unit_id": "u"}).join(Tu, on="u")
        right = dd.rename({"unit_id": "v"}).join(Tv, on="v")
        sv = _join_window(left, right, "h", W, params.max_join_rows)
        emit(sv, "SAME_VICTIM", "SAME_VICTIM", "h", host_kind("h"), host_deg("h"), pl.col("h").is_in(seed_l),
             pl.lit(False))

    # ---- SHARED_EXT
    if "SHARED_EXT" in want:
        ext = (ET["ext"]
               .select("unit_id", pl.concat_str([pl.lit("ip:"), pl.col("h")]).alias("k"),
                       pl.lit("external").alias("kind"), host_deg("h").alias("deg"), pl.lit(False).alias("inf")))
        inf = (ET["inf"]
               .select("unit_id", pl.concat_str([pl.lit("infra:"), pl.col("tok")]).alias("k"),
                       pl.lit("infra").alias("kind"),
                       pl.col("tok").replace_strict({t: deg_map.get(ip, scope.infra_contacts.get(ip, 0))
                                                     for ip, t in infra.items()}, default=0,
                                                    return_dtype=pl.Int64).alias("deg"),
                       pl.lit(True).alias("inf")))
        nm = (names.join(name_deg, on="name", how="left")
              .select("unit_id", pl.concat_str([pl.lit("name:"), pl.col("name")]).alias("k"),
                      pl.lit("external").alias("kind"), pl.col("deg").cast(pl.Int64), pl.lit(False).alias("inf")))
        keys = pl.concat([ext, inf, nm]).unique(["unit_id", "k"])
        if keys.height:
            left = keys.select(pl.col("unit_id").alias("u"), "k", "kind", "deg", "inf").join(Tu, on="u")
            right = keys.select(pl.col("unit_id").alias("v"), "k").join(Tv, on="v")
            se = _join_window(left, right, "k", W, params.max_join_rows)
            if se.height:
                se = se.with_columns(pl.when(pl.col("inf")).then(pl.lit(_PRIO["SHARED_EXT_INFRA"]))
                                     .otherwise(pl.lit(_PRIO["SHARED_EXT_OTHER"])).alias("_p"))
                raw_counts["SHARED_EXT"] = se.select(["u", "v"]).unique().height
                rel_rows.append(se.select(pl.col("u"), pl.col("v"), pl.lit(REL_SHARED_EXT).cast(pl.Int32).alias("bit"),
                                          pl.col("_p").cast(pl.Int32).alias("wprio"), pl.col("k").alias("wkey"),
                                          pl.col("kind").alias("wkind"), pl.col("deg").fill_null(0).alias("wdeg"),
                                          pl.lit(False).alias("wseed"), pl.col("inf").alias("ext_infra")))
            else:
                raw_counts["SHARED_EXT"] = 0
        else:
            raw_counts["SHARED_EXT"] = 0
    info["n_pairs_raw"] = raw_counts
    if not rel_rows:
        info.update({"n_union": 0, "n_cont": 0, "n_pruned": 0, "n_final": 0, "N_max_hit": False})
        return _empty_result(info)

    # ---- 2: union
    allw = pl.concat(rel_rows, how="vertical")
    pairs = allw.group_by(["u", "v"]).agg(
        pl.col("bit").unique().sum().cast(pl.Int32).alias("rel_mask"),
        ((pl.col("bit") == REL_SHARED_EXT) & ~pl.col("ext_infra")).any().alias("ext_other"),
        pl.col("wkind").sort_by(["wprio", "wdeg", "wkey"]).first().alias("shared_entity_kind"),
        pl.col("wdeg").sort_by(["wprio", "wdeg", "wkey"]).first().cast(pl.Int32).alias("shared_entity_deg"),
        pl.col("wseed").any().alias("shared_is_seed"),
    )
    info["n_union"] = pairs.height
    attrs = U.select("unit_id", "ts", "te", "src", "svc_class", "type")
    pairs = (pairs.join(attrs.rename({c: f"{c}_u" for c in attrs.columns if c != "unit_id"}).rename({"unit_id": "u"}), on="u")
             .join(attrs.rename({c: f"{c}_v" for c in attrs.columns if c != "unit_id"}).rename({"unit_id": "v"}), on="v"))
    pairs = pairs.with_columns(
        (pl.col("ts_v") - pl.col("ts_u")).alias("lag_start"),
        (pl.col("ts_v") - pl.col("te_u")).alias("lag_gap"),
        (pl.col("ts_v") <= pl.col("te_u")).alias("overlap"),
    )

    # ---- 3: self-continuation
    cont_mask = ((pl.col("src_u") == pl.col("src_v")) & (pl.col("svc_class_u") == pl.col("svc_class_v"))
                 & (pl.col("type_u") == pl.col("type_v")))
    cont = pairs.filter(cont_mask)
    pairs = pairs.filter(~cont_mask)
    info["n_cont"] = cont.height

    # ---- 4: prune_generic (+ lever iii)
    bp = list(BENIGN_PROTOCOL_TYPES)
    both_bp = pl.col("type_u").is_in(bp) & pl.col("type_v").is_in(bp)
    weak_only = ((pl.col("rel_mask") & ~(REL_SAME_ACTOR | REL_SHARED_EXT)) == 0) & (
        ((pl.col("rel_mask") & REL_SHARED_EXT) == 0) | ~pl.col("ext_other"))
    n0 = pairs.height
    if params.prune_generic:
        pairs = pairs.filter(~(weak_only & both_bp))
    info["n_pruned_generic"] = n0 - pairs.height
    n0 = pairs.height
    if params.lever_prune_all:
        pairs = pairs.filter(~both_bp)
    info["n_pruned_lever_benign_all"] = n0 - pairs.height
    # ---- lever ii: SAME_VICTIM-only pairs must touch a seed
    n0 = pairs.height
    if params.lever_victim_seed:
        if side is not None and "touches_seed" in side.columns:
            ts_ = side.select(pl.col("unit_id"), pl.col("touches_seed"))
        else:
            ts_ = units.select(pl.col("unit_id"), (pl.col("src").is_in(seed_l)
                                                   | pl.col("dsts").list.eval(pl.element().is_in(seed_l)).list.any()
                                                   ).alias("touches_seed"))
        pairs = (pairs.join(ts_.rename({"unit_id": "u", "touches_seed": "tsu"}), on="u", how="left")
                 .join(ts_.rename({"unit_id": "v", "touches_seed": "tsv"}), on="v", how="left"))
        pairs = pairs.filter(~((pl.col("rel_mask") == REL_SAME_VICTIM)
                               & ~(pl.col("tsu").fill_null(False) | pl.col("tsv").fill_null(False))))
        pairs = pairs.drop("tsu", "tsv")
    info["n_dropped_lever_victim_seed"] = n0 - pairs.height

    # ---- 5: per v and relation, the P smallest-lag predecessors (tie: unit-id hash of u)
    uh = pl.DataFrame({"u": U["unit_id"], "hu": [id_hash(int(x)) for x in U["unit_id"].to_list()]},
                      schema={"u": pl.Int64, "hu": pl.UInt64})
    pairs = pairs.join(uh, on="u", how="left")
    P = params.P_eff
    keep_keys = []
    for rel in params.relations:
        b = REL_BITS[rel]
        sub = pairs.filter((pl.col("rel_mask") & b) != 0).select("u", "v", "lag_gap", "hu")
        if sub.height == 0:
            continue
        sub = sub.sort(["v", "lag_gap", "hu"]).with_columns(pl.int_range(pl.len()).over("v").alias("_r"))
        keep_keys.append(sub.filter(pl.col("_r") < P).select("u", "v"))
    n0 = pairs.height
    if keep_keys:
        kk = pl.concat(keep_keys).unique()
        pairs = pairs.join(kk, on=["u", "v"], how="semi")
    info["n_dropped_P_cap"] = n0 - pairs.height

    # ---- 6: N_max
    info["N_max_hit"] = pairs.height > params.N_max
    info["n_before_N_max"] = pairs.height
    if pairs.height > params.N_max:
        vh = uh.rename({"u": "v", "hu": "hv"})
        pairs = (pairs.join(vh, on="v", how="left").sort(["lag_gap", "hu", "hv"]).head(params.N_max).drop("hv"))
    pairs = pairs.sort(["v", "u"]).with_row_index("cand_id").with_columns(pl.col("cand_id").cast(pl.Int64))
    cands = conform(pairs, CANDIDATE_SCHEMA)
    info["n_final"] = cands.height
    info["rel_counts_final"] = {name: int(((cands["rel_mask"] & bit) != 0).sum()) for bit, name in REL_NAMES.items()}
    if cands.height:
        info["lag_gap_max"] = float(cands["lag_gap"].max())
        info["lag_start_max"] = float(cands["lag_start"].max())
    cont_df = conform(cont.sort(["v", "u"]), CONT_SCHEMA) if cont.height else pl.DataFrame(schema=CONT_SCHEMA)
    return CandidateResult(cands=cands, cont=cont_df, info=info)


def rel_mask_names(mask: int) -> list[str]:
    return [name for bit, name in sorted(REL_NAMES.items()) if mask & bit]
