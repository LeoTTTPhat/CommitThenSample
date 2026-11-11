"""Alg. 1 step 6 (DESIGN §3.2): the fixed, label-free incident scope rule.

* INFRA = hosts contacted by ≥ F_infra distinct internal hosts in the window (brokers,
  servers, C2/loader/report servers, a shared scanner, multicast/broadcast groups). They are
  never used for expansion, never members of S, and get per-host tokens INFRA<n>
  (numbered by descending internal-contact count, then first contact time, then IP).
  The seed is never INFRA.
* S = {seed} ∪ N1 ∪ N2 with
  N1 = internal unicast non-INFRA hosts that exchanged ≥ 1 record with the seed (either direction),
  N2 = internal unicast non-INFRA hosts that contacted a non-INFRA external endpoint the seed
       also contacted (shared-external expansion; "contacted" = originated a record to).
  hop(seed) = 0, hop(N1 ∪ N2) = 1.
* Keep units with src ∈ S or dst-set ∩ S ≠ ∅ (full dst set, not the ≤ 256 stored dsts).
  If more than U_max remain, keep the U_max closest by (hop, |t − seed activity|), where
  hop(u) = 0 iff u touches a seed and |t − seed activity| is the gap between [t^s_u, t^e_u]
  and the union of the seed units' intervals (0 if overlapping); ties by first_tiehash.

Record-level application (``prefilter_records``) is exactly equivalent to filtering units:
unit formation, typing and merging never mix records of different (src, svc_class) groups,
so a group can only produce a kept unit if src ∈ S or one of its records goes to a host in S.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl


@dataclass
class ScopeResult:
    seeds: tuple[str, ...]
    infra: dict[str, str]                 # ip → "INFRA<n>"
    infra_contacts: dict[str, int]        # ip → #distinct internal hosts contacting it
    hosts: dict[str, int]                 # S: ip → hop
    n1: tuple[str, ...] = ()
    n2: tuple[str, ...] = ()
    internal_prefixes: tuple[str, ...] = ()
    F_infra: int = 3
    info: dict = field(default_factory=dict)

    def infra_index(self) -> dict[str, int]:
        return {ip: int(tok[5:]) for ip, tok in self.infra.items()}

    def to_json(self) -> dict:
        return {"seeds": list(self.seeds), "F_infra": self.F_infra,
                "internal_prefixes": list(self.internal_prefixes),
                "infra": [{"ip": ip, "token": tok, "internal_contacts": self.infra_contacts.get(ip, 0)}
                          for ip, tok in sorted(self.infra.items(), key=lambda kv: int(kv[1][5:]))],
                "scope_hosts": [{"ip": ip, "hop": hop} for ip, hop in sorted(self.hosts.items())],
                "n1": list(self.n1), "n2": list(self.n2), "info": self.info}

    @classmethod
    def from_json(cls, d: dict) -> "ScopeResult":
        return cls(seeds=tuple(d["seeds"]), infra={r["ip"]: r["token"] for r in d["infra"]},
                   infra_contacts={r["ip"]: r["internal_contacts"] for r in d["infra"]},
                   hosts={r["ip"]: r["hop"] for r in d["scope_hosts"]}, n1=tuple(d.get("n1", ())),
                   n2=tuple(d.get("n2", ())), internal_prefixes=tuple(d.get("internal_prefixes", ())),
                   F_infra=d.get("F_infra", 3), info=d.get("info", {}))


def compute_scope(prep: pl.DataFrame, seeds: Sequence[str], F_infra: int = 3,
                  internal_prefixes: Sequence[str] = ()) -> ScopeResult:
    """INFRA and S from prepared records (sessionize.prepare_records output)."""
    seed_l = sorted({s for s in seeds if s})
    cont = (prep.filter(pl.col("src_int"))
            .group_by("dst").agg(pl.col("src").n_unique().alias("n"), pl.col("ts").min().alias("t0"))
            .filter((pl.col("n") >= F_infra) & ~pl.col("dst").is_in(seed_l))
            .sort(["n", "t0", "dst"], descending=[True, False, False]))
    infra = {ip: f"INFRA{i}" for i, ip in enumerate(cont["dst"].to_list())}
    infra_contacts = {ip: int(n) for ip, n in zip(cont["dst"].to_list(), cont["n"].to_list())}
    infra_l = list(infra)
    a = prep.filter(pl.col("src").is_in(seed_l)).select(pl.col("dst").alias("h"), pl.col("dst_int").alias("hi"),
                                                        pl.col("dst_uni").alias("hu"))
    b = prep.filter(pl.col("dst").is_in(seed_l)).select(pl.col("src").alias("h"), pl.col("src_int").alias("hi"),
                                                        pl.col("src_uni").alias("hu"))
    n1 = (pl.concat([a, b]).filter(pl.col("hi") & pl.col("hu") & ~pl.col("h").is_in(seed_l)
                                   & ~pl.col("h").is_in(infra_l))["h"].unique().sort().to_list())
    eseed = (prep.filter(pl.col("src").is_in(seed_l) & ~pl.col("dst_int") & pl.col("dst_uni")
                         & ~pl.col("dst").is_in(infra_l))["dst"].unique().to_list())
    n2 = (prep.filter(pl.col("src_int") & pl.col("src_uni") & ~pl.col("src").is_in(seed_l)
                      & ~pl.col("src").is_in(infra_l) & pl.col("dst").is_in(eseed))["src"].unique().sort().to_list())
    hosts = {s: 0 for s in seed_l}
    for h in n1 + n2:
        hosts.setdefault(h, 1)
    present = set(prep["src"].unique().to_list()) | set(prep["dst"].unique().to_list()) if prep.height else set()
    info = {"n_infra": len(infra), "n_scope_hosts": len(hosts), "n_n1": len(n1), "n_n2": len(n2),
            "n_seed_external_endpoints": len(eseed),
            "seeds_present": [s for s in seed_l if s in present]}
    return ScopeResult(seeds=tuple(seed_l), infra=infra, infra_contacts=infra_contacts, hosts=hosts,
                       n1=tuple(n1), n2=tuple(n2), internal_prefixes=tuple(internal_prefixes),
                       F_infra=F_infra, info=info)


def prefilter_records(prep: pl.DataFrame, sc: ScopeResult) -> pl.DataFrame:
    """Records of (src, svc_class) groups that can yield a kept unit (exact; see module doc)."""
    S = list(sc.hosts)
    g = prep.filter(pl.col("dst").is_in(S)).select("src", "svc_class").unique()
    keys = [f"{a}|{b}" for a, b in g.iter_rows()]
    mask = prep.select((pl.col("src").is_in(S)
                        | pl.concat_str([pl.col("src"), pl.col("svc_class")], separator="|").is_in(keys)
                        ).alias("m"))["m"]
    return prep if bool(mask.all()) else prep.filter(mask)


def annotate_records(prep: pl.DataFrame, sc: ScopeResult | None) -> pl.DataFrame:
    """Per-record scope annotations used by the unit aggregates."""
    from certichain.units.sessionize import default_annotations

    if sc is None:
        return default_annotations(prep)
    seed_l, S = list(sc.seeds), list(sc.hosts)
    infra_l = list(sc.infra)
    out = prep.with_columns(
        pl.col("src").is_in(seed_l).alias("src_is_seed"), pl.col("dst").is_in(seed_l).alias("dst_is_seed"),
        pl.col("src").is_in(S).alias("src_in_S"), pl.col("dst").is_in(S).alias("dst_in_S"),
        pl.col("src").is_in(infra_l).alias("src_infra"), pl.col("dst").is_in(infra_l).alias("dst_infra"),
    )
    return out.with_columns(
        (pl.col("dst_infra") | ~pl.col("dst_int")).alias("dst_infra_or_ext"),
        pl.when(pl.col("dst_in_S")).then(pl.lit(0)).when(pl.col("dst_infra")).then(pl.lit(1))
        .when(pl.col("dst_int")).then(pl.lit(2)).otherwise(pl.lit(3)).cast(pl.Int8).alias("dst_prio"))


def select_units(units: pl.DataFrame, sc: ScopeResult, U_max: int = 1000,
                 use_scope: bool = True) -> tuple[pl.DataFrame, dict]:
    """Unit-level keep rule + U_max cap; adds ``hop`` and ``seed_dist``."""
    from certichain.units.sessionize import interval_distance, merged_intervals

    info = {"n_units_pre_scope": units.height}
    kept = units.filter(pl.col("touches_S")) if use_scope else units
    info["n_units_in_scope"] = kept.height
    kept = kept.with_columns(pl.when(pl.col("touches_seed")).then(pl.lit(0)).otherwise(pl.lit(1))
                             .cast(pl.Int8).alias("hop"))
    seed_u = kept.filter(pl.col("touches_seed"))
    ms, me = merged_intervals(seed_u["ts"].to_numpy().astype(float), seed_u["te"].to_numpy().astype(float))
    dist = interval_distance(kept["ts"].to_numpy().astype(float), kept["te"].to_numpy().astype(float), ms, me)
    kept = kept.with_columns(pl.Series("seed_dist", dist, dtype=pl.Float64))
    info["U_max_hit"] = kept.height > U_max
    if kept.height > U_max:
        info["n_units_dropped_U_max"] = kept.height - U_max
        kept = kept.sort(["hop", "seed_dist", "first_tiehash"]).head(U_max)
    else:
        info["n_units_dropped_U_max"] = 0
    return kept, info
