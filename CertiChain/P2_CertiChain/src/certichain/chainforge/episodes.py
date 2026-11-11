"""ChainForge episode library (DESIGN §5.4.1): (source, attack type, ATT&CK technique(s),
role set, victim set, duration, record pointers, pool) per library episode.

GENERATOR-SIDE ONLY. The library is label-derived (TON ``episodes.parquet`` carries the
attack ``type``; Gotham episodes come from Gotham labels). Nothing here may be read by
feature, scoring, unit, candidate or certification code.

Layout (results/interim/<ds>/library/):
    episodes.parquet          one row per library episode (no records), incl. role slots and
                              per-mode eligibility flags
    records_<pool>.parquet    RECORD_SCHEMA rows of the pool's episodes + ``lib_eid`` (copied
                              from P2-A's records, untouched; original rec_id kept)
    labels_<pool>.parquet     rec_id, label, detailed_label, lib_eid (quarantined)
    library.json              Table-III statistics + build config

TON roles (role slots of a library episode)
    actor   the TON attacker 192.168.1.30-39 (P2-A's ``attacker``)
    victim  the *primary victim*: the internal unicast host with the most records with the
            actor on non-DNS ports (all ports if the episode has no non-DNS record with an
            internal host). Other hosts are secondary victims/third parties and are rewritten by
            the incident's base map (chainforge/rewrite.py).
Eligibility (a node's binding mode decides which flag is needed; chainforge/templates.py):
    victim bound  ⇒ n_primary ≥ MIN_PRIMARY records between actor and primary victim
    actor only    ⇒ n_actor_orig ≥ MIN_ACTOR records originated by the actor (floods FROM a bot)
    backdoor      ⇒ additionally n_victim_orig ≥ 1 (the victim calls the C2: realistic reverse
                    shell, and what puts a D2 parallel infection in scope via shared-external
                    expansion)
"""
from __future__ import annotations

import ipaddress
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from certichain.io.paths import atomic_write_json, atomic_write_parquet, interim, log
from certichain.io.schema import RECORD_SCHEMA, conform

TON_TYPES = ("scanning", "password", "injection", "xss", "backdoor", "ransomware", "dos", "ddos")
ATTACK_TECHNIQUES: dict[str, tuple[str, ...]] = {
    "scanning": ("T1046", "T1595.001"),
    "password": ("T1110.001",),
    "injection": ("T1190",),
    "xss": ("T1190", "T1059.007"),
    "backdoor": ("T1071", "T1571"),
    "ransomware": ("T1486", "T1021.002"),
    "dos": ("T1499",),
    "ddos": ("T1498",),
    # Gotham (filled by the Gotham library builder)
    "mirai_bf": ("T1110.001",), "mirai_itt": ("T1105", "T1059.004"), "mirai_dl": ("T1105",),
    "mirai_cc": ("T1071",), "mirai_scan": ("T1046", "T1595.001"), "mirai_report": ("T1041",),
    "mirai_udpflood": ("T1498",), "mirai_tcpflood": ("T1498",), "mirai_greflood": ("T1498",),
    "merlin_c2": ("T1071.001", "T1573"), "merlin_icmpflood": ("T1498",), "merlin_udpflood": ("T1498",),
    "merlin_tcpflood": ("T1498",), "udp_scan": ("T1046",), "amplification": ("T1498.002",),
    "mirai_dos_cc": ("T1071",), "masscan": ("T1046", "T1595.001"),
}
ROLE_SET = {"scanning": ("actor", "victims"), "password": ("actor", "victim"), "injection": ("actor", "web_victim"),
            "xss": ("actor", "web_victim"), "backdoor": ("c2", "victim"), "ransomware": ("c2", "victim"),
            "dos": ("actor", "targets"), "ddos": ("actor", "targets")}
MIN_PRIMARY = 5
MIN_ACTOR = 5

LIB_SCHEMA = {
    "lib_eid": pl.Utf8, "dataset": pl.Utf8, "type": pl.Utf8, "techniques": pl.List(pl.Utf8), "roles": pl.List(pl.Utf8),
    "pool": pl.Utf8, "actor": pl.Utf8, "victim": pl.Utf8, "n_rows": pl.Int64, "n_primary": pl.Int64,
    "n_primary_nondns": pl.Int64, "n_victim_orig": pl.Int64, "n_actor_orig": pl.Int64, "n_int_peers": pl.Int32,
    "n_ext_peers": pl.Int32, "int_peers": pl.List(pl.Utf8), "ext_peers": pl.List(pl.Utf8), "t0": pl.Float64,
    "t1": pl.Float64, "te_max": pl.Float64, "dur": pl.Float64, "elig_victim": pl.Boolean, "elig_actor": pl.Boolean,
    "elig_c2": pl.Boolean, "files": pl.List(pl.Utf8), "slots": pl.Utf8,
}


def _is_internal_unicast(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if a.version != 4:
        return False
    private = any(a in ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
    return private and not str(a).endswith(".255") and not str(a).endswith(".0") and not a.is_multicast


def library_dir(ds: str) -> Path:
    return interim(ds, "library")


# --------------------------------------------------------------------------- TON build
def ton_role_table(ep: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """Per-episode role statistics from P2-A's episodes + their records (rec_id joined)."""
    ips = pl.concat([recs["src"], recs["dst"]]).unique().drop_nulls()
    lut = pl.DataFrame({"peer": ips, "peer_int": [_is_internal_unicast(i) for i in ips.to_list()]},
                       schema={"peer": pl.Utf8, "peer_int": pl.Boolean})
    r = recs.join(ep.select(pl.col("episode_id").alias("lib_eid"), pl.col("attacker").alias("actor")),
                  on="lib_eid", how="left")
    r = r.with_columns((pl.col("src") == pl.col("actor")).alias("a_orig"),
                       pl.when(pl.col("src") == pl.col("actor")).then(pl.col("dst")).otherwise(pl.col("src"))
                       .alias("peer"))
    r = r.join(lut, on="peer", how="left").with_columns(pl.col("peer_int").fill_null(False))
    per_peer = (r.filter(pl.col("peer_int") & (pl.col("peer") != pl.col("actor")))
                .group_by("lib_eid", "peer")
                .agg(pl.len().alias("n"), (pl.col("dport") != 53).fill_null(True).sum().alias("n_nondns"),
                     (~pl.col("a_orig")).sum().alias("n_vorig"))
                .sort(["lib_eid", "n_nondns", "n", "peer"], descending=[False, True, True, False]))
    prim = per_peer.group_by("lib_eid", maintain_order=True).first().rename(
        {"peer": "victim", "n": "n_primary", "n_nondns": "n_primary_nondns", "n_vorig": "n_victim_orig"})
    tot = r.group_by("lib_eid").agg(
        pl.len().alias("n_rows"), pl.col("a_orig").sum().alias("n_actor_orig"),
        pl.col("peer").filter(pl.col("peer_int") & (pl.col("peer") != pl.col("actor"))).unique().sort().alias("int_peers"),
        pl.col("peer").filter(~pl.col("peer_int")).unique().sort().alias("ext_peers"),
        pl.col("ts").min().alias("t0"), pl.col("ts").max().alias("t1"), pl.col("te").max().alias("te_max"))
    out = (ep.select(pl.col("episode_id").alias("lib_eid"), "type", "pool", pl.col("attacker").alias("actor"), "files")
           .join(tot, on="lib_eid", how="left").join(prim, on="lib_eid", how="left"))
    out = out.join(type_meta_frame(), on="type", how="left")
    out = out.with_columns(
        pl.lit("ton").alias("dataset"),
        *[pl.col(c).fill_null(0).cast(pl.Int64) for c in ("n_primary", "n_primary_nondns", "n_victim_orig",
                                                           "n_actor_orig", "n_rows")],
        pl.col("int_peers").list.len().cast(pl.Int32).alias("n_int_peers"),
        pl.col("ext_peers").list.len().cast(pl.Int32).alias("n_ext_peers"),
        (pl.col("t1") - pl.col("t0")).alias("dur"),
    )
    out = out.with_columns(
        (pl.col("victim").is_not_null() & (pl.col("n_primary") >= MIN_PRIMARY)).alias("elig_victim"),
        (pl.col("n_actor_orig") >= MIN_ACTOR).alias("elig_actor"),
        (pl.col("n_victim_orig") >= 1).alias("elig_c2"),
        pl.struct(["actor", "victim"]).struct.json_encode().alias("slots"),
    )
    return conform(out, LIB_SCHEMA).sort(["type", "pool", "t0", "lib_eid"])


def type_meta_frame() -> pl.DataFrame:
    types = sorted(set(ATTACK_TECHNIQUES) | set(ROLE_SET))
    return pl.DataFrame({"type": types,
                         "techniques": [list(ATTACK_TECHNIQUES.get(t, ())) for t in types],
                         "roles": [list(ROLE_SET.get(t, ())) for t in types]},
                        schema={"type": pl.Utf8, "techniques": pl.List(pl.Utf8), "roles": pl.List(pl.Utf8)})


def build_ton_library(force: bool = False) -> dict:
    """Read P2-A's TON outputs, write results/interim/ton/library/* (see module docstring)."""
    t_start = time.time()
    src = interim("ton")
    out = library_dir("ton")
    lib_p = out / "episodes.parquet"
    if lib_p.exists() and (out / "library.json").exists() and not force:
        log(f"TON library exists ({lib_p}); --force to rebuild")
        return json.loads((out / "library.json").read_text())
    ep = pl.read_parquet(src / "episodes.parquet")
    m = ep.select(pl.col("episode_id").alias("lib_eid"), pl.col("rec_ids")).explode("rec_ids").rename(
        {"rec_ids": "rec_id"})
    if m["rec_id"].n_unique() != m.height:
        raise ValueError("TON episodes overlap (a rec_id in two episodes)")
    recs = (pl.scan_parquet(str(src / "records" / "*.parquet")).join(m.lazy(), on="rec_id", how="inner").collect())
    if recs.height != m.height:
        raise ValueError(f"TON library: {m.height - recs.height} episode rec_ids missing from records/")
    from certichain.io.labels import read_labels_QUARANTINED  # generator side may read labels
    labs = read_labels_QUARANTINED(sorted(str(p) for p in (src / "labels").glob("*.parquet")),
                                   rec_ids=m["rec_id"].to_list())
    labs = labs.join(m, on="rec_id", how="inner")
    chk = labs.join(ep.select(pl.col("episode_id").alias("lib_eid"), "type"), on="lib_eid")
    bad = chk.filter((pl.col("label") != "1") | (pl.col("detailed_label") != pl.col("type")))
    if bad.height:
        raise ValueError(f"TON library: {bad.height} episode rows whose label disagrees with the episode type")
    table = ton_role_table(ep, recs)
    out.mkdir(parents=True, exist_ok=True)
    for pool in ("train", "caltest"):
        ids = table.filter(pl.col("pool") == pool)["lib_eid"]
        rp = conform(recs.filter(pl.col("lib_eid").is_in(ids.implode())), {**RECORD_SCHEMA, "lib_eid": pl.Utf8})
        atomic_write_parquet(rp.sort(["lib_eid", "ts", "rec_id"]), out / f"records_{pool}.parquet")
        lp = labs.filter(pl.col("lib_eid").is_in(ids.implode())).select("rec_id", "label", "detailed_label", "lib_eid")
        atomic_write_parquet(lp.sort("rec_id"), out / f"labels_{pool}.parquet")
    atomic_write_parquet(table, lib_p)
    stats = library_stats(table)
    info = {"dataset": "ton", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": {"episodes": "P2_CertiChain/results/interim/ton/episodes.parquet",
                       "p2a_meta_config_hash": json.loads((src / "meta.json").read_text()).get("config_hash")},
            "config": {"MIN_PRIMARY": MIN_PRIMARY, "MIN_ACTOR": MIN_ACTOR, "types": list(TON_TYPES)},
            "n_episodes": table.height, "n_rows": int(recs.height), "stats": stats,
            "runtime_s": round(time.time() - t_start, 1)}
    atomic_write_json(out / "library.json", info)
    log(f"TON library: {table.height} episodes, {recs.height} rows -> {out}")
    return info


def library_stats(table: pl.DataFrame) -> list[dict]:
    """Table III: episode counts per type per pool (all, and eligible per binding)."""
    g = (table.group_by("type", "pool").agg(
        pl.len().alias("n_episodes"), pl.col("elig_victim").sum().alias("n_elig_victim"),
        pl.col("elig_actor").sum().alias("n_elig_actor"),
        (pl.col("elig_victim") & pl.col("elig_c2")).sum().alias("n_elig_c2_victim"),
        pl.col("n_rows").sum().alias("n_rows"), pl.col("dur").median().alias("dur_median_s"),
        pl.col("n_int_peers").median().alias("int_peers_median"),
        pl.col("actor").n_unique().alias("n_actors"), pl.col("victim").n_unique().alias("n_primary_victims"))
        .sort("type", "pool"))
    return g.to_dicts()


# --------------------------------------------------------------------------- Gotham build
# (session, Gotham detailed label) -> library type. Pools (DESIGN §5.1.3, no device in both):
GOTHAM_LABEL_TYPES = {
    ("mirai-infection", "Telnet Brute Force"): "mirai_bf", ("mirai-infection", "Ingress Tool Transfer"): "mirai_itt",
    ("mirai-infection", "File Download"): "mirai_dl", ("mirai-infection", "C&C Communication"): "mirai_cc",
    ("mirai-infection", "TCP Scan"): "mirai_scan", ("mirai-infection", "Reporting"): "mirai_report",
    ("mirai-dos", "Mirai C&C Communication"): "mirai_dos_cc", ("mirai-dos", "Mirai UDP Flooding"): "mirai_udpflood",
    ("mirai-dos", "Mirai TCP Flooding"): "mirai_tcpflood", ("mirai-dos", "Mirai GRE Flooding"): "mirai_greflood",
    ("merlin", "Merlin C&C Communication"): "merlin_c2", ("merlin", "Merlin ICMP Flooding"): "merlin_icmpflood",
    ("merlin", "Merlin UDP Flooding"): "merlin_udpflood", ("merlin", "Merlin TCP Flooding"): "merlin_tcpflood",
    ("coap-amplificator", "UDP Scan"): "udp_scan", ("coap-amplificator", "CoAP Amplification"): "amplification",
    ("network-scanning", "Unknown"): "masscan",
}
GOTHAM_FLOODS = ("mirai_udpflood", "mirai_tcpflood", "mirai_greflood", "merlin_icmpflood", "merlin_udpflood",
                 "merlin_tcpflood")
MI_POOLS = {"train": ("192.168.20.10", "192.168.18.15", "192.168.17.10"),
            "caltest": ("192.168.20.19", "192.168.18.10", "192.168.17.15", "192.168.18.17", "192.168.18.18")}
BOT_POOLS = {"train": ("192.168.17.10", "192.168.20.10"), "caltest": ("192.168.17.15",)}
COAP_POOLS = {"train": ("192.168.20.10",), "caltest": ()}
GOTHAM_BG_SPLIT = "2025-01-14 19:43:00"          # benign capture: before → train, after → caltest
MIN_SCAN_FLOWS = 50


def _utc(s: str) -> float:
    import datetime as _dt
    return _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc).timestamp()


def gotham_flows() -> tuple[pl.DataFrame, dict]:
    """All Gotham device captures (P2-A), flows seen by two captures de-duplicated exactly as P2-B's
    Gotham loader does (same 5-tuple, starts < 1 s apart → keep the copy of the capture file sorting first),
    joined with the quarantined labels and tagged with their session."""
    from certichain.io.labels import read_labels_QUARANTINED
    src = interim("gotham")
    files = sorted((src / "records").glob("*.parquet"))
    rec = pl.concat([pl.read_parquet(f).with_columns(pl.lit(f.stem).alias("_cap")) for f in files])
    labs = read_labels_QUARANTINED(sorted(str(p) for p in (src / "labels").glob("*.parquet")))
    k = ["src", "sport", "dst", "dport", "proto"]
    rec = rec.sort(k + ["ts", "_cap"], nulls_last=True)
    same = pl.all_horizontal([(pl.col(c) == pl.col(c).shift(1)).fill_null(pl.col(c).is_null() & pl.col(c).shift(1).is_null())
                              for c in k])
    dup = same & ((pl.col("ts") - pl.col("ts").shift(1)) < 1.0) & (pl.col("_cap") != pl.col("_cap").shift(1))
    n0 = rec.height
    rec = rec.filter(~dup.fill_null(False)).drop("_cap")
    sess = json.loads((src / "sessions.json").read_text())
    expr = pl.lit(None, dtype=pl.Utf8)
    for b in sess["bounds_ns"]:
        expr = (pl.when((pl.col("ts") >= b["start_ns"] / 1e9) & (pl.col("ts") < b["end_ns_excl"] / 1e9))
                .then(pl.lit(b["name"])).otherwise(expr))
    rec = rec.with_columns(expr.alias("session")).join(labs, on="rec_id", how="left")
    return rec, {"n_flows": n0, "n_duplicates_removed": n0 - rec.height, "sessions": sess}


def gotham_episode_rows(flows: pl.DataFrame, sess: dict, reports: pl.DataFrame | None,
                        gap_s: float = 300.0, max_dur_s: float = 600.0) -> pl.DataFrame:
    """Per-(type, device) activities cut into ≤ 10-min sub-windows (bursts at gaps > 300 s), with slots."""
    infra = set(sess["infrastructure"])
    bots = set(BOT_POOLS["train"]) | set(BOT_POOLS["caltest"])
    keys = pl.DataFrame({"session": [k[0] for k in GOTHAM_LABEL_TYPES], "detailed_label": [k[1] for k in GOTHAM_LABEL_TYPES],
                         "type": list(GOTHAM_LABEL_TYPES.values())})
    a = flows.join(keys, on=["session", "detailed_label"], how="inner")
    is_flood = pl.col("type").is_in(list(GOTHAM_FLOODS))
    a = a.with_columns(
        pl.when(is_flood).then(pl.when(pl.col("src").is_in(list(bots))).then(pl.col("src"))
                               .when(pl.col("dst").is_in(list(bots))).then(pl.col("dst")).otherwise(None))
        .when(pl.col("type") == "mirai_scan").then(pl.col("src"))
        .when(pl.col("src").is_in(list(infra))).then(pl.col("dst"))
        .otherwise(pl.col("src")).alias("device"),
        pl.when(pl.col("src").is_in(list(infra))).then(pl.col("src"))
        .when(pl.col("dst").is_in(list(infra))).then(pl.col("dst")).otherwise(None).alias("server"))
    fam_pool = []
    for pool in ("train", "caltest"):
        for t in ("mirai_bf", "mirai_itt", "mirai_dl", "mirai_cc", "mirai_scan", "mirai_report", "masscan"):
            fam_pool += [(t, d, pool) for d in MI_POOLS[pool]]
        for t in ("mirai_dos_cc", "merlin_c2", *GOTHAM_FLOODS):
            fam_pool += [(t, d, pool) for d in BOT_POOLS[pool]]
        for t in ("udp_scan", "amplification"):
            fam_pool += [(t, d, pool) for d in COAP_POOLS[pool]]
    fp = pl.DataFrame(fam_pool, schema={"type": pl.Utf8, "device": pl.Utf8, "pool": pl.Utf8}, orient="row")
    a = a.join(fp, on=["type", "device"], how="inner")
    g = ["type", "device"]
    a = a.sort(g + ["ts", "rec_id"])
    a = a.with_columns((pl.col("ts").diff().over(g).fill_null(float("inf")) > gap_s).cum_sum().over(g)
                       .cast(pl.Int32).alias("burst"))
    a = a.with_columns(pl.col("ts").min().over(g + ["burst"]).alias("_b0"))
    a = a.with_columns(((pl.col("ts") - pl.col("_b0")) // max_dur_s).cast(pl.Int32).alias("sub"))
    oct_ = pl.col("device").str.split(".").list.slice(2, 2).list.join("-")
    a = a.with_columns(pl.format("g-{}-{}-b{}-s{}", pl.col("type"), oct_, pl.col("burst").cast(pl.Utf8).str.zfill(2),
                                 pl.col("sub").cast(pl.Utf8).str.zfill(2)).alias("lib_eid"))
    if (a.filter(pl.col("type") == "mirai_scan").height):
        scans = a.filter(pl.col("type") == "mirai_scan").group_by("lib_eid").agg(pl.len().alias("n"))
        small = scans.filter(pl.col("n") < MIN_SCAN_FLOWS)["lib_eid"]
        a = a.filter(~pl.col("lib_eid").is_in(small.implode()))
    return a


def build_gotham_library(force: bool = False) -> dict:
    t_start = time.time()
    out = library_dir("gotham")
    lib_p = out / "episodes.parquet"
    if lib_p.exists() and (out / "library.json").exists() and not force:
        log(f"Gotham library exists ({lib_p}); --force to rebuild")
        return json.loads((out / "library.json").read_text())
    flows, info = gotham_flows()
    sess = info["sessions"]
    rp = interim("gotham", "reports.parquet")
    reports = pl.read_parquet(rp).select("t", "reporter", "target") if rp.exists() else None
    a = gotham_episode_rows(flows, sess, reports)
    # target slot of propagation scans: the host the bot reported (if any) else its most-contacted internal dst
    rep_t = {}
    if reports is not None and reports.height:
        for r, tgt in reports.sort("t").select("reporter", "target").iter_rows():
            rep_t.setdefault(r, []).append(tgt)
    def _target(eid_df: pl.DataFrame, dev: str) -> str | None:
        dsts = eid_df.filter(pl.col("dst") != dev)["dst"].value_counts(sort=True)
        cands = [d for d in dsts["dst"].to_list() if _is_internal_unicast(d)]
        for t in rep_t.get(dev, []):
            if t in cands:
                return t
        return cands[0] if cands else None
    rows = []
    for (eid,), grp in a.group_by(["lib_eid"], maintain_order=True):
        r0 = grp.row(0, named=True)
        server = grp["server"].drop_nulls().mode().sort().to_list()
        server = server[0] if server else None
        target = _target(grp, r0["device"]) if r0["type"] == "mirai_scan" else None
        slots = {"victim": r0["device"], "server": server, "target": target}
        ext = grp.select(pl.concat_list("src", "dst").explode().alias("ip"))["ip"].unique().to_list()
        rows.append({"lib_eid": eid, "dataset": "gotham", "type": r0["type"],
                     "techniques": list(ATTACK_TECHNIQUES.get(r0["type"], ())),
                     "roles": ["victim"] + (["server"] if server else []) + (["target"] if target else []),
                     "pool": r0["pool"], "actor": server, "victim": r0["device"], "n_rows": grp.height,
                     "n_primary": grp.height, "n_primary_nondns": grp.height, "n_victim_orig": int((grp["src"] == r0["device"]).sum()),
                     "n_actor_orig": int((grp["src"] == server).sum()) if server else 0,
                     "n_int_peers": len([i for i in ext if _is_internal_unicast(i)]) - 1,
                     "n_ext_peers": len([i for i in ext if not _is_internal_unicast(i)]),
                     "int_peers": sorted(i for i in ext if _is_internal_unicast(i) and i != r0["device"])[:256],
                     "ext_peers": sorted(i for i in ext if not _is_internal_unicast(i))[:256],
                     "t0": float(grp["ts"].min()), "t1": float(grp["ts"].max()), "te_max": float(grp["te"].max()),
                     "dur": float(grp["ts"].max() - grp["ts"].min()), "elig_victim": True, "elig_actor": True,
                     "elig_c2": True, "files": sorted(grp["src_file"].unique().to_list()),
                     "slots": json.dumps(slots)})
    table = conform(pl.DataFrame(rows, infer_schema_length=None), LIB_SCHEMA).sort(["type", "pool", "t0", "lib_eid"])
    out.mkdir(parents=True, exist_ok=True)
    recs = conform(a, {**RECORD_SCHEMA, "lib_eid": pl.Utf8})
    labs = a.select("rec_id", "label", "detailed_label", "lib_eid")
    for pool in ("train", "caltest"):
        ids = table.filter(pl.col("pool") == pool)["lib_eid"].implode()
        atomic_write_parquet(recs.filter(pl.col("lib_eid").is_in(ids)).sort(["lib_eid", "ts", "rec_id"]),
                             out / f"records_{pool}.parquet")
        atomic_write_parquet(labs.filter(pl.col("lib_eid").is_in(ids)).sort("rec_id"), out / f"labels_{pool}.parquet")
    atomic_write_parquet(table, lib_p)
    # benign backgrounds (de-duplicated flows of the 01-14 benign capture, split in time)
    split = _utc(GOTHAM_BG_SPLIT)
    ben = flows.filter(pl.col("session") == "benign")
    if ben.filter(pl.col("label") != "Benign").height:
        raise ValueError("Gotham benign capture contains non-benign flows")
    bg_info = {}
    for pool, f in (("train", pl.col("ts") < split), ("caltest", pl.col("ts") >= split)):
        b = conform(ben.filter(f), RECORD_SCHEMA).sort(["ts", "rec_id"])
        atomic_write_parquet(b, out / f"background_{pool}.parquet")
        bg_info[pool] = {"n_records": b.height, "t0": float(b["ts"].min()), "t1": float(b["ts"].max())}
    stats = library_stats(table)
    inf = {"dataset": "gotham", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "flows": {k: v for k, v in info.items()
                                                                                         if k != "sessions"},
           "config": {"label_types": {f"{k[0]}|{k[1]}": v for k, v in GOTHAM_LABEL_TYPES.items()}, "mi_pools": MI_POOLS,
                      "bot_pools": BOT_POOLS, "coap_pools": COAP_POOLS, "bg_split": GOTHAM_BG_SPLIT,
                      "sub_window_s": 600, "burst_gap_s": 300, "min_scan_flows": MIN_SCAN_FLOWS, "row_sampling": None},
           "n_episodes": table.height, "n_rows": int(recs.filter(pl.col("lib_eid").is_in(table["lib_eid"].implode())).height),
           "backgrounds": bg_info, "stats": stats, "runtime_s": round(time.time() - t_start, 1)}
    atomic_write_json(out / "library.json", inf)
    log(f"Gotham library: {table.height} episodes -> {out}")
    return inf


def gotham_background_pools() -> dict:
    d = library_dir("gotham")
    return {"train": {"source": "gotham benign 2025-01-14 18:39:53-19:43:00 (dedup)", "path": d / "background_train.parquet"},
            "caltest": {"source": "gotham benign 2025-01-14 19:43:00-20:47:32 (dedup)", "path": d / "background_caltest.parquet"}}


# --------------------------------------------------------------------------- runtime access
def requirement(etype: str, binds: dict[str, str]) -> tuple[str, ...]:
    """Eligibility flags a library episode needs for a node binding (see module doc)."""
    req = ["elig_victim"] if "victim" in binds else ["elig_actor"]
    if etype == "backdoor":
        req.append("elig_c2")
    return tuple(req)


@dataclass
class EpisodeLibrary:
    ds: str
    table: pl.DataFrame
    _rec: dict = field(default_factory=dict)
    _lab: dict = field(default_factory=dict)
    _cand: dict = field(default_factory=dict)
    _rows: dict = field(default_factory=dict)

    @classmethod
    def load(cls, ds: str) -> "EpisodeLibrary":
        d = library_dir(ds)
        return cls(ds=ds, table=pl.read_parquet(d / "episodes.parquet"))

    def row(self, lib_eid: str) -> dict:
        if lib_eid not in self._rows:
            r = self.table.filter(pl.col("lib_eid") == lib_eid)
            if r.height != 1:
                raise KeyError(lib_eid)
            d = r.row(0, named=True)
            d["slots"] = json.loads(d["slots"])
            self._rows[lib_eid] = d
        return self._rows[lib_eid]

    def candidates(self, pool: str, etype: str, binds: dict[str, str]) -> list[str]:
        key = (pool, etype, requirement(etype, binds))
        if key not in self._cand:
            f = (pl.col("pool") == pool) & (pl.col("type") == etype)
            for flag in key[2]:
                f = f & pl.col(flag)
            self._cand[key] = self.table.filter(f).sort("lib_eid")["lib_eid"].to_list()
        return self._cand[key]

    def types(self, pool: str) -> set[str]:
        return set(self.table.filter(pl.col("pool") == pool)["type"].unique().to_list())

    def _store(self, pool: str):
        if pool not in self._rec:
            d = library_dir(self.ds)
            rec = pl.read_parquet(d / f"records_{pool}.parquet")
            lab = pl.read_parquet(d / f"labels_{pool}.parquet")
            self._rec[pool] = {k[0]: g for k, g in rec.partition_by("lib_eid", as_dict=True).items()}
            self._lab[pool] = {k[0]: g for k, g in lab.partition_by("lib_eid", as_dict=True).items()}
        return self._rec[pool], self._lab[pool]

    def records(self, pool: str, lib_eid: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        rec, lab = self._store(pool)
        return rec[lib_eid].drop("lib_eid"), lab[lib_eid].drop("lib_eid")

    def sample(self, rng: np.random.Generator, pool: str, etype: str, binds: dict[str, str],
               used: set[str] | None = None) -> str:
        """Unif(library[pool][type]) among eligible episodes, avoiding ``used`` when possible."""
        c = self.candidates(pool, etype, binds)
        if not c:
            raise LookupError(f"no eligible {etype} episode in pool {pool} for binding {binds}")
        fresh = [e for e in c if not used or e not in used]
        pick = fresh if fresh else c
        return pick[int(rng.integers(len(pick)))]


__all__ = ["build_gotham_library", "gotham_background_pools", "gotham_flows", "GOTHAM_LABEL_TYPES", "TON_TYPES", "ATTACK_TECHNIQUES", "MIN_PRIMARY", "MIN_ACTOR", "LIB_SCHEMA", "library_dir",
           "ton_role_table", "build_ton_library", "library_stats", "requirement", "EpisodeLibrary"]
