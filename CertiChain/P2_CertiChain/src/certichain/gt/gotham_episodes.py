"""Gotham 2025 GT extractor (DESIGN §5.2.3) — schedule- and payload-derived episodes.

Record-level episodes (flow label = majority packet label, P2-A), per session:

mirai-infection, per victim x (roles: scanner .0.100, C&C .33.10, report .33.11, loader .33.12,
download .33.13):
  BF0(x)  scanner → x on telnet ports (23/2323)
  ITT(x)  loader ↔ x:23 (either direction)
  DL(x)   x → download:80
  CC(x)   x ↔ C&C:23
  SCAN(x) rows labelled 'TCP Scan' or 'Telnet Brute Force' with src = x (a bot, ≠ scanner),
          starting at/after DL(x) starts (bot→bot brute force included, caveat 4)
  REP(x)  x → report:48101
  T*: ITT(x)→DL(x), DL(x)→CC(x), DL(x)→SCAN(x), SCAN(x)→REP(x) (when both exist), and the
  cracker resolution of every ITT(y) (t_y = first ITT(y) record):
    (a) a decoded report of bot b names y and precedes t_y  → REP(b)→ITT(y) ∈ T*; BF0(y)⇝ITT(y) AMB
    (b) else, if every REP record before t_y was decoded    → BF0(y)→REP0(y)→ITT(y), REP0(y) hidden
                                                             (bridge BF0(y)⇝ITT(y) = 1)
    (c) otherwise (a needed payload is missing)             → BF0(y)⇝ITT(y) and every REP(b)→ITT(y)
                                                             with b reporting before t_y are AMB
  Without results/interim/gotham/reports.parquet nothing is decoded, so every ITT(y) is case (c).
mirai-dos:  CC(bot) → FLOOD(bot→victim, kind) for every (bot, victim, kind) ('Mirai * Flooding').
merlin:     C2(agent) ('Merlin C&C Communication' or agent ↔ .34.10:443) → FLOOD(agent→victim, kind).
coap:       UDPSCAN(.35.10 → .20.10, 'UDP Scan') → AMP(.0.200 ↔ .20.10 or 'CoAP Amplification').
masscan:    one episode ('Unknown' rows of .35.10, caveat 2), no transitions.
Everything else is ⊥. Campaign = "gotham:<session>".
Unit episode = the episode held by ≥ 50 % of the unit's records (unique max), else MIXED (AMB).
SEED-PRIORITY rule (coordinator decision, 18 Oct 19:52; Gotham-real only, ``seed_priority=True``): a unit
whose episode records span ≥ 2 victims' episodes of the SAME stage (e.g. the scanner's BF0 sweep or the
loader's ITT logins as one fan-out unit) takes the seed victim's episode of that stage when the seed is
among them (i.e. among its destinations), else it is MIXED (AMB). Other units use the default rule.

Report table expected from P2-A (one row per report flow; extra columns ignored):
  ts (float; alias t), reporter (str; alias src), victim (str|null; alias target/victim_ip/target_ip),
  decoded (bool; default = victim is not null). A REP record matches a report row with the same
  reporter and |Δts| ≤ 1 s.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from certichain.gt.semantics import EpisodeGT, read_quarantined, unit_majority


@dataclass(frozen=True)
class GothamRoles:
    scanner: str = "192.168.0.100"
    cnc: str = "192.168.33.10"
    report: str = "192.168.33.11"
    loader: str = "192.168.33.12"
    download: str = "192.168.33.13"
    merlin: str = "192.168.34.10"
    udp_scanner: str = "192.168.35.10"
    coap_device: str = "192.168.20.10"
    amp_peer: str = "192.168.0.200"
    telnet_ports: tuple[int, ...] = (23, 2323)
    loader_ports: tuple[int, ...] = (23,)
    download_ports: tuple[int, ...] = (80,)
    cnc_ports: tuple[int, ...] = (23,)
    report_ports: tuple[int, ...] = (48101,)
    merlin_ports: tuple[int, ...] = (443,)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "GothamRoles":
        cfg = dict(cfg or {})
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in cls.__dataclass_fields__}
        return cls(**kw)


SESSION_ALIASES = {"mirai-infection": "mirai-infection", "mirai-dos": "mirai-dos", "merlin": "merlin",
                   "coap-amplificator": "coap", "coap": "coap", "network-scanning": "masscan",
                   "masscan": "masscan", "benign": "benign"}
REPORT_MATCH_S = 1.0


def _flood_kind(label: str) -> str:
    lab = (label or "").upper()
    for k in ("UDP", "TCP", "GRE", "ICMP", "SYN", "ACK", "HTTP"):
        if k in lab.split():
            return k
    return "OTHER"


def record_episodes(rec: pl.DataFrame, labels: pl.DataFrame, session: str,
                    roles: GothamRoles = GothamRoles()) -> pl.DataFrame:
    """rec_id → (episode, kind, x) for the session's episode table (⊥ rows omitted)."""
    s = SESSION_ALIASES.get(session, session)
    r = rec.select("rec_id", "ts", "src", "dst", "dport").join(
        labels.select("rec_id", pl.col("detailed_label").fill_null("").alias("dl")), on="rec_id", how="left")
    r = r.with_columns(pl.col("dl").fill_null(""))
    src, dst, dport, dl = pl.col("src"), pl.col("dst"), pl.col("dport"), pl.col("dl")
    parts: list[pl.DataFrame] = []

    def add(df: pl.DataFrame, kind: str, x: pl.Expr, extra: pl.Expr | None = None) -> None:
        if df.height == 0:
            return
        eid = pl.concat_str([pl.lit(kind + "("), x, pl.lit(")")]) if extra is None else \
            pl.concat_str([pl.lit(kind + "("), x, pl.lit("->"), extra, pl.lit(")")])
        parts.append(df.select("rec_id", "ts", eid.alias("episode"), pl.lit(kind).alias("kind"), x.alias("x")))

    if s in ("mirai-infection", "mirai-dos"):
        cc = r.filter(((dst == roles.cnc) & dport.is_in(list(roles.cnc_ports))) | (src == roles.cnc))
        add(cc, "CC", pl.when(pl.col("src") == roles.cnc).then(pl.col("dst")).otherwise(pl.col("src")))
    if s == "mirai-infection":
        add(r.filter((src == roles.scanner) & dport.is_in(list(roles.telnet_ports))), "BF0", pl.col("dst"))
        itt = r.filter(((src == roles.loader) & dport.is_in(list(roles.loader_ports))) | (dst == roles.loader))
        add(itt, "ITT", pl.when(pl.col("src") == roles.loader).then(pl.col("dst")).otherwise(pl.col("src")))
        dl_ = r.filter((dst == roles.download) & dport.is_in(list(roles.download_ports)))
        add(dl_, "DL", pl.col("src"))
        add(r.filter((dst == roles.report) & dport.is_in(list(roles.report_ports))), "REP", pl.col("src"))
        dl_start = dl_.group_by("src").agg(pl.col("ts").min().alias("t_dl"))
        sc = (r.filter(dl.is_in(["TCP Scan", "Telnet Brute Force"]) & (src != roles.scanner))
              .join(dl_start, on="src", how="inner").filter(pl.col("ts") >= pl.col("t_dl")))
        add(sc, "SCAN", pl.col("src"))
    if s == "mirai-dos":
        fl = r.filter(dl.str.starts_with("Mirai ") & dl.str.ends_with("Flooding"))
        fl = fl.with_columns(pl.col("dl").map_elements(_flood_kind, return_dtype=pl.Utf8).alias("fk"))
        add(fl, "FLOOD", pl.col("src"), pl.concat_str([pl.col("dst"), pl.lit(":"), pl.col("fk")]))
    if s == "merlin":
        c2 = r.filter((dl == "Merlin C&C Communication")
                      | (((dst == roles.merlin) & dport.is_in(list(roles.merlin_ports))) | (src == roles.merlin)))
        add(c2, "C2", pl.when(pl.col("src") == roles.merlin).then(pl.col("dst")).otherwise(pl.col("src")))
        fl = r.filter(dl.str.starts_with("Merlin ") & dl.str.ends_with("Flooding"))
        fl = fl.with_columns(pl.col("dl").map_elements(_flood_kind, return_dtype=pl.Utf8).alias("fk"))
        add(fl, "FLOOD", pl.col("src"), pl.concat_str([pl.col("dst"), pl.lit(":"), pl.col("fk")]))
    if s == "coap":
        add(r.filter((dl == "UDP Scan") & (src == roles.udp_scanner) & (dst == roles.coap_device)),
            "UDPSCAN", pl.col("dst"))
        amp = r.filter((dl == "CoAP Amplification")
                       | ((src == roles.amp_peer) & (dst == roles.coap_device))
                       | ((src == roles.coap_device) & (dst == roles.amp_peer)))
        add(amp, "AMP", pl.lit(roles.coap_device))
    if s == "masscan":
        add(r.filter((dl == "Unknown") & (src == roles.udp_scanner)), "MASSCAN", pl.lit(roles.udp_scanner))
    if not parts:
        return pl.DataFrame(schema={"rec_id": pl.Int64, "ts": pl.Float64, "episode": pl.Utf8, "kind": pl.Utf8,
                                    "x": pl.Utf8})
    # a record matching several patterns keeps the first in this priority order
    prio = {"ITT": 0, "DL": 1, "REP": 2, "CC": 3, "C2": 3, "BF0": 4, "SCAN": 5, "FLOOD": 6, "UDPSCAN": 7,
            "AMP": 8, "MASSCAN": 9}
    out = pl.concat(parts).with_columns(pl.col("kind").replace_strict(prio, default=99).alias("_p"))
    return out.sort(["rec_id", "_p"]).unique("rec_id", keep="first").drop("_p").sort("rec_id")


def _reports(reports: pl.DataFrame | None) -> pl.DataFrame | None:
    if reports is None or reports.height == 0:
        return None
    cols = reports.columns
    rep = "reporter" if "reporter" in cols else ("src" if "src" in cols else None)
    vic = next((c for c in ("victim", "victim_ip", "target_ip", "target") if c in cols), None)
    tcol = "ts" if "ts" in cols else ("t" if "t" in cols else None)
    if rep is None or tcol is None:
        return None
    df = reports.select(pl.col(tcol).cast(pl.Float64).alias("ts"), pl.col(rep).cast(pl.Utf8).alias("reporter"),
                        (pl.col(vic).cast(pl.Utf8) if vic else pl.lit(None, dtype=pl.Utf8)).alias("victim"),
                        (pl.col("decoded").cast(pl.Boolean) if "decoded" in cols else
                         (pl.col(vic).is_not_null() if vic else pl.lit(False))).alias("decoded"))
    return df


def _seed_priority(units: pl.DataFrame, re_: pl.DataFrame, seeds) -> dict[int, str | None]:
    """unit_id → episode (seed's same-stage episode) or None (MIXED) for multi-victim same-stage units."""
    from certichain.units.sessionize import unit_record_map

    m = unit_record_map(units).join(re_.select("rec_id", "episode", "kind", "x"), on="rec_id", how="inner")
    g = m.group_by("unit_id").agg(pl.col("episode").n_unique().alias("n_ep"), pl.col("kind").n_unique().alias("n_kind"),
                                  pl.col("kind").first().alias("kind"), pl.col("x").unique().alias("xs"))
    out: dict[int, str | None] = {}
    seeds = set(seeds)
    for uid, n_ep, n_kind, kind, xs in g.iter_rows():
        if n_ep >= 2 and n_kind == 1:
            hit = sorted(seeds & set(xs))
            out[int(uid)] = f"{kind}({hit[0]})" if hit else None
    return out


def build_gotham_gt(records: pl.DataFrame, units: pl.DataFrame, labels: pl.DataFrame, session: str,
                    reports: pl.DataFrame | None = None, roles: GothamRoles = GothamRoles(),
                    seeds=(), seed_priority: bool = False) -> EpisodeGT:
    s = SESSION_ALIASES.get(session, session)
    camp = f"gotham:{s}"
    re_ = record_episodes(records, labels, s, roles)
    gt = EpisodeGT(unit_episode={}, unit_campaign={}, transitions=set(), hidden=set(), episode_campaign={},
                   meta={"source": "gotham", "session": s})
    eps = re_.group_by(["episode", "kind", "x"]).agg(pl.col("ts").min().alias("t0"), pl.len().alias("n"))
    floods = sorted((e, x) for e, k, x in eps.select("episode", "kind", "x").iter_rows() if k == "FLOOD")
    ep_of = {(k, x): e for e, k, x in eps.select("episode", "kind", "x").iter_rows() if k != "FLOOD"}
    t0 = dict(zip(eps["episode"].to_list(), eps["t0"].to_list()))
    for e in eps["episode"].to_list():
        gt.episode_campaign[e] = camp

    def E(kind: str, x: str) -> str | None:
        return ep_of.get((kind, x))

    if s == "mirai-infection":
        hosts = {x for (_, x) in ep_of}
        for x in sorted(hosts):
            itt, dl_, cc, sc, rep = E("ITT", x), E("DL", x), E("CC", x), E("SCAN", x), E("REP", x)
            if itt and dl_:
                gt.transitions.add((itt, dl_))
            if dl_ and cc:
                gt.transitions.add((dl_, cc))
            if dl_ and sc:
                gt.transitions.add((dl_, sc))
            if sc and rep:
                gt.transitions.add((sc, rep))
        # cracker resolution
        rep_recs = re_.filter(pl.col("kind") == "REP").select("rec_id", "ts", pl.col("x").alias("reporter"))
        rtab = _reports(reports)
        if rtab is not None and rep_recs.height:
            m = rep_recs.join(rtab, on="reporter", how="left", suffix="_r")
            m = m.with_columns(((pl.col("ts_r") - pl.col("ts")).abs() <= REPORT_MATCH_S).fill_null(False).alias("hit"))
            hits = m.filter(pl.col("hit")).group_by("rec_id").agg(pl.col("decoded").any().alias("decoded"),
                                                                 pl.col("victim").drop_nulls().alias("victims"))
            rep_recs = rep_recs.join(hits, on="rec_id", how="left").with_columns(pl.col("decoded").fill_null(False))
        else:
            rep_recs = rep_recs.with_columns(pl.lit(False).alias("decoded"),
                                             pl.lit(None, dtype=pl.List(pl.Utf8)).alias("victims"))
        cases = {}
        for (kind, y), itt in sorted(ep_of.items()):
            if kind != "ITT":
                continue
            ty = t0[itt]
            before = rep_recs.filter(pl.col("ts") < ty)
            naming = sorted({b for b, vs, dec in before.select("reporter", "victims", "decoded").iter_rows()
                             if dec and vs and y in vs})
            bf0 = E("BF0", y) or f"BF0({y})"
            gt.episode_campaign.setdefault(bf0, camp)
            if naming:
                for b in naming:
                    if E("REP", b):
                        gt.transitions.add((E("REP", b), itt))
                gt.amb_pairs.add((bf0, itt))
                cases[y] = "a"
            elif before.height == 0 or bool(before["decoded"].all()):
                rep0 = f"REP0({y})"
                gt.hidden.add(rep0)
                gt.episode_campaign[rep0] = camp
                gt.transitions.update({(bf0, rep0), (rep0, itt)})
                cases[y] = "b"
            else:
                gt.amb_pairs.add((bf0, itt))
                for b in sorted(set(before["reporter"].to_list())):
                    if E("REP", b):
                        gt.amb_pairs.add((E("REP", b), itt))
                cases[y] = "c"
        gt.meta["cracker_cases"] = cases
        gt.meta["reports_available"] = rtab is not None
    elif s == "mirai-dos":
        for e, bot in floods:                       # CC(bot) → FLOOD(bot → victim, kind), every pair
            if E("CC", bot):
                gt.transitions.add((E("CC", bot), e))
    elif s == "merlin":
        for e, agent in floods:                     # C2(agent) → FLOOD(agent → victim, kind)
            if E("C2", agent):
                gt.transitions.add((E("C2", agent), e))
    elif s == "coap":
        a, b = E("UDPSCAN", roles.coap_device), E("AMP", roles.coap_device)
        if a and b:
            gt.transitions.add((a, b))

    if units.height:
        maj = unit_majority(units, re_.select("rec_id", "episode"), "episode", min_share=0.5, denominator="all")
        prio = _seed_priority(units, re_, seeds) if seed_priority else {}
        n_prio = n_prio_mixed = 0
        for uid, val, mixed, _, _ in maj.iter_rows():
            uid = int(uid)
            if uid in prio:                       # multi-victim same-stage unit: seed priority
                if prio[uid] is None:
                    mixed, val = True, None
                    n_prio_mixed += 1
                else:
                    mixed, val = False, prio[uid]
                    n_prio += 1
            if mixed:
                gt.mixed_units.add(uid)
                gt.unit_campaign[uid] = camp
            elif val is not None:
                gt.unit_episode[uid] = val
                gt.unit_campaign[uid] = camp
        gt.meta.update({"seed_priority": bool(seed_priority), "n_units_seed_priority": n_prio,
                        "n_units_seed_priority_mixed": n_prio_mixed})
    gt.meta.update({"n_episodes": len(ep_of) + len(floods), "n_transitions": len(gt.transitions), "n_hidden": len(gt.hidden),
                    "n_amb_pairs": len(gt.amb_pairs)})
    return gt


def gotham_gt(records: pl.DataFrame, units: pl.DataFrame, session: str, label_paths,
              roles: GothamRoles = GothamRoles(), reports_path: str | None = None, seeds=(),
              seed_priority: bool = False) -> EpisodeGT:
    """Read quarantined labels (+ decoded reports if present) and build the EpisodeGT.

    ``records`` = ALL records of the incident window (not only the in-scope ones): episode extents
    and the cracker resolution (which bots reported before ITT(y), were their payloads decoded) are
    properties of the window's evidence, not of the seed's scope."""
    paths = label_paths if isinstance(label_paths, (list, tuple)) else [label_paths]
    ids = records["rec_id"].to_list()
    labels = pl.concat([read_quarantined("gotham", path=p, rec_ids=ids) for p in paths]) if paths else \
        pl.DataFrame(schema={"rec_id": pl.Int64, "label": pl.Utf8, "detailed_label": pl.Utf8})
    reports = read_quarantined("gotham", what="reports", path=reports_path)
    return build_gotham_gt(records, units, labels, session, reports, roles, seeds=seeds, seed_priority=seed_priority)
