"""Gotham 2025 unit labels from the orchestration rules (DESIGN §5.5; data/gotham2025/MANIFEST.md).

Re-implements (never executes) github_labeller/src/helpers/labeller.py on units:
1. the capture device is the n-th ``device_ip`` of its family (file name ``iotsim-<family>-<n>_...``);
   packets not involving the device IP, non-IP and IPv6 packets are dropped by the labeller
   (our label: ``unknown``);
2. device ↔ any of its family's ``server_ip`` → Benign (Benign takes precedence);
3. the scenario's rules in file order, first match wins: ``re.match`` on the IP text with
   ``x.x`` → ``.*`` (unescaped dots, as in the labeller), optional source/destination port pair
   (TCP only, as the labeller uses tcp.srcport/tcp.dstport) and ``protocol``;
4. anything left is ``Unknown`` (our ``unknown``).
Rules are evaluated per unit direction (they depend only on src, dst, ports and proto, so all
packets of one direction share a label). A unit is malicious iff any packet is; its family is the
malicious label with the most packets. Scenario network-scanning uses metadata-masscan.json
(MANIFEST caveat 2). ``class`` = grouped label (Benign, Network Scanning, Brute Force, Infection,
C&C Communication, DoS; Reporting is grouped under C&C Communication).
Incident = (scenario, victim device, stage) run of content-bearing units, split at gaps > 600 s.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import polars as pl

from ctsample.utils.io import DATA

META = DATA / "gotham2025" / "github_labeller" / "metadata"
SCEN_FILE = {"mirai-infection": "metadata-mirai-infection.json", "mirai-dos": "metadata-mirai-dos.json",
             "merlin": "metadata-merlin.json", "coap-amplificator": "metadata-coap-amplificator.json",
             "network-scanning": "metadata-masscan.json", "benign": None}
GROUP = {"TCP Scan": "Network Scanning", "UDP Scan": "Network Scanning", "Telnet Brute Force": "Brute Force",
         "Ingress Tool Transfer": "Infection", "File Download": "Infection", "Reporting": "C&C Communication",
         "Mirai C&C Communication": "C&C Communication", "Merlin C&C Communication": "C&C Communication",
         "C&C Communication": "C&C Communication", "CoAP Amplification": "DoS", "Merlin TCP Flooding": "DoS",
         "Merlin UDP Flooding": "DoS", "Merlin ICMP Flooding": "DoS", "Mirai TCP Flooding": "DoS",
         "Mirai UDP Flooding": "DoS", "Mirai GRE Flooding": "DoS", "Benign": "Benign"}
GAP_S = 600.0
FILE_RE = re.compile(r"^(?:[a-z-]+__)?(iotsim-[a-zA-Z-]+)-(\d+)_")


@lru_cache(maxsize=None)
def benign_meta() -> dict:
    return json.loads((META / "metadata-benign.json").read_text())


@lru_cache(maxsize=None)
def scenario_rules(scenario: str) -> tuple:
    fn = SCEN_FILE.get(scenario)
    if not fn:
        return ()
    out = []
    for r in json.loads((META / fn).read_text()):
        out.append((re.compile(r.get("source_ip", "").replace("x.x", ".*")),
                    re.compile(r.get("destination_ip", "").replace("x.x", ".*")),
                    (int(r["source_port"]), int(r["destination_port"]))
                    if "source_port" in r and "destination_port" in r else None,
                    int(r["protocol"]) if "protocol" in r else None, r.get("label", "Malicious")))
    return tuple(out)


def device_of(source: str) -> tuple[str, str, list[str]]:
    m = FILE_RE.match(source)
    if not m:
        raise ValueError(f"cannot parse Gotham device from {source!r}")
    fam, n = m.group(1), int(m.group(2))
    info = benign_meta()[fam]
    return fam, info["device_ip"][n - 1], list(info.get("server_ip", []))


def packet_label(src: str, dst: str, sport: int, dport: int, proto: int, dev: str, servers: list[str],
                 rules: tuple) -> str | None:
    """Label of the packets of one direction; None = dropped by the labeller (not the device's)."""
    if src != dev and dst != dev:
        return None
    if (src == dev and dst in servers) or (dst == dev and src in servers):
        return "Benign"
    for rs, rd, ports, pr, lab in rules:
        if not (rs.match(src) and rd.match(dst)):
            continue
        if ports is not None and not (proto == 6 and (sport, dport) == ports):
            continue
        if pr is not None and proto != pr:
            continue
        return lab
    return "Unknown"


def label_source(units: pl.DataFrame, src: dict) -> tuple[pl.DataFrame, dict]:
    scen = src.get("scenario") or "benign"
    fam, dev, servers = device_of(src["source"])
    rules = scenario_rules(scen)
    u = units.select("unit_gid", "pay", "proto", "ip_a", "port_a", "ip_b", "port_b", "first_ts_us", "device_id")
    if "pkts0" in units.columns:
        u = u.with_columns(units["pkts0"], units["pkts1"])
    rows = []
    cache: dict = {}
    for r in u.iter_rows(named=True):
        p = r["proto"]
        if p == 0 or ":" in r["ip_a"] or ":" in r["ip_b"]:            # non-IP or IPv6: dropped by the labeller
            rows.append((None, None))
            continue
        k = (p, r["ip_a"], r["port_a"], r["ip_b"], r["port_b"])
        lab = cache.get(k)
        if lab is None:
            sp, dp = (r["port_a"], r["port_b"]) if p in (6, 17) else (0, 0)
            l0 = packet_label(r["ip_a"], r["ip_b"], sp, dp, p, dev, servers, rules)
            l1 = packet_label(r["ip_b"], r["ip_a"], dp, sp, p, dev, servers, rules)
            lab = cache[k] = (l0, l1)
        rows.append(lab)
    l0 = [x[0] for x in rows]
    l1 = [x[1] for x in rows]
    lab = u.with_columns(pl.Series("l0", l0, dtype=pl.String), pl.Series("l1", l1, dtype=pl.String))
    if "pkts0" not in lab.columns:
        lab = lab.with_columns(pl.lit(1).alias("pkts0"), pl.lit(1).alias("pkts1"))
    mal0 = pl.col("l0").is_not_null() & ~pl.col("l0").is_in(["Benign", "Unknown"]) & (pl.col("pkts0") > 0)
    mal1 = pl.col("l1").is_not_null() & ~pl.col("l1").is_in(["Benign", "Unknown"]) & (pl.col("pkts1") > 0)
    fam_expr = (pl.when(mal0 & mal1).then(pl.when(pl.col("pkts0") >= pl.col("pkts1")).then(pl.col("l0"))
                                          .otherwise(pl.col("l1")))
                .when(mal0).then(pl.col("l0")).when(mal1).then(pl.col("l1")).otherwise(None))
    ben = ((pl.col("l0") == "Benign") & (pl.col("pkts0") > 0)) | ((pl.col("l1") == "Benign") & (pl.col("pkts1") > 0))
    lab = lab.with_columns(fam_expr.alias("detailed_label")).with_columns(
        pl.col("detailed_label").is_not_null().alias("is_malicious"))
    lab = lab.with_columns(
        pl.when(pl.col("is_malicious")).then(pl.lit("Malicious")).when(ben).then(pl.lit("Benign"))
        .otherwise(pl.lit("unknown")).alias("label"),
        pl.lit("orchestration_rules").alias("label_source"), pl.lit(None, dtype=pl.String).alias("conn_uid"),
        pl.lit(None, dtype=pl.Float64).alias("match_dstart_s"))
    lab = lab.with_columns(
        pl.when(pl.col("label") == "Benign").then(pl.lit("-")).otherwise(pl.col("detailed_label"))
        .alias("detailed_label"))
    lab = lab.with_columns(
        pl.col("detailed_label").replace_strict(GROUP, default=None, return_dtype=pl.String).alias("class"))
    lab = lab.with_columns(pl.when(pl.col("label") == "Benign").then(pl.lit("Benign"))
                           .otherwise(pl.col("class")).alias("class"),
                           pl.col("detailed_label").alias("family"))
    cb = (lab.filter(pl.col("is_malicious") & pl.col("pay")).select("unit_gid", "class", "first_ts_us")
          .sort(["class", "first_ts_us"]))
    if cb.height:
        cb = cb.with_columns(((pl.col("first_ts_us").diff().over("class").fill_null(0) > GAP_S * 1e6)
                              .cast(pl.Int32).cum_sum().over("class")).alias("k"))
        cb = cb.with_columns((pl.lit(f"{scen}|{dev}|") + pl.col("class") + "|" + pl.col("k").cast(pl.String))
                             .alias("incident_id"))
        lab = lab.join(cb.select("unit_gid", "incident_id"), on="unit_gid", how="left")
    else:
        lab = lab.with_columns(pl.lit(None, dtype=pl.String).alias("incident_id"))
    pay = lab.filter(pl.col("pay"))
    qa = {"units": lab.height, "payload_units": pay.height, "scenario": scen, "device_ip": dev, "family": fam,
          "label_counts": dict(lab.group_by("label").len().iter_rows()),
          "label_counts_payload": dict(pay.group_by("label").len().iter_rows()),
          "family_counts_mal_payload": dict(pay.filter(pl.col("is_malicious")).group_by("family").len().iter_rows()),
          "match_rate_payload": float((pay["label"] != "unknown").mean()) if pay.height else None,
          "match_rate_all": float((lab["label"] != "unknown").mean()) if lab.height else None,
          "malicious_payload_units": int(pay["is_malicious"].sum())}
    keep = ["unit_gid", "pay", "device_id", "label", "detailed_label", "label_source", "conn_uid", "match_dstart_s",
            "family", "class", "is_malicious", "incident_id"]
    return lab.select(keep), qa
