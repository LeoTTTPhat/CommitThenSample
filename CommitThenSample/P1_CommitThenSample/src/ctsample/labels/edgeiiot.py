"""Edge-IIoTset unit labels: the attacker rule (DESIGN §5.5; CSV join dropped).

* In an attack capture a unit is malicious iff 192.168.0.170 (the attacker) is an endpoint.
* Spoofed DDoS ICMP / TCP-SYN / UDP floods: also malicious iff the unit's initiator frame
  source MAC is the attacker NIC d8:f2:ca:8e:17:69 inside the capture's attack window,
  excluding TCP/1883 (the benign MQTT client shares that NIC). The attack window is the span
  [first, last packet time] of the spoofed units, i.e. units from the attacker MAC whose
  initiator IP is neither 192.168.0.170 nor a known benign IP of that NIC.
* MITM: ARP (L2 unit, ethertype 0x0806) sent by the attacker MAC (zero-payload: a metadata
  estimand).
* class = the file's class for malicious units, "Normal" otherwise; everything else in attack
  captures (mDNS, NTP, ARP, IPv6, CDN) and every unit of a normal capture is benign.
* Incident = (class, attacker) run of content-bearing units, split at gaps > 600 s.
"""
from __future__ import annotations

import polars as pl

ATTACKER_IP = "192.168.0.170"
ATTACKER_MAC = "d8:f2:ca:8e:17:69"
NIC_BENIGN_IPS = ("192.168.0.101", "192.168.1.101", "192.168.2.194", "192.168.3.18", "192.168.4.73",
                  "192.168.5.47", "192.168.6.56", "192.168.7.62", "192.168.8.163")
GAP_S = 600.0


def label_source(units: pl.DataFrame, src: dict) -> tuple[pl.DataFrame, dict]:
    cls = src.get("class") or "Normal"
    role = src.get("role")
    u = units.select("unit_gid", "pay", "proto", "ip_a", "ip_b", "port_a", "port_b", "mac_a", "first_ts_us",
                     "last_ts_us", "device_id")
    mal = pl.lit(False)
    window = None
    if role == "attack":
        mal = (pl.col("ip_a") == ATTACKER_IP) | (pl.col("ip_b") == ATTACKER_IP)
        if src.get("spoofed_flood"):
            sp = u.filter((pl.col("mac_a") == ATTACKER_MAC) & (pl.col("proto") != 0)
                          & ~pl.col("ip_a").is_in([ATTACKER_IP, *NIC_BENIGN_IPS]))
            if sp.height:
                window = (int(sp["first_ts_us"].min()), int(sp["last_ts_us"].max()))
                in_win = (pl.col("last_ts_us") >= window[0]) & (pl.col("first_ts_us") <= window[1])
                mqtt = (pl.col("proto") == 6) & ((pl.col("port_a") == 1883) | (pl.col("port_b") == 1883))
                mal = mal | ((pl.col("mac_a") == ATTACKER_MAC) & (pl.col("proto") != 0) & in_win & ~mqtt)
        if cls == "MITM":
            mal = mal | ((pl.col("proto") == 0) & (pl.col("ip_a") == ATTACKER_MAC) & (pl.col("port_a") == 0x0806))
    lab = u.with_columns(mal.alias("is_malicious")).with_columns(
        pl.when(pl.col("is_malicious")).then(pl.lit("Malicious")).otherwise(pl.lit("Benign")).alias("label"),
        pl.when(pl.col("is_malicious")).then(pl.lit(cls)).otherwise(pl.lit("Normal")).alias("detailed_label"),
        pl.lit("attacker_rule" if role == "attack" else "normal_capture").alias("label_source"),
        pl.lit(None, dtype=pl.String).alias("conn_uid"), pl.lit(None, dtype=pl.Float64).alias("match_dstart_s"))
    lab = lab.with_columns(pl.col("detailed_label").alias("family"), pl.col("detailed_label").alias("class"))
    cb = (lab.filter(pl.col("is_malicious") & pl.col("pay")).select("unit_gid", "first_ts_us").sort("first_ts_us"))
    if cb.height:
        cb = cb.with_columns((pl.col("first_ts_us").diff().fill_null(0) > GAP_S * 1e6).cast(pl.Int32).cum_sum()
                             .alias("k"))
        cb = cb.with_columns((pl.lit(f"{src['source']}|{cls}|{ATTACKER_IP}|") + pl.col("k").cast(pl.String))
                             .alias("incident_id"))
        lab = lab.join(cb.select("unit_gid", "incident_id"), on="unit_gid", how="left")
    else:
        lab = lab.with_columns(pl.lit(None, dtype=pl.String).alias("incident_id"))
    pay = lab.filter(pl.col("pay"))
    qa = {"units": lab.height, "payload_units": pay.height, "class": cls, "role": role,
          "malicious_units": int(lab["is_malicious"].sum()), "malicious_payload_units": int(pay["is_malicious"].sum()),
          "attack_window_us": window, "match_rate_payload": 1.0, "match_rate_all": 1.0}
    keep = ["unit_gid", "pay", "device_id", "label", "detailed_label", "label_source", "conn_uid", "match_dstart_s",
            "family", "class", "is_malicious", "incident_id"]
    return lab.select(keep), qa
