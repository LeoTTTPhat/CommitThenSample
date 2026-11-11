"""Suricata ``eve.json`` content enrichment (SHOULD; DESIGN §3.1.2 -- NOT on the MUST path).

Stub. Not implemented in wave 1: no MUST deliverable depends on it, and the IoT-23/Gotham
record tables are complete from the Zeek logs and tshark CSVs without it. Content fields in
RECORD_SCHEMA (``dns_q``, ``http_*``, ``tls_sni``) are populated from the TON processed CSVs
and TON normal Bro logs today; for IoT-23 and the Gotham pcaps they stay NULL.

Planned shape when implemented (Mon 19, after throughput is measured, §3.1.2):
    * Input: Suricata 8.0.7 eve.json over the 17 IoT-23 pcaps and the Gotham mirai-infection
      pcaps, using ``data/rules/suricata.yaml`` (ET Open 11313). Pre-split pcaps > 2 GB with
      ``editcap -c 5000000`` so an exit-75 restart loses at most one slice.
    * Events used: ``flow``, ``http``, ``dns``, ``tls``, ``mqtt``, ``modbus``.
    * Join: to the conn/flow records on (5-tuple, |Δts| <= 1 s); one content row per flow
      (first by ts), written as an enrichment side table keyed by rec_id, then merged into the
      record content columns. No label columns (quarantine still applies).
    * Output: ``results/interim/<ds>/content/<unit>.parquet`` (rec_id + content columns),
      and a merged records variant selected by config.

Keep the public names below stable so scripts can import them once this lands.
"""
from __future__ import annotations

import os

import polars as pl

from certichain.io.schema import RECORD_SCHEMA

# Content columns this enricher is responsible for (subset of RECORD_SCHEMA).
CONTENT_COLUMNS = ["dns_q", "dns_rcode", "http_method", "http_uri", "http_host", "http_ua", "http_status",
                   "http_resp_mime", "tls_sni"]
EVE_EVENT_TYPES = ("flow", "http", "dns", "tls", "mqtt", "modbus")
JOIN_DELTA_S = 1.0

_NOT_IMPLEMENTED = ("suricata_eve is a SHOULD stub (DESIGN §3.1.2); not implemented in wave 1. "
                    "Content fields come from the TON CSVs/Bro logs; IoT-23 and Gotham content stays NULL.")


def parse_eve(path: str | os.PathLike) -> pl.DataFrame:  # pragma: no cover - stub
    raise NotImplementedError(_NOT_IMPLEMENTED)


def enrich_records(records: pl.DataFrame, eve: pl.DataFrame, delta_s: float = JOIN_DELTA_S) -> pl.DataFrame:  # pragma: no cover
    raise NotImplementedError(_NOT_IMPLEMENTED)


def is_implemented() -> bool:
    return False
