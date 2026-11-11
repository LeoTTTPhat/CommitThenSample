"""CIC-IoT2023 benign background: every unit is benign (DESIGN §5.5)."""
from __future__ import annotations

import polars as pl


def label_source(units: pl.DataFrame, src: dict) -> tuple[pl.DataFrame, dict]:
    lab = units.select("unit_gid", "pay", "device_id").with_columns(
        pl.lit("Benign").alias("label"), pl.lit("-").alias("detailed_label"),
        pl.lit("cic_benign_all").alias("label_source"), pl.lit(None, dtype=pl.String).alias("conn_uid"),
        pl.lit(None, dtype=pl.Float64).alias("match_dstart_s"), pl.lit(None, dtype=pl.String).alias("family"),
        pl.lit("Benign").alias("class"), pl.lit(False).alias("is_malicious"),
        pl.lit(None, dtype=pl.String).alias("incident_id"))
    return lab, {"units": lab.height, "payload_units": int(lab["pay"].sum()), "match_rate_payload": 1.0,
                 "match_rate_all": 1.0}
