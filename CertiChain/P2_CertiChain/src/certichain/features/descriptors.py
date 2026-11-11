"""phi(I): label-free incident descriptor (42-d) for the bank-vs-deployment two-sample AUC (DESIGN §3.8),
phi-based strata (Mondrian, OPTIONAL) and incident-level diagnostics.

Computed from UNIT_SCHEMA (FEATURE_SAFE columns only), the side table's ``touches_seed`` and
CANDIDATE_SCHEMA; no label, id, absolute time or raw address enters it.

Groups: size (5: log #units, log N, log #records, log span, log N/#units), unit-type group fractions
(13), service-group fractions (6), direction fractions (4), unit intensity summaries (6), relation
fractions among candidates (6), shared-entity summaries (2).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from certichain.io.schema import REL_PIVOT, REL_REVERSE, REL_SAME_ACTOR, REL_SAME_VICTIM, REL_SHARED_EXT

TYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "scan": ("SCAN_H", "SCAN_V", "SWEEP_ICMP"),
    "brute": ("BRUTE_TELNET", "BRUTE_SSH", "BRUTE_FTP", "BRUTE_MQTT", "BRUTE_HTTP"),
    "web": ("WEB_ATTACK",),
    "transfer": ("HTTP_DOWNLOAD", "HTTP_UPLOAD", "SMB_SESSION"),
    "c2": ("C2_PERIODIC", "CONN_FAIL_BURST"),
    "interactive": ("INTERACTIVE_SESSION",),
    "flood": ("FLOOD_SYN", "FLOOD_UDP", "FLOOD_ICMP", "FLOOD_HTTP", "FLOOD_GRE", "FLOOD_ACK"),
    "amp": ("AMPLIFICATION",),
    "dns_anom": ("DNS_ANOMALY",),
    "exfil": ("EXFIL_BULK",),
    "report": ("REPORT_SMALL",),
    "benign_proto": ("MQTT_PUBSUB", "COAP", "RTSP_STREAM", "DNS_ACTIVITY", "TLS_SESSION"),
    "benign_generic": ("BENIGN_GENERIC_TCP", "BENIGN_GENERIC_UDP", "BENIGN_GENERIC_TCP_SYS", "BENIGN_GENERIC_TCP_HIGH",
                       "BENIGN_GENERIC_UDP_SYS", "BENIGN_GENERIC_UDP_HIGH"),
}
SVC_GROUPS: dict[str, tuple[str, ...]] = {
    "remote": ("telnet", "ssh", "ftp"), "web": ("http", "https"), "dns": ("dns", "mdns"),
    "iot": ("mqtt", "coap", "rtsp", "modbus"), "other": ("other_tcp", "other_udp", "other"),
    "icmp": ("icmp", "gre"),
}
DIRS = ("int->int", "int->ext", "ext->int", "ext->ext")
RELS = {"pivot": REL_PIVOT, "reverse": REL_REVERSE, "same_actor": REL_SAME_ACTOR,
        "same_victim": REL_SAME_VICTIM, "shared_ext": REL_SHARED_EXT}

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "d_lunits", "d_lN", "d_lrecords", "d_lspan", "d_ldensity",
    *(f"d_type_{g}" for g in TYPE_GROUPS), *(f"d_svc_{g}" for g in SVC_GROUPS),
    *(f"d_dir_{d.replace('->', '2')}" for d in DIRS),
    "d_fanout", "d_fail", "d_lbytes", "d_startcv", "d_seed", "d_content",
    *(f"d_rel_{k}" for k in RELS), "d_relcount", "d_lentdeg", "d_entseed",
)


def type_group(t: str) -> str | None:
    """Group of a unit type; unknown (newer typer) names fall back to their prefix family."""
    for g, members in TYPE_GROUPS.items():
        if t in members:
            return g
    for pre, g in (("BENIGN_GENERIC", "benign_generic"), ("FLOOD_", "flood"), ("BRUTE_", "brute"), ("SCAN_", "scan"),
                   ("C2", "c2"), ("SMB", "transfer")):
        if t.startswith(pre):
            return g
    return None


def descriptor(units: pl.DataFrame, cands: pl.DataFrame, side: pl.DataFrame | None = None) -> np.ndarray:
    """phi(I) as a float64 vector in DESCRIPTOR_NAMES order (NaN-free)."""
    nu = int(units.height)
    N = int(cands.height)
    out: list[float] = []
    if nu:
        span = float(units["te"].max() - units["ts"].min())     # duration only (no absolute time)
        nrec = float(units["n_records"].fill_null(1).sum())
    else:
        span, nrec = 0.0, 0.0
    out += [np.log1p(nu), np.log1p(N), np.log1p(nrec), np.log1p(max(span, 0.0)), np.log1p(N / max(nu, 1))]
    types = [type_group(t) for t in units["type"].fill_null("").to_list()] if nu else []
    for g in TYPE_GROUPS:
        out.append(sum(t == g for t in types) / max(nu, 1))
    svcs = units["svc_class"].fill_null("").to_list() if nu else []
    for g, members in SVC_GROUPS.items():
        out.append(sum(s in members for s in svcs) / max(nu, 1))
    dirs = units["dir_class"].fill_null("").to_list() if nu else []
    for d in DIRS:
        out.append(sum(x == d for x in dirs) / max(nu, 1))
    if nu:
        fan = float(units["fanout"].fill_null(False).cast(pl.Float64).mean())
        fail = float(units["fail_frac"].fill_null(0.0).mean())
        lb = float(np.log1p((units["obytes"].fill_null(0) + units["rbytes"].fill_null(0)).cast(pl.Float64)
                            .to_numpy()).mean())
        cv = units["start_cv"].fill_nan(None).drop_nulls()
        cvm = float(cv.mean()) if cv.len() else 0.0
        seed = float(side["touches_seed"].cast(pl.Float64).mean()) if (side is not None and side.height
                                                                      and "touches_seed" in side.columns) else 0.0
        cont = float((units["content_flags"].fill_null(0) != 0).cast(pl.Float64).mean())
    else:
        fan = fail = lb = cvm = seed = cont = 0.0
    out += [fan, fail, lb, cvm, seed, cont]
    if N:
        rm = cands["rel_mask"].fill_null(0).to_numpy().astype(np.int64)
        cnt = np.zeros(N)
        for k, b in RELS.items():
            f = ((rm & b) != 0).astype(np.float64)
            out.append(float(f.mean()))
            cnt += f
        out.append(float(cnt.mean()))
        out.append(float(np.log1p(cands["shared_entity_deg"].fill_null(0).cast(pl.Float64).to_numpy()).mean()))
        out.append(float(cands["shared_is_seed"].fill_null(False).cast(pl.Float64).mean()))
    else:
        out += [0.0] * (len(RELS) + 3)
    v = np.asarray(out, dtype=np.float64)
    v[~np.isfinite(v)] = 0.0
    assert v.size == len(DESCRIPTOR_NAMES)
    return v
