"""Low-prevalence gateway streams at table level (DESIGN §5.2): windows, whole-epoch offsets, address remap,
thinning to a target prevalence, online rolling features over the merged stream, StreamTables output.

``build_stream(spec)`` (spec from streams/recipe.py):

1. Background window (CIC half) and foreground windows are read from the S1/S3 tables (no pcap parsing);
   every source keeps its own epoch grid (anchored at its first packet); a window [e0, e0+n) of source
   epochs is placed at stream epochs [s0, s0+n) by shifting every timestamp by the whole-epoch offset
   Δ = (t0_stream + s0 T) − (t0_src + e0 T). Unit boundaries, f values and Suricata results are reused.
2. Thinning: every malicious payload connection (flow_uid with >= 1 malicious payload unit) of a thinned
   foreground is kept independently with probability q = min(1, p N_ben / ((1 − p) N_mal)), N_mal = its
   malicious payload units, N_ben = all other payload units of the merged stream (= the background's
   payload units when the foreground has no benign payload units; DESIGN writes N^bg_pay). Uniforms come
   from Generator(PCG64(SeedSequence(spec.seed))) in order of first appearance in the stream. Every segment
   of a dropped connection is removed; benign, unknown and zero-payload foreground units are kept.
3. Address remap: the configured IPs (seed host / attacker / victim) are rewritten in the flow keys, the
   configured MACs (and, for IoT-23, every other foreground MAC) in device identities and L2 keys; device
   ids are re-densified by first appearance in the merged stream (a virtual capture's flow table would do
   the same). The direction class is recomputed (units/features.side_and_dir on the rewritten key; the
   remap targets are LAN addresses, so it is unchanged). enc(m_i) is re-encoded with the new key,
   device id, shifted timestamps and the recomputed flow_uid = BLAKE3("CTS/v1/flow" ‖ key ‖ first_ts)[:16].
4. Canonical order inside every stream epoch: (first_ts_us, key bytes); features via
   units.features.features_batch over the merged stream (rolling statistics online over the merged stream).
5. Output (results/streams/<stream_id>/): StreamTables files (sim/tables.py; queries = detector estimands,
   Suricata families when S2 is ready, label estimands), plus features_pay.npy (payload units, 38) int32, cols_pay.npz (their decoded metadata columns), units.parquet
   (per stream unit: source, unit_gid, re-encoded meta, flow ids, device MAC), meta.json "extra" with the
   recipe, seeds, q, realised prevalence, payload units/epoch, phi0 inputs and the incident table.

Recipe variant (rev1 sensitivity, review R3-M1(a); only when ``spec.extra["zero_thin"]`` is set, see
``thin_zero_payload`` and recipe.zero_variant_spec): after step 2, the pure zero-payload connections of the
configured seed host(s) are thinned too (variant zq: keep probability = the stream's payload q; z0: all removed),
with an independent child seed stream, so the payload keep decisions stay bit-identical to the original recipe.
Without the key the build is exactly the original recipe.
"""
from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path

import numpy as np

from ..sim.tables import StreamTables, write_tables
from ..units import features as F
from ..units._hash_shim import flow_uid as _flow_uid
from ..utils.io import RESULTS, atomic_save_npy, atomic_write_json, atomic_write_parquet, is_done
from .recipe import (DET_QUERIES, EDGE_L_CLASSES, FAMILY_CODES, IOT23_L, S_QUERIES, StreamSpec, load_source,
                     query_catalogue, source_t0)

__all__ = ["build_stream", "thin_connections", "remap_ip16", "mac_bytes", "ip16", "STREAMS_DIR", "add_utilities",
           "load_stream_extra", "incidents_from_units", "thin_zero_payload", "zero_keep_prob", "ZERO_SPAWN_KEY"]

STREAMS_DIR = RESULTS / "streams"
_TABLE_QUERIES = DET_QUERIES + S_QUERIES
T_US = 30_000_000
GAP_US = 600 * 1_000_000


def ip16(txt: str) -> bytes:
    import ipaddress
    a = ipaddress.ip_address(txt)
    return (b"\x00" * 10 + b"\xff\xff" + a.packed) if a.version == 4 else a.packed


def mac_bytes(txt: str) -> bytes:
    return bytes(int(x, 16) for x in txt.split(":"))


def remap_ip16(arr: np.ndarray, old: bytes, new: bytes) -> int:
    """In-place rewrite of 16-byte addresses equal to ``old``; returns the number of rows changed."""
    o = np.frombuffer(old, dtype=np.uint8)
    m = np.all(arr == o, axis=1)
    arr[m] = np.frombuffer(new, dtype=np.uint8)
    return int(m.sum())


# ------------------------------------------------------------------------------------------ #
def thin_connections(conn_first_order: np.ndarray, n_mal_units: int, n_benign_pay: int, p: float, seed) -> dict:
    """Keep-mask over malicious payload connections (given in their stream order)."""
    n_conn = len(conn_first_order)
    if n_mal_units <= 0 or n_conn == 0:
        return {"q": 1.0, "keep": np.ones(n_conn, dtype=bool), "u": np.zeros(n_conn)}
    q = min(1.0, p * n_benign_pay / ((1.0 - p) * n_mal_units))
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(seed))))
    u = rng.random(n_conn)
    return {"q": float(q), "keep": u < q, "u": u}


# [variant] zero-payload keep decisions: Generator(PCG64(SeedSequence(seed, spawn_key=(0,)))) = the first child
# (SeedSequence(seed).spawn(1)[0]) of the payload-thinning seed: an independent stream, the payload draws unchanged
ZERO_SPAWN_KEY = (0,)


def zero_keep_prob(zt: dict, q_stream: float) -> float:
    """Keep probability of the seed host's pure zero-payload connections: 'q' = the stream's payload q."""
    k = zt.get("keep", "q")
    return float(q_stream) if k == "q" else float(k)


def _label_counts(t, mask: np.ndarray) -> dict:
    import polars as pl
    lab = t["label"].fill_null("unknown").filter(pl.Series(mask))
    return {str(r["label"]): int(r["len"]) for r in lab.to_frame("label").group_by("label").len().to_dicts()}


def thin_zero_payload(parts, zt: dict, q_stream: float, seed) -> dict:
    """[variant, review R3-M1(a)] Thin the pure zero-payload connections of the configured seed host(s).

    Candidates = units of a configured part (``zt["sources"]``: source -> device MACs before the remap) of a seed
    MAC that carry no payload and whose connection (flow_uid) has no payload unit in the part: scans, failed
    handshakes, ICMP, L2. Zero-payload segments of payload connections follow their connection (malicious ones
    were thinned with q by the payload rule; the others are kept). Each candidate connection is kept independently
    with probability ``zero_keep_prob`` (zq: the stream's payload q; z0: 0); uniforms from
    Generator(PCG64(SeedSequence(seed, spawn_key=ZERO_SPAWN_KEY))) in order of first appearance in the stream
    (stream epoch, shifted first_ts, part, row), so the payload keep decisions (parent stream) are untouched.
    Every unit of a dropped connection is removed. Mutates the parts (frame, sepoch, malpay); returns a report."""
    import polars as pl
    keep_p = zero_keep_prob(zt, q_stream)
    srcs = zt.get("sources", {}) or {}
    work, keys = [], []
    for i, P in enumerate(parts):
        macs = srcs.get(P["source"])
        if not macs:
            continue
        t = P["frame"]
        seedm = t["device_mac"].fill_null("").is_in(list(macs)).to_numpy().astype(bool)
        pay = t["pay"].to_numpy().astype(bool)
        fu = t["flow_uid"]
        conn_pay = fu.is_in(fu.filter(pl.Series(pay)).unique().implode()).to_numpy().astype(bool)
        conn_mal = fu.is_in(fu.filter(pl.Series(P["malpay"])).unique().implode()).to_numpy().astype(bool)
        cand = seedm & ~pay & ~conn_pay
        rows = np.nonzero(cand)[0]
        g = (pl.DataFrame({"r": rows.astype(np.int64), "fu": fu.gather(rows)})
             .group_by("fu", maintain_order=True).agg(pl.col("r").first()))
        r_first = g["r"].to_numpy().astype(np.int64)
        fts = t["first_ts_us"].to_numpy().astype(np.int64)
        keys.append(np.stack([P["sepoch"][r_first].astype(np.int64), fts[r_first] + int(P["delta_us"]),
                              np.full(len(r_first), i, np.int64), r_first], axis=1))
        work.append({"i": i, "seedm": seedm, "pay": pay, "conn_pay": conn_pay, "conn_mal": conn_mal, "cand": cand,
                     "conn_fu": g["fu"], "n_conn": len(r_first)})
    K = np.concatenate(keys) if keys else np.zeros((0, 4), np.int64)
    order = np.lexsort((K[:, 3], K[:, 2], K[:, 1], K[:, 0])) if len(K) else np.zeros(0, np.int64)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(seed), spawn_key=ZERO_SPAWN_KEY)))
    u_sorted = rng.random(len(order))
    u = np.empty(len(order))
    u[order] = u_sorted                       # u[j] = uniform of the j-th connection (part-major listing)
    keep_all = u < keep_p
    rep_parts, off = [], 0
    for w in work:
        P = parts[w["i"]]
        t = P["frame"]
        kc = keep_all[off:off + w["n_conn"]]
        off += w["n_conn"]
        drop_fu = w["conn_fu"].filter(pl.Series(~kc))
        drop = w["cand"] & t["flow_uid"].is_in(drop_fu.implode()).to_numpy().astype(bool)
        s_tab = t["s"].fill_null(0).to_numpy().astype(np.int64)
        zero = w["seedm"] & ~w["pay"]
        rep = {"source": P["source"], "macs": list(srcs[P["source"]]), "n_seed_units": int(w["seedm"].sum()),
               "n_seed_payload_units": int((w["seedm"] & w["pay"]).sum()),
               "n_seed_zero_units_before": int(zero.sum()), "n_seed_zero_units_after": int((zero & ~drop).sum()),
               "n_zero_in_mal_payload_conns": int((zero & w["conn_mal"]).sum()),
               "n_zero_in_other_payload_conns": int((zero & w["conn_pay"] & ~w["conn_mal"]).sum()),
               "n_pure_zero_conns_before": int(w["n_conn"]), "n_pure_zero_conns_after": int(kc.sum()),
               "n_pure_zero_units_before": int(w["cand"].sum()), "n_pure_zero_units_after": int((w["cand"] & ~drop).sum()),
               "pure_zero_bytes_before": int(s_tab[w["cand"]].sum()), "pure_zero_bytes_after": int(s_tab[w["cand"] & ~drop].sum()),
               "pure_zero_labels_before": _label_counts(t, w["cand"]),
               "pure_zero_labels_after": _label_counts(t, w["cand"] & ~drop), "n_zero_dropped_units": int(drop.sum())}
        if drop.any():
            P["frame"] = t.filter(pl.Series(~drop))
            P["sepoch"] = P["sepoch"][~drop]
            P["malpay"] = P["malpay"][~drop]
        P["n_zero_dropped_units"] = int(drop.sum())
        rep_parts.append(rep)
    n_c = int(len(keep_all))
    return {"variant": zt.get("variant"), "keep_rule": zt.get("keep", "q"), "keep_prob": keep_p,
            "q_stream": float(q_stream), "seed_rule": f"SeedSequence(seed={list(seed)}, spawn_key={ZERO_SPAWN_KEY})",
            "n_pure_zero_conns": n_c, "n_pure_zero_conns_kept": int(keep_all.sum()),
            "realised_keep_rate": (float(keep_all.mean()) if n_c else None),
            "n_units_dropped": int(sum(r["n_zero_dropped_units"] for r in rep_parts)), "parts": rep_parts,
            "base_stream_id": zt.get("base_stream_id"), "stable_id_of": zt.get("stable_id_of")}


def incidents_from_units(device, family, first_ts, content, by_device: bool = True, gap_us: int = GAP_US):
    """Incident ids (dense, -1 = none) of content-bearing units: runs of (device, family) [or family only]
    in first_ts order split at gaps > 600 s (the S3 rule of labels/iot23.py, applied to the merged stream).
    Ids are numbered by first appearance in the stream. Returns (ids, table)."""
    n = len(first_ts)
    inc = np.full(n, -1, dtype=np.int64)
    idx = np.nonzero(np.asarray(content, dtype=bool))[0]
    if not len(idx):
        return inc, []
    dev = np.asarray(device)[idx] if by_device else np.zeros(len(idx), np.int64)
    fam = np.asarray(family)[idx]
    ts = np.asarray(first_ts, dtype=np.int64)[idx]
    _, fam_code = np.unique(fam, return_inverse=True)
    fam_code = fam_code.ravel()
    o = np.lexsort((idx, ts, fam_code, dev))
    dev_o, fam_o, ts_o, pos_o = dev[o], fam_code[o], ts[o], idx[o]
    new_grp = np.ones(len(o), dtype=bool)
    new_grp[1:] = (dev_o[1:] != dev_o[:-1]) | (fam_o[1:] != fam_o[:-1]) | (np.diff(ts_o) > gap_us)
    run = np.cumsum(new_grp) - 1
    n_run = int(run[-1]) + 1
    first_pos = np.full(n_run, np.iinfo(np.int64).max)
    np.minimum.at(first_pos, run, pos_o)
    last_pos = np.full(n_run, -1)
    np.maximum.at(last_pos, run, pos_o)
    k = np.bincount(run, minlength=n_run)
    rank = np.empty(n_run, dtype=np.int64)
    rank[np.argsort(first_pos, kind="stable")] = np.arange(n_run)
    inc[pos_o] = rank[run]
    table = []
    for r in np.argsort(first_pos, kind="stable"):
        fp = int(first_pos[r])
        table.append({"incident": int(rank[r]), "k": int(k[r]), "first_unit": fp, "last_unit": int(last_pos[r]),
                      "device": int(np.asarray(device)[fp]), "family": str(np.asarray(family)[fp])})
    return inc, table


# ------------------------------------------------------------------------------------------ #
def _part_frames(spec: StreamSpec, log):
    """Load background + foreground windows; returns list of dicts (frame, meta about the part)."""
    import polars as pl
    from .recipe import GOTHAM_L
    parts = []
    if spec.background is not None:
        b = spec.background
        t, info = load_source(b["ds"], b["source"], b["src_epoch0"], b["src_epoch0"] + spec.n_epochs, spec.label_scheme,
                              queries=_TABLE_QUERIES)
        parts.append({"role": "background", "ds": b["ds"], "source": b["source"], "src_epoch0": b["src_epoch0"],
                      "stream_epoch0": 0, "thin": False, "ip_map": {}, "mac_map": {}, "mac_other": None, "k": 0,
                      "frame": t, "info": info})
        log(f"[lp] background {b['source']}: {t.height} units ({info})")
    for fg in spec.foregrounds:
        hi = fg.src_epoch0 + min(fg.n_epochs, spec.n_epochs - fg.stream_epoch0)
        t, info = load_source(fg.ds, fg.source, fg.src_epoch0, hi, spec.label_scheme,
                              queries=_TABLE_QUERIES + (tuple(GOTHAM_L) + ("L1",) if spec.label_scheme == "gotham" else ()))
        if getattr(fg, "drop_unknown", False):
            n0 = t.height
            t = t.filter(pl.col("label").fill_null("unknown") != "unknown")
            info = dict(info, dropped_unknown=n0 - t.height)
        parts.append({"role": fg.role, "ds": fg.ds, "source": fg.source, "src_epoch0": fg.src_epoch0,
                      "stream_epoch0": fg.stream_epoch0, "thin": fg.thin, "ip_map": fg.ip_map, "mac_map": fg.mac_map,
                      "mac_other": fg.mac_other, "k": fg.k, "frame": t, "info": info})
        log(f"[lp] foreground {fg.source}: epochs [{fg.src_epoch0}, {hi}) -> {t.height} units ({info})")
    return parts


def _label_cols(pay, lab, mal, fam, cls, scheme: str):
    """L estimands from the labels (identical to scripts/03_build_tables.py)."""
    known = lab != "unknown"
    out = {"L1": ((pay & mal & known).astype(np.int64), (pay & known).astype(np.int64))}
    if scheme == "iot23":
        for k, f in IOT23_L.items():
            out[k] = ((pay & mal & known & (fam == f)).astype(np.int64), (pay & known).astype(np.int64))
    elif scheme == "edge":
        for c in EDGE_L_CLASSES:
            out[f"L_{c}"] = ((pay & mal & (cls == c)).astype(np.int64), pay.astype(np.int64))
    elif scheme == "gotham":
        from .recipe import GOTHAM_L
        for k, c in GOTHAM_L.items():
            out[k] = ((pay & mal & known & (cls == c)).astype(np.int64), (pay & known).astype(np.int64))
    return out


def build_stream(spec: StreamSpec, out_root: Path = STREAMS_DIR, force: bool = False, log=print,
                 write_units: bool = True) -> dict:
    import polars as pl
    out = Path(out_root) / spec.stream_id
    if is_done(out) and not force:
        return json.loads((out / "meta.json").read_text())
    t_start = time.time()
    parts = _part_frames(spec, log)
    # ---- stream time origin: background (or first foreground) window start
    p0 = parts[0]
    t0_stream = source_t0(p0["ds"], p0["source"]) + (p0["src_epoch0"] - p0["stream_epoch0"]) * T_US
    for P in parts:
        t = P["frame"]
        se = t["epoch"].to_numpy().astype(np.int64) - P["src_epoch0"] + P["stream_epoch0"]
        keep = (se >= 0) & (se < spec.n_epochs)
        if not keep.all():
            t = t.filter(pl.Series(keep))
            se = se[keep]
        P["frame"], P["sepoch"] = t, se
        P["delta_us"] = (t0_stream + P["stream_epoch0"] * T_US) - (source_t0(P["ds"], P["source"])
                                                                   + P["src_epoch0"] * T_US)
    # ---- thinning (malicious payload connections of thinned foregrounds)
    n_ben_pay = 0
    all_conns = []
    for i, P in enumerate(parts):
        t = P["frame"]
        pay = t["pay"].to_numpy().astype(bool)
        mal = t["is_malicious"].fill_null(False).to_numpy().astype(bool)
        P["malpay"] = pay & mal
        P["flow_list"] = None
        if P["thin"] and P["malpay"].any():
            fu = t["flow_uid"].to_list()
            fts = t["first_ts_us"].to_numpy()
            seen = set()
            for j in np.nonzero(P["malpay"])[0]:
                c = fu[j]
                if c not in seen:
                    seen.add(c)
                    all_conns.append((int(P["sepoch"][j]), int(fts[j]) + P["delta_us"], i, int(j), c))
            P["flow_list"] = fu
        n_ben_pay += int((pay & ~P["malpay"]).sum()) if P["thin"] else int(pay.sum())
    all_conns.sort()
    n_mal_units = int(sum(int(P["malpay"].sum()) for P in parts if P["thin"]))
    thin = {"q": 1.0, "keep": np.ones(len(all_conns), bool)}
    if spec.p is not None:
        thin = thin_connections(np.arange(len(all_conns)), n_mal_units, n_ben_pay, spec.p, spec.seed)
    dropped = {(c[2], c[4]) for c, k in zip(all_conns, thin["keep"]) if not k}
    for i, P in enumerate(parts):
        P["n_dropped_units"] = 0
        if P["flow_list"] is None or not dropped:
            continue
        fu = P["flow_list"]
        drop = np.fromiter(((i, f) in dropped for f in fu), dtype=bool, count=len(fu))
        if drop.any():
            P["frame"] = P["frame"].filter(pl.Series(~drop))
            P["sepoch"] = P["sepoch"][~drop]
            P["malpay"] = P["malpay"][~drop]
        P["n_dropped_units"] = int(drop.sum())
    # ---- [variant only] thinning of the seed host's pure zero-payload connections (absent: original recipe)
    zt = (spec.extra or {}).get("zero_thin")
    zero_rep = thin_zero_payload(parts, zt, thin["q"], spec.seed) if zt else None
    # ---- remap, shift, decode
    cols_parts, devmac_parts, remap_report = [], [], []
    for i, P in enumerate(parts):
        t = P["frame"]
        c = F.cols_from_meta(t["meta"].to_arrow())
        t = t.drop("meta")
        P["frame"] = t
        rep = {"source": P["source"], "ip": {}, "mac": {}}
        for old, new in P["ip_map"].items():
            o16, n16 = ip16(old), ip16(new)
            na = remap_ip16(c["ip_a"], o16, n16)
            nb = remap_ip16(c["ip_b"], o16, n16)
            rep["ip"][old] = {"new": new, "rows": na + nb}
        dmac = t["device_mac"].fill_null("").to_numpy().astype(object)
        l2 = c["proto"] == 0
        l2rows = np.nonzero(l2)[0]
        l2mac = [":".join(f"{x:02x}" for x in c["ip_a"][r, :6]) for r in l2rows]
        mmap = {}
        jn = 0
        for m in dict.fromkeys(list(dmac) + l2mac):
            if not m or m.startswith("ip:"):
                continue
            if m in P["mac_map"]:
                mmap[m] = P["mac_map"][m]
            elif P["mac_other"]:
                jn += 1
                mmap[m] = P["mac_other"].replace("{j:02x}", f"{jn:02x}")
        rep["mac"] = mmap
        if mmap:
            dmac = np.array([mmap.get(m, m) for m in dmac], dtype=object)
            for r, m in zip(l2rows, l2mac):
                if m in mmap:
                    c["ip_a"][r, :6] = np.frombuffer(mac_bytes(mmap[m]), dtype=np.uint8)
        d = P["delta_us"]
        c["first_ts"] = c["first_ts"] + d
        c["last_ts"] = c["last_ts"] + d
        P["flow_first"] = t["flow_first_ts_us"].to_numpy().astype(np.int64) + d
        cols_parts.append(c)
        devmac_parts.append(dmac)
        remap_report.append(rep)
    if spec.background is not None:                      # remap targets must be unused by the background
        cb = cols_parts[0]
        bg_ips = {bytes(x) for x in np.concatenate([cb["ip_a"], cb["ip_b"]])}
        bg_macs = set(devmac_parts[0]) | {":".join(f"{x:02x}" for x in r[:6]) for r in cb["ip_a"][cb["proto"] == 0]}
        for rep in remap_report[1:]:
            for old, v in rep["ip"].items():
                if ip16(v["new"]) in bg_ips:
                    raise ValueError(f"remap target {v['new']} is used by the background")
            for old, new in rep["mac"].items():
                if new in bg_macs:
                    raise ValueError(f"remap MAC {new} is used by the background")
    # ---- merge in canonical order: (stream epoch, first_ts, key bytes)
    C = F.concat_cols(cols_parts)
    del cols_parts
    sep = np.concatenate([P["sepoch"] for P in parts])
    part_id = np.concatenate([np.full(len(P["sepoch"]), i, dtype=np.int32) for i, P in enumerate(parts)])
    keyb = np.concatenate([C["proto"].astype(np.uint8)[:, None], C["ip_a"],
                           (C["port_a"] & 0xFFFF).astype("<u2").view(np.uint8).reshape(-1, 2), C["ip_b"],
                           (C["port_b"] & 0xFFFF).astype("<u2").view(np.uint8).reshape(-1, 2)], axis=1)
    kw = _key_words(keyb)
    order = np.lexsort([kw[:, j] for j in range(kw.shape[1] - 1, -1, -1)] + [C["first_ts"], sep])
    C = F.take_cols(C, order)
    sep, part_id, keyb = sep[order], part_id[order], keyb[order]
    dmac_all = np.concatenate(devmac_parts)[order].astype(str)
    flow_first = np.concatenate([P["flow_first"] for P in parts])[order]

    def gather(col, dtype=np.int64, fill=0):
        vals = []
        for P in parts:
            t = P["frame"]
            v = t[col].fill_null(fill).to_numpy() if col in t.columns else np.full(t.height, fill)
            vals.append(np.asarray(v, dtype=dtype))
        return np.concatenate(vals)[order]

    def gather_obj(col, fill):
        return np.concatenate([P["frame"][col].fill_null(fill).to_numpy().astype(object) for P in parts])[order]

    # ---- devices: dense by first appearance of the (remapped) device MAC
    _, first_idx, dev_inv = np.unique(dmac_all, return_index=True, return_inverse=True)
    rank = np.argsort(np.argsort(first_idx))
    device = rank[dev_inv.ravel()].astype(np.int64)
    C["device_id"] = device
    # ---- flow uid of payload units recomputed for the rewritten key and the shifted flow start
    pay = C["pay"].astype(bool)
    new_fu = np.zeros((len(sep), 16), dtype=np.uint8)
    for r in np.nonzero(pay)[0]:
        new_fu[r] = np.frombuffer(_flow_uid(bytes(keyb[r]), int(flow_first[r])), dtype=np.uint8)
    C["flow_uid"] = new_fu
    eptr = np.searchsorted(sep, np.arange(spec.n_epochs + 1))
    index = np.arange(len(sep)) - eptr[sep]
    # ---- features over the merged stream
    X = F.features_batch(C, sep)
    _, dirc = F.side_and_dir(C)
    # ---- costs, queries, labels
    s_tab = gather("s", np.int64)
    if not np.array_equal(np.where(pay, s_tab, 0), C["s"]):
        raise RuntimeError("s mismatch between enc(m_i) and the unit table")
    s_all = s_tab.astype(np.float64)
    with_s = all(P["info"].get("suricata_ready", False) for P in parts)
    queries = query_catalogue(spec.label_scheme, with_s)
    labs = gather_obj("label", "unknown").astype(str)
    malv = gather("is_malicious", bool, False)
    famv = gather_obj("family", "other").astype(str)
    clsv = gather_obj("class", "").astype(str)
    Lc = _label_cols(pay, labs, malv, famv, clsv, spec.label_scheme)
    b = np.zeros((len(sep), len(queries)), dtype=np.float32)
    hits = []
    for qi, (q, kind) in enumerate(queries):
        if q in Lc:
            f, bq = Lc[q]
        else:
            f, bq = gather(f"f_{q}", np.int64), gather(f"b_{q}", np.int64)
        bq = np.where(pay, bq, 0)
        f = np.where(pay, f, 0)
        if np.any(f > bq):
            raise RuntimeError(f"{spec.stream_id}: f > b for query {q}")
        b[:, qi] = bq
        u = np.nonzero(f > 0)[0]
        hits.append((u, f[u].astype(np.float64)))
    fam_eff = clsv if spec.label_scheme in ("edge", "gotham") else famv
    known = labs != "unknown"
    lab_code = np.where(known, 0, -1).astype(np.int64)
    mcodes = np.array([FAMILY_CODES.get(f, FAMILY_CODES["other"]) for f in fam_eff], dtype=np.int64)
    lab_code = np.where(malv & known, mcodes, lab_code)
    content = pay & malv & known
    inc, inc_table = incidents_from_units(device, fam_eff, C["first_ts"], content,
                                         by_device=(spec.label_scheme in ("iot23", "gotham")))
    for row in inc_table:
        row["first_epoch"] = int(sep[row["first_unit"]])
        row["last_epoch"] = int(sep[row["last_unit"]])
    # ---- extra files first (DONE is written last by write_tables)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        (out / "DONE").unlink()
    atomic_save_npy(out / "features_pay.npy", X[pay])            # features of the payload units (stream order)
    pidx = np.nonzero(pay)[0]
    _save_npz_plain(out / "cols_pay.npz", **{k: v[pidx] for k, v in C.items()})   # their metadata columns
    if write_units:
        src_names = np.array([P["source"] for P in parts], dtype=object)[part_id].astype(str)
        orig_fu = np.concatenate([np.array(P["frame"]["flow_uid"].to_list() + [b""], dtype=object)[:-1]
                                  for P in parts])[order]
        ut = pl.DataFrame({"stream_index": np.arange(len(sep), dtype=np.int64), "epoch": sep.astype(np.int32),
                           "index": index.astype(np.int32), "source": src_names, "part": part_id,
                           "unit_gid": gather("unit_gid", np.int64), "pay": pay, "s": s_tab,
                           "device": device.astype(np.int32), "device_mac": dmac_all,
                           "dir_class": dirc.astype(np.int8), "first_ts_us": C["first_ts"],
                           "meta": pl.Series(F.encode_cols(C), dtype=pl.Binary),
                           "flow_uid_orig": pl.Series(list(orig_fu), dtype=pl.Binary),
                           "flow_uid": pl.Series([bytes(r) for r in C["flow_uid"]], dtype=pl.Binary)})
        atomic_write_parquet(out / "units.parquet", ut)
    n_pay = int(pay.sum())
    n_malpay = int(content.sum())
    E = spec.n_epochs
    extra = {
        "recipe": spec.to_json(), "seed": list(spec.seed), "target_prevalence": spec.p, "q": float(thin["q"]),
        "n_mal_payload_units_before": n_mal_units, "n_benign_payload_units": n_ben_pay,
        "n_mal_connections": len(all_conns), "n_mal_connections_kept": int(np.sum(thin["keep"])),
        "n_units_dropped": int(sum(P["n_dropped_units"] for P in parts)),
        "realised_prevalence": n_malpay / max(n_pay, 1), "n_units": int(len(sep)), "n_payload_units": n_pay,
        "n_mal_payload_units": n_malpay, "payload_units_per_epoch": n_pay / E, "units_per_epoch": len(sep) / E,
        "sum_s_all": float(s_all.sum()), "sum_s_pay": float(s_all[pay].sum()),
        "s_all_over_s_pay": float(s_all.sum() / max(s_all[pay].sum(), 1.0)),
        "phi0_note": "phi0 = gamma rho sum_s_all / sum_s_pay; the design value comes from the TRAINING LP stream",
        "t0_stream_us": int(t0_stream), "T_us": T_US,
        "parts": [{"role": P["role"], "ds": P["ds"], "source": P["source"], "src_epoch0": int(P["src_epoch0"]),
                   "stream_epoch0": int(P["stream_epoch0"]), "delta_us": int(P["delta_us"]),
                   "n_units": int(len(P["sepoch"])), "tables": P["info"].get("tables"),
                   "suricata_ready": bool(P["info"].get("suricata_ready", False)),
                   "n_dropped_units": int(P["n_dropped_units"]),
                   "dropped_unknown": int(P["info"].get("dropped_unknown", 0))} for P in parts],
        "remap": remap_report, "provisional_tables": any(P["info"].get("tables") != "p1b" for P in parts),
        "suricata_ready": bool(with_s), "feature_version": F.FEATURE_VERSION, "feature_names": F.FEATURE_NAMES,
        "label_codes": FAMILY_CODES, "incidents": inc_table, "n_incidents": len(inc_table),
        "dir_class_counts": np.bincount(dirc, minlength=4).tolist(), "build_s": round(time.time() - t_start, 2)}
    if zero_rep is not None:
        extra["zero_thin"] = zero_rep
    # a variant stream keeps the integer id of its base stream (common random numbers in every MC)
    sid_int = _stable_id(zt.get("stable_id_of") or spec.stream_id) if zt else _stable_id(spec.stream_id)
    write_tables(out, s=s_all, pay=pay, epoch=sep.astype(np.int32), device=device.astype(np.int32), b=b, hits=hits,
                 query_names=[q for q, _ in queries], query_kinds=[k for _, k in queries], label=lab_code,
                 incident=inc, content=content, utils={}, name=spec.stream_id,
                 stream_id=sid_int, n_devices=int(device.max()) + 1 if len(device) else 0,
                 n_epochs=E, extra=extra)
    log(f"[lp] {spec.stream_id}: {len(sep)} units, {n_pay} payload, prevalence {extra['realised_prevalence']:.5f} "
        f"(target {spec.p}), q={thin['q']:.4f}, {len(inc_table)} incidents, {extra['build_s']:.1f}s")
    if zero_rep is not None:
        log(f"[lp] {spec.stream_id}: zero-payload variant {zero_rep['variant']} keep={zero_rep['keep_prob']:.4f}: "
            f"{zero_rep['n_pure_zero_conns_kept']}/{zero_rep['n_pure_zero_conns']} pure zero-payload connections kept, "
            f"{zero_rep['n_units_dropped']} units dropped")
    return json.loads((out / "meta.json").read_text())


def _save_npz_plain(path: Path, **arrays) -> None:
    import os
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _key_words(keyb: np.ndarray) -> np.ndarray:
    """(N, 37) key bytes -> (N, 5) big-endian uint64 words (lexicographic byte order = word order)."""
    n = keyb.shape[0]
    pad = np.zeros((n, 40), dtype=np.uint8)
    pad[:, :keyb.shape[1]] = keyb
    return pad.view(">u8").reshape(n, 5).astype(np.uint64)


def _global_rows(parts, part_id, local):
    offs = np.cumsum([0] + [len(P["sepoch"]) for P in parts])[:-1]
    return offs[part_id] + local


def _stable_id(name: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little") & ((1 << 31) - 1)


# ------------------------------------------------------------------------------------------ #
def load_stream_extra(stream_dir) -> dict:
    return json.loads((Path(stream_dir) / "meta.json").read_text()).get("extra", {})


def add_utilities(stream_dir, utils: dict) -> None:
    """Add util_<name>.npy columns to a complete StreamTables directory (atomic per file; meta.json's
    utility list is rewritten last)."""
    d = Path(stream_dir)
    if not is_done(d):
        raise FileNotFoundError(f"{d} is not a complete stream")
    meta = json.loads((d / "meta.json").read_text())
    n = int(meta["n_units"])
    for k, v in utils.items():
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (n,):
            raise ValueError(f"utility {k} has shape {v.shape}, stream has {n} units")
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError(f"utility {k} must be finite and >= 0")
        atomic_save_npy(d / f"util_{k}.npy", v)
    meta["utilities"] = sorted(set(meta.get("utilities", [])) | set(utils))
    atomic_write_json(d / "meta.json", meta)
