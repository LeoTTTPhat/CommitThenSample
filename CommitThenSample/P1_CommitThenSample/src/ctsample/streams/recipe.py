"""Stream recipes (DESIGN §5.2) and the per-source table loader used by streams/lp.py.

Recipes are pure data (``StreamSpec`` / ``Foreground``) derived from ``config/streams.yaml`` and
``results/inventory.json``; ``streams.lp.build_stream`` turns a spec into a StreamTables directory.

Per-source unit table (``load_source``): S1 units (metadata, flow_uid, s, pay, ...) joined on unit_gid
with the S3 table ``results/tables/<ds>/<src>.parquet`` (f_q, b_q per estimand, labels). If the S3 table is
not there yet (or was written before S2 finished), an equivalent table is built in memory from the S1 hits,
the S2 alerts with the pooled subset-audit sids and the S3 labels, following scripts/03_build_tables.py
exactly; such streams are flagged ``provisional_tables`` in meta.json and should be rebuilt (--force) once
P1-B's tables exist.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..utils.io import P1_ROOT, RESULTS

__all__ = ["CONFIG_PATH", "load_config", "load_inventory", "Foreground", "StreamSpec", "DET_QUERIES", "S_QUERIES",
           "IOT23_L", "EDGE_L_CLASSES", "query_catalogue", "load_source", "source_t0", "source_n_epochs",
           "lp_folds", "fold_k", "lp_iot23_window", "lp_iot23_spec", "lp_edge_spec", "train_lp_spec",
           "training_stream_ids", "lp_gotham_spec", "GOTHAM_L",
           "natural_spec", "all_specs", "p_ppm", "stream_name", "FAMILY_CODES",
           "ZERO_VARIANT_KINDS", "zero_variant_cfg", "variant_stream_id", "zero_variant_spec", "variant_specs",
           "seed_host"]

CONFIG_PATH = P1_ROOT / "config" / "streams.yaml"
T_US = 30_000_000

DET_QUERIES = ("D1o", "D1c", "D2o", "D2c", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11o", "D12", "D13")
DET_KINDS = {"D1c": "count", "D2c": "count", "D5": "count", "D6": "count", "D9": "count"}
S_QUERIES = ("S1", "S2", "S3", "S4", "S5", "S6")
IOT23_L = {"L2": "C&C", "L3": "FileDownload", "L4": "Attack", "L5": "Okiru"}
EDGE_L_CLASSES = ("Backdoor", "OS_Fingerprinting", "Password", "Port_Scanning", "Ransomware", "SQL_injection",
                  "Uploading", "Vulnerability_scanner", "XSS")
GOTHAM_L = {"L_Network_Scanning": "Network Scanning", "L_Brute_Force": "Brute Force", "L_Infection": "Infection",
            "L_CC_Communication": "C&C Communication", "L_DoS": "DoS"}
# label.npy codes of the streams (0 benign, -1 unknown, >= 1 malicious family)
FAMILY_CODES = {"C&C": 1, "FileDownload": 2, "Attack": 3, "Okiru": 4, "PortScan": 5,
                **{c: 10 + i for i, c in enumerate(EDGE_L_CLASSES)}, "MITM": 19, "other": 30,
                **{v: 40 + i for i, v in enumerate(GOTHAM_L.values())}}


def load_config(path=CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_inventory(path=RESULTS / "inventory.json") -> dict:
    return json.loads(Path(path).read_text())


def p_ppm(p: float) -> int:
    return int(round(float(p) * 1e6))


def stream_name(kind: str, **kw) -> str:
    if kind == "lp_iot23":
        return f"lp_iot23_{kw['scenario'].replace('CTU-IoT-Malware-Capture-', '')}_p{p_ppm(kw['p'])}"
    if kind == "lp_edge":
        return f"lp_edge_p{p_ppm(kw['p'])}"
    if kind == "train_lp":
        return f"trainlp_{kw['scenario'].replace('CTU-IoT-Malware-Capture-', '')}_p{p_ppm(kw['p'])}"
    if kind == "natural":
        return f"nat_iot23_{kw['scenario'].replace('CTU-IoT-Malware-Capture-', '')}"
    if kind == "lp_gotham":
        return "lp_gotham_nat" if kw["p"] is None else f"lp_gotham_p{p_ppm(kw['p'])}"
    raise ValueError(kind)


def query_catalogue(label_scheme: str, with_s: bool) -> list[tuple[str, str]]:
    """[(query id, kind)] of a stream: detector estimands, Suricata families (if S2 is ready), label estimands."""
    q = [(x, DET_KINDS.get(x, "indicator")) for x in DET_QUERIES]
    if with_s:
        q += [(x, "indicator") for x in S_QUERIES]
    q.append(("L1", "indicator"))
    if label_scheme == "iot23":
        q += [(k, "indicator") for k in IOT23_L]
    elif label_scheme == "edge":
        q += [(f"L_{c}", "indicator") for c in EDGE_L_CLASSES]
    elif label_scheme == "gotham":
        q += [(k, "indicator") for k in GOTHAM_L]
    return q


# ------------------------------------------------------------------------------------------ #
# specs
# ------------------------------------------------------------------------------------------ #
@dataclass
class Foreground:
    ds: str
    source: str
    src_epoch0: int                    # first epoch of the window on the source's own grid
    n_epochs: int                      # window length (epochs)
    stream_epoch0: int                 # where the window starts in the stream
    ip_map: dict = field(default_factory=dict)      # ip text -> ip text
    mac_map: dict = field(default_factory=dict)     # mac text -> mac text (L2 keys and device MACs)
    mac_other: str | None = None       # template for every other device MAC of this source ({j:02x})
    thin: bool = True                  # malicious payload connections are subject to thinning
    role: str = "attack"
    k: int = 0
    drop_unknown: bool = False         # drop units labelled 'unknown' (Gotham: transit copies of other ports)


@dataclass
class StreamSpec:
    stream_id: str
    kind: str                          # lp_iot23 | lp_edge | train_lp | natural
    n_epochs: int
    background: dict | None            # {ds, source, src_epoch0, n_epochs} (placed at stream epoch 0)
    foregrounds: list
    p: float | None                    # target prevalence (None = natural: no thinning)
    seed: list
    label_scheme: str                  # iot23 | edge
    fold: str | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def _src_done(ds: str, source: str) -> dict:
    p = RESULTS / "units" / ds / source / "DONE"
    if not p.exists():
        raise FileNotFoundError(f"S1 units of {ds}/{source} are not complete")
    return json.loads(p.read_text())


def source_t0(ds: str, source: str) -> int:
    return int(_src_done(ds, source)["t0_us"])


def source_n_epochs(ds: str, source: str) -> int:
    q = json.loads((RESULTS / "qa" / ds / f"{source}.json").read_text())
    return int(q["epochs"])


def lp_folds(cfg: dict, inv: dict) -> list[str]:
    """Eligible LOSO scenarios in the configured (numeric) order; must equal the inventory's list."""
    order = list(cfg["lp_iot23"]["fold_order"])
    elig = set(inv["iot23_loso_eligible"])
    if set(order) != elig:
        raise ValueError(f"config fold_order {sorted(order)} != inventory eligible {sorted(elig)}")
    return order


def fold_k(cfg: dict, inv: dict, scenario: str) -> int:
    return lp_folds(cfg, inv).index(scenario) + 1


def _seed_ip_mac(scenario: str, inv: dict) -> tuple[str, str]:
    """The infected host's IP (inventory seed_ip) and its MAC (device_mac of its units)."""
    import polars as pl
    sc = inv["datasets"]["iot23"]["scenarios"][scenario]
    ip = sc["seed_ip"]
    u = pl.scan_parquet(str(RESULTS / "units" / "iot23" / scenario / "part-*.parquet")).select(
        ["ip_a", "ip_b", "lan_side", "device_mac", "proto"])
    dev_ip = pl.when(pl.col("lan_side") == 0).then(pl.col("ip_a")).otherwise(pl.col("ip_b"))
    m = (u.filter((pl.col("proto") != 0) & (dev_ip == ip)).group_by("device_mac").len().sort("len", descending=True)
         .collect())
    if m.height == 0:
        raise ValueError(f"{scenario}: no unit with the seed IP {ip} on the device side")
    return ip, m["device_mac"][0]


def lp_iot23_window(cfg: dict, inv: dict, scenario: str) -> dict:
    lc = cfg["lp_iot23"]
    sc = inv["datasets"]["iot23"]["scenarios"][scenario]
    t_first = sc["first_mal_payload_ts_in_pcap_window"]
    if t_first is None:
        raise ValueError(f"{scenario} has no malicious payload connection")
    t0 = source_t0("iot23", scenario)
    e_first = (int(round(t_first * 1e6)) - t0) // T_US
    e0 = (int(round((t_first - lc["pre_seconds"]) * 1e6)) - t0) // T_US
    clipped = e0 < 0
    e0 = max(0, e0)
    n_src = source_n_epochs("iot23", scenario)
    n = min(int(lc["window_epochs"]), n_src - e0)
    return {"src_epoch0": int(e0), "n_epochs": int(n), "first_mal_payload_ts": t_first,
            "first_mal_epoch": int(e_first), "pre_clipped": bool(clipped), "source_epochs": n_src}


def _iot23_foreground(cfg: dict, inv: dict, scenario: str, stream_epoch0: int = 0) -> tuple[Foreground, dict]:
    lc = cfg["lp_iot23"]
    k = fold_k(cfg, inv, scenario)
    w = lp_iot23_window(cfg, inv, scenario)
    ip, mac = _seed_ip_mac(scenario, inv)
    new_ip = lc["remap_ip"].replace("{230+k}", str(230 + k))
    new_mac = lc["remap_mac_seed"].replace("{k:02x}", f"{k:02x}")
    other = lc["remap_mac_other"].replace("{k:02x}", f"{k:02x}")
    fg = Foreground(ds="iot23", source=scenario, src_epoch0=w["src_epoch0"], n_epochs=w["n_epochs"],
                    stream_epoch0=stream_epoch0, ip_map={ip: new_ip}, mac_map={mac: new_mac}, mac_other=other,
                    thin=True, role="attack", k=k)
    return fg, {**w, "seed_ip": ip, "seed_mac": mac, "remap_ip": new_ip, "remap_mac": new_mac}


def _bg(cfg: dict, which: str) -> dict:
    b = cfg["background"][which]
    n = source_n_epochs(b["dataset"], b["source"])
    return {"ds": b["dataset"], "source": b["source"], "src_epoch0": 0, "n_epochs": n}


def lp_iot23_spec(cfg: dict, inv: dict, scenario: str, p: float) -> StreamSpec:
    k = fold_k(cfg, inv, scenario)
    fg, info = _iot23_foreground(cfg, inv, scenario, int(cfg["lp_iot23"]["stream_epoch0"]))
    bg = _bg(cfg, "test")
    return StreamSpec(stream_id=stream_name("lp_iot23", scenario=scenario, p=p), kind="lp_iot23",
                      n_epochs=bg["n_epochs"], background=bg, foregrounds=[fg], p=float(p), seed=[k, p_ppm(p)],
                      label_scheme="iot23", fold=scenario, extra={"window": info, "k": k})


def train_lp_spec(cfg: dict, inv: dict, scenario: str, p: float) -> StreamSpec:
    """Calibration stream of ONE training scenario: CIC training half + that scenario's LP window, same recipe
    as the test streams (thinning, remap, offsets). A fold's calibration pools the streams of its training
    scenarios (scripts/04_train_utility.py); the zero-shot fold pools all of them."""
    k = fold_k(cfg, inv, scenario)
    fg, info = _iot23_foreground(cfg, inv, scenario, int(cfg["lp_iot23"]["stream_epoch0"]))
    bg = _bg(cfg, "train")
    code = int(cfg["training_lp"]["code_base"]) + k
    return StreamSpec(stream_id=stream_name("train_lp", scenario=scenario, p=p), kind="train_lp",
                      n_epochs=bg["n_epochs"], background=bg, foregrounds=[fg], p=float(p), seed=[code, p_ppm(p)],
                      label_scheme="iot23", fold=scenario, extra={"window": info, "k": k})


def lp_edge_spec(cfg: dict, inv: dict, p: float) -> StreamSpec:
    ec = cfg["lp_edge"]
    ds = ec["dataset"]
    bg = _bg(cfg, "test")
    fgs = []
    for j, s in enumerate(ec["normal_sources"]):
        n = min(int(ec["normal_first_epochs"]), source_n_epochs(ds, s))
        fgs.append(Foreground(ds=ds, source=s, src_epoch0=0, n_epochs=n, stream_epoch0=80 * j, thin=False,
                              role="normal", k=j))
    for j, s in enumerate(ec["attack_sources"]):
        st = 10 + 90 * j
        n = min(int(ec["attack_slot_epochs"]), source_n_epochs(ds, s), bg["n_epochs"] - st)
        fgs.append(Foreground(ds=ds, source=s, src_epoch0=0, n_epochs=n, stream_epoch0=st, ip_map=dict(ec["remap_ip"]),
                              mac_map=dict(ec["remap_mac"]), thin=True, role="attack", k=j))
    for fg in fgs:
        if fg.role == "normal":
            fg.ip_map = dict(ec["remap_ip"])
            fg.mac_map = dict(ec["remap_mac"])
    return StreamSpec(stream_id=stream_name("lp_edge", p=p), kind="lp_edge", n_epochs=bg["n_epochs"], background=bg,
                      foregrounds=fgs, p=float(p), seed=[int(ec["code"]), p_ppm(p)], label_scheme="edge", fold=None,
                      extra={})


def natural_spec(cfg: dict, inv: dict, scenario: str) -> StreamSpec:
    w = lp_iot23_window(cfg, inv, scenario)
    fg = Foreground(ds="iot23", source=scenario, src_epoch0=w["src_epoch0"], n_epochs=w["n_epochs"], stream_epoch0=0,
                    thin=False, role="attack", k=fold_k(cfg, inv, scenario))
    return StreamSpec(stream_id=stream_name("natural", scenario=scenario), kind="natural", n_epochs=w["n_epochs"],
                      background=None, foregrounds=[fg], p=None, seed=[], label_scheme="iot23", fold=scenario,
                      extra={"window": w})


def lp_gotham_spec(cfg: dict, inv: dict, p: float | None) -> StreamSpec:
    """LP-Gotham [SHOULD]: the 78 benign per-device captures (one session, 2.1 h) form the background; the
    mirai-infection session (8 captures) is shifted to start at +30 min, natural prevalence (p None) or thinned.
    Each capture keeps its own epoch grid and is placed at the nearest whole stream epoch (sub-epoch shift
    <= 15 s, the captures of one session started within seconds of each other); `unknown` transit copies are
    dropped; same testbed and IP plan, so no address remap. The stream horizon is the benign session; the
    mirai session is clipped there."""
    gc = cfg.get("lp_gotham", {})
    srcs = inv["datasets"]["gotham"]["sources"]
    ben = sorted([s for s in srcs if s["role"] == "benign"], key=lambda s: s["source"])
    mal = sorted([s for s in srcs if s["source"].startswith("mirai-infection__")], key=lambda s: s["source"])
    t0b = min(source_t0("gotham", s["source"]) for s in ben)
    t1b = max(int(round(s["last_ts"] * 1e6)) for s in ben)
    E = (t1b - t0b) // T_US + 1
    t0m = min(source_t0("gotham", s["source"]) for s in mal)
    shift = int(gc.get("mirai_start_epoch", 60))
    fgs = []
    for j, s in enumerate(ben):
        st = int(round((source_t0("gotham", s["source"]) - t0b) / T_US))
        fgs.append(Foreground(ds="gotham", source=s["source"], src_epoch0=0, n_epochs=source_n_epochs("gotham", s["source"]),
                              stream_epoch0=st, thin=False, role="benign", k=j, drop_unknown=True))
    for j, s in enumerate(mal):
        st = shift + int(round((source_t0("gotham", s["source"]) - t0m) / T_US))
        fgs.append(Foreground(ds="gotham", source=s["source"], src_epoch0=0, n_epochs=source_n_epochs("gotham", s["source"]),
                              stream_epoch0=st, thin=True, role="attack", k=j, drop_unknown=True))
    return StreamSpec(stream_id=stream_name("lp_gotham", p=p), kind="lp_gotham", n_epochs=int(E), background=None,
                      foregrounds=fgs, p=None if p is None else float(p),
                      seed=[] if p is None else [int(gc.get("code", 60)), p_ppm(p)], label_scheme="gotham", fold=None,
                      extra={"mirai_start_epoch": shift})


def all_specs(cfg: dict, inv: dict, kinds=("lp_iot23", "train_lp", "lp_edge", "natural", "lp_gotham")) -> list[StreamSpec]:
    out = []
    folds = lp_folds(cfg, inv)
    ps = [float(x) for x in cfg["prevalences"]]
    if "lp_iot23" in kinds:
        out += [lp_iot23_spec(cfg, inv, s, p) for s in folds for p in ps]
    if "train_lp" in kinds:
        out += [train_lp_spec(cfg, inv, s, p) for s in folds for p in ps]
    if "lp_edge" in kinds:
        out += [lp_edge_spec(cfg, inv, p) for p in ps]
    if "natural" in kinds:
        out += [natural_spec(cfg, inv, s) for s in folds]
    if "lp_gotham" in kinds and cfg.get("lp_gotham", {}).get("enabled", False):
        out += [lp_gotham_spec(cfg, inv, p) for p in cfg["lp_gotham"].get("prevalences", [None, 0.01])]
    return out


# ------------------------------------------------------------------------------------------ #
# per-source table loader
# ------------------------------------------------------------------------------------------ #
UNIT_COLS = ["unit_gid", "epoch", "index", "flow_uid", "proto", "port_a", "port_b", "pay", "fps", "s", "meta",
             "device_id", "device_mac", "first_ts_us", "last_ts_us", "flow_first_ts_us", "pkts0", "pkts1"]
LABEL_COLS = ["unit_gid", "label", "family", "class", "is_malicious", "incident_id", "label_source"]


def _kept_sids(ds: str):
    pa_ = RESULTS / "rules" / "audit_all.json"
    if pa_.exists():
        a = json.loads(pa_.read_text())
        if any(d["dataset"] == ds for d in a.get("datasets_included", [])):
            return set(int(x) for x in a["kept_sids"])
    p = RESULTS / "rules" / f"audit_{ds}.json"
    if not p.exists():
        return None
    return set(int(x) for x in json.loads(p.read_text())["kept_sids"])


def _estimand_frame(ds: str, source: str, u, label_scheme: str):
    """f_/b_ columns for the units frame ``u`` (provisional path, mirrors scripts/03_build_tables.py)."""
    import polars as pl
    from ..content.detectors import DET_ESTIMAND_IDS, eligibility_exprs
    from ..content.suricata import FAMILIES
    gids = u.select("unit_gid")
    hp = RESULTS / "hits" / ds / f"{source}.parquet"
    h = pl.read_parquet(str(hp)) if hp.exists() else pl.DataFrame({"unit_gid": [], "query_id": [], "value": []})
    h = h.filter(pl.col("query_id").is_in(list(DET_ESTIMAND_IDS)))
    if h.height:
        hv = h.pivot(on="query_id", index="unit_gid", values="value", aggregate_function="sum")
    else:
        hv = pl.DataFrame({"unit_gid": pl.Series([], dtype=pl.Int64)})
    t = u.join(hv, on="unit_gid", how="left")
    t = t.with_columns([(pl.col(q) if q in t.columns else pl.lit(0)).fill_null(0).cast(pl.Int32).alias(f"f_{q}")
                        for q in DET_ESTIMAND_IDS])
    t = t.drop([c for c in t.columns if c in set(DET_ESTIMAND_IDS)])
    el = eligibility_exprs()
    t = t.with_columns([v[1].alias(f"b_{k}") for k, v in el.items()])
    kept = _kept_sids(ds)
    ap = RESULTS / "alerts" / ds / f"{source}.parquet"
    ready = kept is not None and ap.exists()
    if ready:
        al = pl.read_parquet(str(ap), columns=["unit_gid", "sid", "family"]).filter(pl.col("unit_gid") >= 0)
        alk = al.filter(pl.col("sid").is_in(sorted(kept)) & pl.col("family").is_not_null())
        fam = (alk.select("unit_gid", "family").unique().with_columns(pl.lit(1, dtype=pl.Int32).alias("v"))
               .pivot(on="family", index="unit_gid", values="v", aggregate_function="max")) if alk.height else \
            pl.DataFrame({"unit_gid": pl.Series([], dtype=pl.Int64)})
        t = t.join(fam, on="unit_gid", how="left")
        t = t.with_columns([(pl.col(f) if f in t.columns else pl.lit(0)).fill_null(0).cast(pl.Int32).alias(f"f_{f}")
                            for f in FAMILIES])
        t = t.drop([f for f in FAMILIES if f in t.columns])
        t = t.with_columns([pl.col("pay").cast(pl.Int64).alias(f"b_{f}") for f in FAMILIES])
    del gids
    return t, ready


def load_source(ds: str, source: str, epoch_lo: int = 0, epoch_hi: int | None = None, label_scheme: str = "iot23",
                prefer_tables: bool = True, queries=None, unit_cols=None):
    """Units of epochs [epoch_lo, epoch_hi) of one source with f_*/b_* and label columns (polars DataFrame).

    Returns (frame, info) with info = {"tables": "p1b"|"provisional", "suricata_ready": bool}."""
    import polars as pl
    lz = pl.scan_parquet(str(RESULTS / "units" / ds / source / "part-*.parquet")).select(list(unit_cols or UNIT_COLS))
    lz = lz.filter(pl.col("epoch") >= epoch_lo)
    if epoch_hi is not None:
        lz = lz.filter(pl.col("epoch") < epoch_hi)
    u = lz.collect().sort(["epoch", "index"])
    lab_p = RESULTS / "labels" / ds / f"{source}.parquet"
    if not lab_p.exists():
        raise FileNotFoundError(f"labels of {ds}/{source} missing (S3 --labels-only)")
    lab = pl.read_parquet(str(lab_p), columns=LABEL_COLS)
    tp = RESULTS / "tables" / ds / f"{source}.parquet"
    info = {}
    use_p1b = False
    if prefer_tables and tp.exists():
        sch = pl.read_parquet_schema(str(tp))
        want = None if queries is None else {f"{x}_{q}" for q in queries for x in ("f", "b")}
        tcols = [c for c in sch if (c.startswith("f_") or c.startswith("b_")) and (want is None or c in want)]
        tcols += ["unit_gid", "suricata_ready"]
        lzt = pl.scan_parquet(str(tp)).filter(pl.col("epoch") >= epoch_lo)
        if epoch_hi is not None:
            lzt = lzt.filter(pl.col("epoch") < epoch_hi)
        tb = lzt.select(tcols).with_columns([pl.col(c).cast(pl.Int32) for c in tcols
                                             if c.startswith("f_") or c.startswith("b_")]).collect()
        ready = bool(tb["suricata_ready"].all()) if tb.height else False
        if ready or _kept_sids(ds) is None:
            use_p1b = True
            t = u.join(tb.drop("suricata_ready"), on="unit_gid", how="left")
            info = {"tables": "p1b", "suricata_ready": ready}
    if not use_p1b:
        t, ready = _estimand_frame(ds, source, u, label_scheme)
        info = {"tables": "provisional", "suricata_ready": ready}
    t = t.join(lab, on="unit_gid", how="left")
    if t.height != u.height:
        raise RuntimeError(f"{ds}/{source}: join changed the row count")
    return t, info


def training_stream_ids(cfg: dict, inv: dict, fold_scenario: str | None, p: float, variant: str | None = None) -> list[str]:
    """Calibration streams of a fold: one per training scenario (every eligible scenario except the test one;
    all of them for the zero-shot fold). ``variant``: the recipe-variant streams (id suffix _<variant>)."""
    return [variant_stream_id(stream_name("train_lp", scenario=s, p=p), variant) for s in lp_folds(cfg, inv)
            if s != fold_scenario]


# ------------------------------------------------------------------------------------------ #
# recipe variants (rev1 sensitivity, review R3-M1(a)): zero-payload thinning of the infected (seed) host
# ------------------------------------------------------------------------------------------ #
ZERO_VARIANT_KINDS = ("lp_iot23", "train_lp")


def zero_variant_cfg(cfg: dict, variant: str) -> dict:
    """The ``zero_payload_variants.<variant>`` block of config/streams.yaml ({keep, kinds, prevalences})."""
    v = (cfg.get("zero_payload_variants") or {}).get(variant)
    if v is None:
        raise ValueError(f"unknown stream recipe variant {variant!r} (config/streams.yaml zero_payload_variants)")
    return v


def variant_stream_id(stream_id: str, variant: str | None) -> str:
    return stream_id if not variant else f"{stream_id}_{variant}"


def seed_host(scenario: str, inv: dict) -> tuple[str, str]:
    """(seed IP, seed MAC) of an IoT-23 scenario: the inventory's infected host and the MAC of its units."""
    return _seed_ip_mac(scenario, inv)


def zero_variant_spec(spec: StreamSpec, variant: str, keep) -> StreamSpec:
    """Variant of an LP-IoT23 / training LP spec: same recipe (window, remap, payload thinning with the same seed)
    plus thinning of the seed host's pure zero-payload connections with keep probability ``keep`` ('q' = the
    stream's payload q; a number = constant, 0 = all removed) — streams/lp.thin_zero_payload. The stream id gets
    the suffix _<variant>; the integer stream id stays the base stream's (common random numbers)."""
    from dataclasses import replace
    if spec.kind not in ZERO_VARIANT_KINDS:
        raise ValueError(f"{spec.stream_id}: recipe variants are defined for {ZERO_VARIANT_KINDS} only")
    fg = spec.foregrounds[0]
    w = spec.extra.get("window", {})
    seed_mac = w.get("seed_mac")
    if not seed_mac or fg.mac_map.get(seed_mac) is None:
        raise ValueError(f"{spec.stream_id}: seed MAC unknown")
    ex = dict(spec.extra)
    ex["zero_thin"] = {"variant": variant, "keep": keep, "sources": {fg.source: [seed_mac]},
                       "seed_ip": w.get("seed_ip"), "base_stream_id": spec.stream_id, "stable_id_of": spec.stream_id}
    return replace(spec, stream_id=variant_stream_id(spec.stream_id, variant), extra=ex)


def variant_specs(cfg: dict, inv: dict, variant: str, kinds=None) -> list[StreamSpec]:
    """Specs of a recipe variant (config/streams.yaml zero_payload_variants.<variant>): LP-IoT23 test streams and
    training LP streams of every LOSO scenario at the configured prevalences."""
    v = zero_variant_cfg(cfg, variant)
    kinds = tuple(kinds or v.get("kinds", ZERO_VARIANT_KINDS))
    out = []
    folds = lp_folds(cfg, inv)
    for p in [float(x) for x in v["prevalences"]]:
        if "lp_iot23" in kinds:
            out += [zero_variant_spec(lp_iot23_spec(cfg, inv, s, p), variant, v["keep"]) for s in folds]
        if "train_lp" in kinds:
            out += [zero_variant_spec(train_lp_spec(cfg, inv, s, p), variant, v["keep"]) for s in folds]
    return out
