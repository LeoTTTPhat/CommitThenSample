"""Statistical candidate features (DESIGN §3.4.1) — label-free except the two training-only priors.

``base_features(units, cands, side)`` returns one row per candidate (cand_id order) with every
label-free feature; ``add_prior_features(df, priors)`` adds the two prior features (transition
prior, lag-LLR) and the lag-LLR rank among v's predecessors, from a ``PairPriors`` fitted on a
TRAINING pool (features/priors.py). ``feature_matrix(df, names)`` -> float32 matrix for LightGBM.

Feature groups (59 features; FEATURES lists them, PREFIX_FEATURES drops the 2 rarity features):
* lag (3): log1p(lag_start), signed log1p(lag_gap), overlap
* relations (6): the 5 relation one-hots + their count
* types (2): type_u, type_v (LightGBM categoricals over the fixed Appendix-C vocabulary)
* training-only priors (2): transition prior (type pair log-odds), lag-LLR
* competition, prefix-computable (5): number of candidate predecessors of v; rank of u among them by
  lag_gap and by lag-LLR; units of v's activity (src, svc, type) that started before v (log1p);
  units of u's activity started strictly between u and v (log1p, 0 = u is the latest of its activity)
* rarity, offline only (2): -log freq of (type_u, type_v) among the incident's candidates;
  -log freq of type_v among the incident's units
* entity (5): shared-entity kind (categorical), log1p degree, shared entity is the seed; u / v touch a seed
* intensity of u and of v (2 x 12): log1p bytes out / in, log1p packets, log1p #records, log1p #dsts
  (fan-out), failure ratio, log1p duration, start-periodicity CV, log1p #dports, service class
  (categorical), IoT-port class of the modal port (categorical), direction class (categorical)
* content of u and of v (2 x 5): exec-download, injection-regex, tool-UA, NXDOMAIN-heavy, high-entropy
  names (content_flags bits; DNS entropy / NXDOMAIN live in bits 8/16).
  SNI rarity is NOT included: it needs the raw tls_snis column, which is not FEATURE_SAFE (and the
  TON-derived CF variants carry no SNI at all).

EXCLUDED everywhere (anti-leakage / anti-splice, DESIGN §3.4.1): labels, absolute timestamps, raw
IPs/MACs, TTL/IP-ID/TCP window, dataset/episode/campaign/template ids, rec_id, normalised position
in the incident and time since t0 (E15 artifact probes only), seed_dist. Unit columns are read only
from io.schema.FEATURE_SAFE_UNIT_COLS plus the label-free side-table column ``touches_seed``;
raw IPs (src) are used only through equality (activity grouping), never as values.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from certichain.io.schema import (FEATURE_SAFE_UNIT_COLS, REL_PIVOT, REL_REVERSE, REL_SAME_ACTOR,
                                  REL_SAME_VICTIM, REL_SHARED_EXT)
from certichain.units.sessionize import SVC_CLASSES
from certichain.units.unit_types import ALL_TYPES

# ----------------------------------------------------------------------------- vocabularies
TYPE_VOCAB: tuple[str, ...] = tuple(ALL_TYPES)
SVC_VOCAB: tuple[str, ...] = tuple(SVC_CLASSES)
IOT_PORTS: tuple[int, ...] = (23, 2323, 22, 80, 8080, 443, 1883, 8883, 502, 5683, 554, 48101)
DIR_VOCAB: tuple[str, ...] = ("int->int", "int->ext", "ext->int", "ext->ext")
ENTITY_VOCAB: tuple[str, ...] = ("internal", "external", "infra", "none")
CONTENT_BITS = {"exec": 1, "inj": 2, "toolua": 4, "nxdom": 8, "hient": 16}
RELS = {"pivot": REL_PIVOT, "reverse": REL_REVERSE, "same_actor": REL_SAME_ACTOR,
        "same_victim": REL_SAME_VICTIM, "shared_ext": REL_SHARED_EXT}

UNIT_FEATS = ("lobytes", "lrbytes", "lpkts", "lnrec", "lndst", "fail", "ldur", "startcv", "lndports",
              "svc", "port", "dir")
UNIT_CAT = ("svc", "port", "dir")
CONTENT_FEATS = tuple(f"c_{k}" for k in CONTENT_BITS)

FEATURES: tuple[str, ...] = (
    "llag_start", "llag_gap", "overlap",
    *(f"rel_{k}" for k in RELS), "rel_count",
    "type_u", "type_v",
    "trans_prior", "lag_llr",
    "n_pred_v", "rank_lag_u", "rank_llr_u", "act_before_v", "act_between_u",
    "rar_pair", "rar_type_v",
    "ent_kind", "ent_ldeg", "ent_seed", "seed_u", "seed_v",
    *(f"{f}_u" for f in UNIT_FEATS), *(f"{f}_v" for f in UNIT_FEATS),
    *(f"{f}_u" for f in CONTENT_FEATS), *(f"{f}_v" for f in CONTENT_FEATS),
)
RARITY_FEATURES = ("rar_pair", "rar_type_v")
PRIOR_FEATURES = ("trans_prior", "lag_llr", "rank_llr_u")          # need fitted training priors
PREFIX_FEATURES: tuple[str, ...] = tuple(f for f in FEATURES if f not in RARITY_FEATURES)
CATEGORICAL: tuple[str, ...] = ("type_u", "type_v", "ent_kind", *(f"{c}_u" for c in UNIT_CAT),
                                *(f"{c}_v" for c in UNIT_CAT))
# columns of the per-candidate key frame kept next to the features (NOT features)
KEY_COLS = ("cand_id", "u", "v", "type_u_str", "type_v_str", "lag_gap")

_UNIT_READ = ("unit_id", "type", "svc_class", "dir_class", "dports", "n_dst", "n_dports", "n_records",
              "obytes", "rbytes", "opkts", "rpkts", "fail_frac", "dur", "start_cv", "content_flags")


def _code(values: Sequence, vocab: Sequence) -> list[int]:
    m = {v: i for i, v in enumerate(vocab)}
    unk = len(vocab)
    return [m.get(v, unk) for v in values]


def port_class(port: int | None) -> int:
    """IoT-port class: index in IOT_PORTS, else 12 = other well-known (< 1024), 13 = other high, 14 = none."""
    if port is None:
        return len(IOT_PORTS) + 2
    p = int(port)
    if p in IOT_PORTS:
        return IOT_PORTS.index(p)
    return len(IOT_PORTS) if p < 1024 else len(IOT_PORTS) + 1


# ----------------------------------------------------------------------------- per-unit table
def unit_features(units: pl.DataFrame, side: pl.DataFrame | None = None) -> pl.DataFrame:
    """Label-free per-unit features (unit_id + UNIT_FEATS + content bits + seed flag + activity key)."""
    assert set(_UNIT_READ) - {"unit_id"} <= set(FEATURE_SAFE_UNIT_COLS)
    u = units.select(*_UNIT_READ, "src", "ts")       # src / ts: equality / ordering only (never values)
    first_port = [(int(d[0]) if d is not None and len(d) else None) for d in u["dports"].to_list()]
    out = pl.DataFrame({
        "unit_id": u["unit_id"],
        "type_str": u["type"].fill_null("UNKNOWN"),
        "type": pl.Series(_code(u["type"].fill_null("UNKNOWN").to_list(), TYPE_VOCAB), dtype=pl.Int32),
        "lobytes": np.log1p(u["obytes"].fill_null(0).clip(0, None).to_numpy().astype(np.float64)),
        "lrbytes": np.log1p(u["rbytes"].fill_null(0).clip(0, None).to_numpy().astype(np.float64)),
        "lpkts": np.log1p((u["opkts"].fill_null(0) + u["rpkts"].fill_null(0)).clip(0, None).to_numpy()
                          .astype(np.float64)),
        "lnrec": np.log1p(u["n_records"].fill_null(1).to_numpy().astype(np.float64)),
        "lndst": np.log1p(u["n_dst"].fill_null(1).to_numpy().astype(np.float64)),
        "fail": u["fail_frac"].fill_null(0.0).to_numpy().astype(np.float64),
        "ldur": np.log1p(u["dur"].fill_null(0.0).clip(0, None).to_numpy().astype(np.float64)),
        "startcv": u["start_cv"].fill_nan(None).to_numpy().astype(np.float64),
        "lndports": np.log1p(u["n_dports"].fill_null(1).to_numpy().astype(np.float64)),
        "svc": pl.Series(_code(u["svc_class"].fill_null("other").to_list(), SVC_VOCAB), dtype=pl.Int32),
        "port": pl.Series([port_class(p) for p in first_port], dtype=pl.Int32),
        "dir": pl.Series(_code(u["dir_class"].fill_null("").to_list(), DIR_VOCAB), dtype=pl.Int32),
    })
    cf = u["content_flags"].fill_null(0).to_numpy().astype(np.int64)
    for k, b in CONTENT_BITS.items():
        out = out.with_columns(pl.Series(f"c_{k}", ((cf & b) != 0).astype(np.float64)))
    if side is not None and "touches_seed" in side.columns:
        ts_ = side.select("unit_id", pl.col("touches_seed").cast(pl.Float64).alias("seed"))
        out = out.join(ts_, on="unit_id", how="left").with_columns(pl.col("seed").fill_null(0.0))
    else:
        out = out.with_columns(pl.lit(np.nan).alias("seed"))
    # activity key (src, svc_class, type) -> integer id; order of starts within the activity
    act = u.select("unit_id", "ts", pl.concat_str([pl.col("src").fill_null(""), pl.col("svc_class").fill_null(""),
                                                    pl.col("type").fill_null("")], separator="|").alias("_k"))
    act = act.with_columns(pl.col("_k").rank("dense").cast(pl.Int64).alias("act_id"))
    out = out.join(act.select("unit_id", "act_id", "ts"), on="unit_id", how="left")
    return out


# ----------------------------------------------------------------------------- candidates
def base_features(units: pl.DataFrame, cands: pl.DataFrame, side: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per candidate (sorted by cand_id): KEY_COLS + every label-free feature of FEATURES
    (prior-dependent features are NaN until ``add_prior_features``)."""
    uf = unit_features(units, side)
    c = cands.sort("cand_id").select("cand_id", "u", "v", "rel_mask", "lag_start", "lag_gap", "overlap",
                                     "shared_entity_kind", "shared_entity_deg", "shared_is_seed")
    if c.height == 0:
        return pl.DataFrame(schema={**{k: pl.Float64 for k in FEATURES}, "cand_id": pl.Int64, "u": pl.Int64,
                                    "v": pl.Int64, "type_u_str": pl.Utf8, "type_v_str": pl.Utf8,
                                    "lag_gap": pl.Float64}).select(*KEY_COLS, *FEATURES)
    ucols = ["unit_id", "type_str", "type", *UNIT_FEATS, *CONTENT_FEATS, "seed", "act_id", "ts"]
    left = uf.select(ucols).rename({k: f"{k}_u" for k in ucols if k != "unit_id"}).rename({"unit_id": "u"})
    right = uf.select(ucols).rename({k: f"{k}_v" for k in ucols if k != "unit_id"}).rename({"unit_id": "v"})
    d = c.join(left, on="u", how="left").join(right, on="v", how="left").sort("cand_id")
    rm = d["rel_mask"].fill_null(0).to_numpy().astype(np.int64)
    lag_s = d["lag_start"].to_numpy().astype(np.float64)
    lag_g = d["lag_gap"].to_numpy().astype(np.float64)
    feats: dict[str, np.ndarray] = {
        "llag_start": np.log1p(np.clip(lag_s, 0, None)),
        "llag_gap": np.sign(lag_g) * np.log1p(np.abs(lag_g)),
        "overlap": d["overlap"].fill_null(False).to_numpy().astype(np.float64),
    }
    cnt = np.zeros(rm.size)
    for k, b in RELS.items():
        f = ((rm & b) != 0).astype(np.float64)
        feats[f"rel_{k}"] = f
        cnt += f
    feats["rel_count"] = cnt
    feats["type_u"] = d["type_u"].to_numpy().astype(np.float64)
    feats["type_v"] = d["type_v"].to_numpy().astype(np.float64)
    feats["trans_prior"] = np.full(rm.size, np.nan)
    feats["lag_llr"] = np.full(rm.size, np.nan)
    feats["rank_llr_u"] = np.full(rm.size, np.nan)
    # competition (prefix-computable): among the candidate predecessors of v
    comp = d.select("cand_id", "v", "lag_gap", "lag_start").with_columns(
        pl.len().over("v").alias("n_pred"),
        pl.col("lag_gap").rank("min").over("v").alias("rk"))
    feats["n_pred_v"] = comp["n_pred"].to_numpy().astype(np.float64)
    feats["rank_lag_u"] = comp["rk"].to_numpy().astype(np.float64)
    # activity novelty (prefix-computable; units only)
    ua = uf.select("unit_id", "act_id", "ts").sort(["ts", "unit_id"])
    act_ts: dict[int, np.ndarray] = {int(a): g["ts"].to_numpy() for (a,), g in ua.group_by(["act_id"])}
    au, av = d["act_id_u"].to_numpy(), d["act_id_v"].to_numpy()
    tu, tv = d["ts_u"].to_numpy(), d["ts_v"].to_numpy()
    before_v = np.empty(rm.size)
    between_u = np.empty(rm.size)
    for i in range(rm.size):
        a = act_ts[int(av[i])]
        before_v[i] = np.searchsorted(a, tv[i], side="left")             # same-activity units starting before v
        b = act_ts[int(au[i])]
        between_u[i] = np.searchsorted(b, tv[i], side="left") - np.searchsorted(b, tu[i], side="right")
    feats["act_before_v"] = np.log1p(before_v)
    feats["act_between_u"] = np.log1p(np.clip(between_u, 0, None))
    # rarity (offline only)
    pair = d.select(pl.concat_str([pl.col("type_str_u"), pl.col("type_str_v")], separator="|").alias("p"))["p"]
    pc = pair.value_counts()
    pmap = dict(zip(pc["p"].to_list(), pc["count"].to_list()))
    n = float(rm.size)
    feats["rar_pair"] = -np.log(np.array([pmap[p] for p in pair.to_list()], dtype=np.float64) / n)
    tcount = uf["type_str"].value_counts()
    tmap = dict(zip(tcount["type_str"].to_list(), tcount["count"].to_list()))
    nu = float(uf.height)
    feats["rar_type_v"] = -np.log(np.array([tmap.get(t, 1) for t in d["type_str_v"].to_list()],
                                           dtype=np.float64) / nu)
    # entity
    feats["ent_kind"] = np.asarray(_code(d["shared_entity_kind"].fill_null("none").to_list(), ENTITY_VOCAB),
                                   dtype=np.float64)
    feats["ent_ldeg"] = np.log1p(d["shared_entity_deg"].fill_null(0).clip(0, None).to_numpy().astype(np.float64))
    feats["ent_seed"] = d["shared_is_seed"].fill_null(False).to_numpy().astype(np.float64)
    feats["seed_u"] = d["seed_u"].to_numpy().astype(np.float64)
    feats["seed_v"] = d["seed_v"].to_numpy().astype(np.float64)
    for side_ in ("u", "v"):
        for f in (*UNIT_FEATS, *CONTENT_FEATS):
            feats[f"{f}_{side_}"] = d[f"{f}_{side_}"].to_numpy().astype(np.float64)
    keys = {"cand_id": d["cand_id"], "u": d["u"], "v": d["v"], "type_u_str": d["type_str_u"],
            "type_v_str": d["type_str_v"], "lag_gap": d["lag_gap"]}
    out = pl.DataFrame(keys).with_columns([pl.Series(k, feats[k]) for k in FEATURES])
    return out


def add_prior_features(df: pl.DataFrame, priors) -> pl.DataFrame:
    """Fill trans_prior, lag_llr and rank_llr_u from a PairPriors fitted on a training pool."""
    if df.height == 0:
        return df
    tu = df["type_u_str"].to_numpy()
    tv = df["type_v_str"].to_numpy()
    lg = df["lag_gap"].to_numpy().astype(np.float64)
    tp = priors.transition_prior(tu, tv)
    ll = priors.lag_llr(tu, tv, lg)
    return set_prior_columns(df, tp, ll)


def set_prior_columns(df: pl.DataFrame, trans_prior: np.ndarray, lag_llr: np.ndarray) -> pl.DataFrame:
    out = df.with_columns(pl.Series("trans_prior", np.asarray(trans_prior, dtype=np.float64)),
                          pl.Series("lag_llr", np.asarray(lag_llr, dtype=np.float64)))
    rk = out.select(pl.col("lag_llr").rank("min", descending=True).over("v").alias("r"))["r"]
    return out.with_columns(pl.Series("rank_llr_u", rk.to_numpy().astype(np.float64)))


def feature_matrix(df: pl.DataFrame, names: Sequence[str] = FEATURES) -> np.ndarray:
    return np.column_stack([df[n].to_numpy().astype(np.float32) for n in names]) if df.height else \
        np.zeros((0, len(names)), dtype=np.float32)


def categorical_indices(names: Sequence[str]) -> list[int]:
    return [i for i, n in enumerate(names) if n in CATEGORICAL]


def feature_set(name: str) -> tuple[str, ...]:
    if name in ("stat", "full"):
        return FEATURES
    if name == "prefix":
        return PREFIX_FEATURES
    raise ValueError(name)
