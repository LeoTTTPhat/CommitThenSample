"""Pipeline driver helpers (P2-F): per-incident data access with the pre-registration guard, an optional
candidate-view filter (G1 extra lever), cached label-free features, scores I/O, certification inputs, and
the LLM band driver (score the union of bands once per (incident, candidate) with ``MLXScorer.run``;
DESIGN §3.4.3, §5.3, §8.2).

Integrity guard (DESIGN §5.3, §6.4): ``load_labels`` (and every GT-side loader) refuses calibration/test
incidents (CF cal/test pools, every IoT-23 window, Gotham-real) until ``results/PREREG_2026-10-21.json``
exists. Training and dev labels are always available (training / gates / pilots). Labels are the
PAIR_LABEL_SCHEMA tables written by scripts/07 (GT side); record-level label tables are never touched.

Candidate view: ``set_candidate_filter(mode)`` drops extra self-continuation pairs AFTER P2-B's
generate_candidates (coordinator's G1 lever, 18 Oct): 'src_type' = same src and same unit type (any
svc_class); 'src_type_dstgap' = additionally same src, same dst set and lag_gap <= tau_merge. Default: off.
cand_id stays the scripts/07 id; arrays handed to certification are POSITIONAL (sorted by cand_id).

Layout (DESIGN §7.4):
  interim/<ds>/features[__<filter>]/<inc_id>.parquet   label-free base features (stats.base_features)
  scores/<ds>/<scorer_id>/<inc_id>.parquet             cand_id, s (+ s_stat, s_llm, llm_called for fusion)
  llm/<ds>/<group>/items.parquet + plan.json           frozen item list of a scoring group
  llm/<ds>/<group>/<model_tag>/<prompt_hash>/part-NNNNN.parquet   MLXScorer.run shards
"""
from __future__ import annotations

import glob
import inspect
import json
import os
import time
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import polars as pl

from certichain.io.paths import P2_ROOT, atomic_write_json, atomic_write_parquet, results_root

CF_VARIANTS = ("cf_ton", "cf_ton_h", "cf_gotham", "cf_gotham_t", "cf_gotham_v2", "cf_gotham_t_v2",
               "cf_gotham_xa", "cf_gotham_xb")   # xa / xb: P2-X Gotham cross-fit halves (exploratory, 19 Oct)
PREREG_NAME = "PREREG_2026-10-21.json"
TRAINING_ROLES = ("train", "dev")
FEATURE_CODE_FILES = ("features/stats.py", "features/priors.py")
SCORER_CODE_FILES = ("features/stats.py", "features/priors.py", "scoring/fusion.py", "scoring/cascade.py")
CAND_FILTERS = ("src_type", "src_type_dstgap")
TAU_MERGE = 600.0
BENCHMARK_SEED = 20261018


# ----------------------------------------------------------------------------- paths / views
_INTERIM_OVERRIDE: dict[str, Path] = {}
_VIEW: dict = {"filter": None}
_KEEP_CACHE: dict = {}


def set_interim_root(ds: str, path: str | Path | None) -> None:
    """Redirect interim/<ds> reads (G1 lever runs write P2-B's 07 outputs to a separate root)."""
    if path is None:
        _INTERIM_OVERRIDE.pop(ds, None)
    else:
        _INTERIM_OVERRIDE[ds] = Path(path)


def set_candidate_filter(mode: str | None) -> None:
    if mode is not None and mode not in CAND_FILTERS:
        raise ValueError(f"unknown candidate filter {mode!r}")
    _VIEW["filter"] = mode


def candidate_filter() -> str | None:
    return _VIEW["filter"]


def interim_dir(ds: str) -> Path:
    return _INTERIM_OVERRIDE.get(ds) or (results_root() / "interim" / ds)


def prereg_path() -> Path:
    return results_root() / PREREG_NAME


def prereg_exists() -> bool:
    return prereg_path().exists()


def role_of(ds: str, inc_id: str) -> str:
    """train | dev | caltest (CF variants); IoT-23 windows and Gotham-real incidents are 'test'."""
    if ds in CF_VARIANTS:
        parts = inc_id.split("/")
        return parts[1] if len(parts) >= 3 else "caltest"
    return "test"


def list_incidents(ds: str, role: str | None = None) -> list[str]:
    """Incident ids with finished scripts/07 outputs (stats/<inc>.json), optionally of one role."""
    base = interim_dir(ds) / "stats"
    files = sorted(glob.glob(str(base / "**" / "*.json"), recursive=True))
    ids = []
    for f in files:
        p = Path(f)
        if p.name.startswith("_"):
            continue
        ids.append(str(p.relative_to(base).with_suffix("")))
    if role is not None:
        ids = [i for i in ids if role_of(ds, i) == role]
    return sorted(ids)


def _p(ds: str, sub: str, inc: str, ext: str = "parquet") -> Path:
    return interim_dir(ds) / sub / f"{inc}.{ext}"


def load_units(ds: str, inc: str) -> pl.DataFrame:
    return pl.read_parquet(_p(ds, "units", inc))


def load_side(ds: str, inc: str) -> pl.DataFrame | None:
    p = _p(ds, "units_side", inc)
    return pl.read_parquet(p) if p.exists() else None


def load_stats(ds: str, inc: str) -> dict:
    return json.loads(_p(ds, "stats", inc, "json").read_text())


def load_scope(ds: str, inc: str) -> dict:
    return json.loads(_p(ds, "scope", inc, "json").read_text())


def continuation_keep(units: pl.DataFrame, cands: pl.DataFrame, mode: str, tau_merge: float = TAU_MERGE
                      ) -> np.ndarray:
    """Keep mask (cands order) of the extended self-continuation exclusion (label-free)."""
    if cands.height == 0:
        return np.zeros(0, dtype=bool)
    u = units.select("unit_id", "src", "type", pl.col("dsts").list.sort().alias("dset"))
    d = (cands.select("cand_id", "u", "v", "lag_gap")
         .join(u.rename({"unit_id": "u", "src": "src_u", "type": "type_u", "dset": "dset_u"}), on="u", how="left")
         .join(u.rename({"unit_id": "v", "src": "src_v", "type": "type_v", "dset": "dset_v"}), on="v", how="left"))
    d = cands.select("cand_id").join(d, on="cand_id", how="left", maintain_order="left")
    same_src = (d["src_u"] == d["src_v"]).fill_null(False)
    drop = same_src & (d["type_u"] == d["type_v"]).fill_null(False)
    if mode == "src_type_dstgap":
        drop = drop | (same_src & (d["dset_u"] == d["dset_v"]).fill_null(False) & (d["lag_gap"] <= tau_merge))
    elif mode != "src_type":
        raise ValueError(mode)
    return ~drop.to_numpy()


def _raw_cands(ds: str, inc: str) -> pl.DataFrame:
    return pl.read_parquet(_p(ds, "candidates", inc)).sort("cand_id")


def kept_cand_ids(ds: str, inc: str) -> np.ndarray | None:
    """cand_ids of the current view (None = all candidates, no filter)."""
    mode = _VIEW["filter"]
    if mode is None:
        return None
    key = (str(interim_dir(ds)), inc, mode)
    if key not in _KEEP_CACHE:
        c = _raw_cands(ds, inc)
        keep = continuation_keep(load_units(ds, inc), c, mode)
        _KEEP_CACHE[key] = c["cand_id"].to_numpy()[keep]
    return _KEEP_CACHE[key]


def load_cands(ds: str, inc: str) -> pl.DataFrame:
    c = _raw_cands(ds, inc)
    keep = kept_cand_ids(ds, inc)
    return c if keep is None else c.filter(pl.col("cand_id").is_in(keep.tolist()))


class PreregGuard(PermissionError):
    pass


def check_label_access(ds: str, inc: str) -> None:
    if role_of(ds, inc) not in TRAINING_ROLES and not prereg_exists():
        raise PreregGuard(f"labels of {ds}:{inc} (role {role_of(ds, inc)}) are sealed until {PREREG_NAME} exists")


def load_labels(ds: str, inc: str) -> pl.DataFrame:
    """PAIR_LABEL_SCHEMA (cand_id, y, bridge) of the current view, sorted by cand_id. GT side; guarded."""
    check_label_access(ds, inc)
    lab = pl.read_parquet(_p(ds, "labels_pairs", inc)).sort("cand_id")
    keep = kept_cand_ids(ds, inc)
    return lab if keep is None else lab.filter(pl.col("cand_id").is_in(keep.tolist()))


def load_transitions(ds: str, inc: str) -> pl.DataFrame:
    """Transition table; under a candidate filter, enabling pairs that were filtered out are not in C."""
    check_label_access(ds, inc)
    tr = pl.read_parquet(_p(ds, "transitions", inc))
    keep = kept_cand_ids(ds, inc)
    if keep is None or tr.height == 0:
        return tr
    ok = pl.col("cand_id").is_in(keep.tolist()).fill_null(False)
    return tr.with_columns(pl.when(ok).then(pl.col("cand_id")).otherwise(None).alias("cand_id"),
                           (pl.col("in_C") & ok).alias("in_C"))


def load_gt_units(ds: str, inc: str) -> pl.DataFrame:
    check_label_access(ds, inc)
    return pl.read_parquet(_p(ds, "gt_units", inc))


def load_gt_json(ds: str, inc: str) -> dict:
    check_label_access(ds, inc)
    return json.loads(_p(ds, "gt", inc, "json").read_text())


# ----------------------------------------------------------------------------- code hashes / features cache
def _hash_files(files: Sequence[str]) -> str:
    import blake3
    src = P2_ROOT / "src" / "certichain"
    h = blake3.blake3()
    for f in files:
        h.update(f.encode())
        h.update((src / f).read_bytes())
    return h.hexdigest()[:16]


def _vocab_hash() -> str:
    """Feature values depend on P2-B's type / service vocabularies (categorical codes): hash them too."""
    import blake3

    from certichain.features import stats as F
    return blake3.blake3(json.dumps([F.TYPE_VOCAB, F.SVC_VOCAB, F.IOT_PORTS, F.DIR_VOCAB, F.ENTITY_VOCAB]).encode()
                         ).hexdigest()[:8]


def feature_code_hash() -> str:
    return _hash_files(FEATURE_CODE_FILES) + _vocab_hash()


def scorer_code_hash() -> str:
    """BLAKE3 over the feature / fusion / cascade sources + the feature vocabularies (resume checks, PREREG)."""
    return _hash_files(SCORER_CODE_FILES) + _vocab_hash()


def features_path(ds: str, inc: str) -> Path:
    mode = _VIEW["filter"]
    return _p(ds, "features" if mode is None else f"features__{mode}", inc)


def load_features(ds: str, inc: str, *, compute: bool = True, write: bool = True) -> pl.DataFrame:
    """Label-free base features of one incident in the current view (cached; recomputed if the feature
    code or the scripts/07 candidate file changed)."""
    import pyarrow.parquet as pq

    from certichain.features.stats import base_features
    p = features_path(ds, inc)
    cp = _p(ds, "candidates", inc)
    tag = f"{feature_code_hash()}|{cp.stat().st_mtime_ns if cp.exists() else 0}"
    if p.exists():
        md = pq.read_metadata(p).metadata or {}
        if md.get(b"certichain.features", b"").decode() == tag:
            return pl.read_parquet(p)
    if not compute:
        raise FileNotFoundError(p)
    df = base_features(load_units(ds, inc), load_cands(ds, inc), load_side(ds, inc))
    if write:
        tbl = df.to_arrow().replace_schema_metadata({"certichain.features": tag})
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp-{os.getpid()}")
        pq.write_table(tbl, tmp, compression="zstd")
        os.replace(tmp, p)
    return df


# ----------------------------------------------------------------------------- scores / models / gates
def scores_path(ds: str, scorer_id: str, inc: str) -> Path:
    return results_root() / "scores" / ds / scorer_id / f"{inc}.parquet"


def write_scores(ds: str, scorer_id: str, inc: str, df: pl.DataFrame) -> Path:
    return atomic_write_parquet(df, scores_path(ds, scorer_id, inc))


def read_scores(ds: str, scorer_id: str, inc: str) -> pl.DataFrame:
    return pl.read_parquet(scores_path(ds, scorer_id, inc)).sort("cand_id")


def models_dir(scorer: str) -> Path:
    return results_root() / "models" / scorer


def atomic_save_npy(path, arr) -> Path:
    """np.save atomically (tmp file handle + fsync + os.replace; np.save would append '.npy' to a tmp name)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(arr))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def gates_dir() -> Path:
    d = results_root() / "gates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_gate(name: str, obj: dict) -> Path:
    return atomic_write_json(gates_dir() / f"{name}.json", obj)


def file_blake3(path) -> str | None:
    import blake3
    p = Path(path)
    if not p.exists():
        return None
    h = blake3.blake3()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- scorer registry
SCORERS: dict[str, dict] = {
    # scorer -> training incidents (dataset, role) and the datasets it scores (DESIGN §5.3)
    "S_TON": {"train": [("cf_ton", "train")], "targets": ["cf_ton", "gotham_real"]},
    "S_TON_H": {"train": [("cf_ton_h", "train")], "targets": ["cf_ton_h"]},
    "S_GOT": {"train": [("cf_gotham", "train")], "targets": ["cf_gotham", "cf_gotham_t"]},
    "S_CF": {"train": [("cf_ton", "train"), ("cf_gotham", "train")], "targets": ["iot23"]},
    # post-PREREG (19 Oct): CF-GOTHAM regenerated without the integer-second splice artifact (E15 check)
    "S_GOT2": {"train": [("cf_gotham_v2", "train")], "targets": ["cf_gotham_v2", "cf_gotham_t_v2"]},
    # post-PREREG corrected IoT-23 transfer scorer (S_CF stays the pre-registered primary)
    "S_CF2": {"train": [("cf_ton", "train"), ("cf_gotham_v2", "train")], "targets": ["iot23"]},
    # P2-X exploratory cross-fit (review R3-M4): scorer trained on train-pool half A only; bank from half B
    "S_GOTXA": {"train": [("cf_gotham_xa", "train")], "targets": ["cf_gotham_xb", "gotham_real"]},
}
DATASET_SCORER = {"cf_ton": "S_TON", "cf_ton_h": "S_TON_H", "cf_gotham": "S_GOT", "cf_gotham_t": "S_GOT",
                  "iot23": "S_CF", "gotham_real": "S_TON", "gotham": "S_TON", "cf_gotham_v2": "S_GOT2",
                  "cf_gotham_t_v2": "S_GOT2", "cf_gotham_xa": "S_GOTXA", "cf_gotham_xb": "S_GOTXA"}


def training_incidents(scorer: str) -> list[tuple[str, str]]:
    out = []
    for ds, role in SCORERS[scorer]["train"]:
        out += [(ds, i) for i in list_incidents(ds, role)]
    return out


# ----------------------------------------------------------------------------- LLM inputs (cards)
def unit_text_from_records(ds: str, inc: str, units: pl.DataFrame, max_per_field: int = 4) -> dict:
    """Per-unit packet text for cards (HTTP URI / UA; host / dns / sni come from UNIT_SCHEMA lists).

    CF variants: records live in interim/<ds>/incidents/<split>/<idx>/records.parquet; IoT-23 / Gotham:
    content fields are NULL in RECORD_SCHEMA (Suricata enrichment not run), so an empty map is returned.
    The most frequent distinct values per unit are kept (ties: lexicographic), at most max_per_field.
    """
    if ds not in CF_VARIANTS:
        return {}
    parts = inc.split("/")
    rp = interim_dir(ds) / "incidents" / Path(*parts[1:]) / "records.parquet"
    if not rp.exists():
        return {}
    cols = pl.read_parquet_schema(rp)
    want = [c for c in ("rec_id", "http_uri", "http_ua") if c in cols]
    if len(want) < 2:
        return {}
    rec = pl.read_parquet(rp, columns=want)
    m = units.select("unit_id", "rec_ids").explode("rec_ids").rename({"rec_ids": "rec_id"}).drop_nulls("rec_id")
    j = m.join(rec, on="rec_id", how="inner")
    out: dict[int, dict] = {}
    for col, key in (("http_uri", "uri"), ("http_ua", "ua")):
        if col not in j.columns:
            continue
        g = (j.filter(pl.col(col).is_not_null() & (pl.col(col) != "") & (pl.col(col) != "-"))
             .group_by(["unit_id", col]).agg(pl.len().alias("n"))
             .sort(["unit_id", "n", col], descending=[False, True, False])
             .group_by("unit_id", maintain_order=True).agg(pl.col(col).head(max_per_field)))
        for uid, vals in zip(g["unit_id"].to_list(), g[col].to_list()):
            out.setdefault(int(uid), {})[key] = list(vals)
    return out


def incident_t0(ds: str, inc: str, units: pl.DataFrame) -> float:
    """Incident clock t0: CF meta.json t0 (case window start), else the first unit start."""
    if ds in CF_VARIANTS:
        parts = inc.split("/")
        mp = interim_dir(ds) / "incidents" / Path(*parts[1:]) / "meta.json"
        if mp.exists():
            t0 = json.loads(mp.read_text()).get("t0")
            if t0 is not None:
                return float(t0)
    return float(units["ts"].min()) if units.height else 0.0


class CardBuilder:
    """Renders evidence cards for one incident (P2-D's features/cards.py), cached context."""

    def __init__(self, ds: str, inc: str, count_tokens=None, budget: int = 220):
        from certichain.features.cards import build_context
        self.ds, self.inc = ds, inc
        self.units = load_units(ds, inc)
        self.cands = _raw_cands(ds, inc)
        self.unit_text = unit_text_from_records(ds, inc, self.units)
        self.has_text = bool(self.unit_text)
        self.t0 = incident_t0(ds, inc, self.units)
        self.ctx = build_context(self.units, self.t0, self.unit_text)
        self._rows = {int(r["unit_id"]): r for r in self.units.iter_rows(named=True)}
        self._cand = {int(r["cand_id"]): r for r in self.cands.iter_rows(named=True)}
        self.count_tokens = count_tokens
        self.budget = budget

    def card(self, cand_id: int) -> str:
        from certichain.features.cards import render_card
        c = self._cand[int(cand_id)]
        return render_card(self._rows[int(c["u"])], self._rows[int(c["v"])], c, self.ctx,
                           count_tokens=self.count_tokens, budget=self.budget)


def card_input_source_hash() -> str:
    """BLAKE3 of the source of the functions here that decide card INPUTS (unit text, t0, CardBuilder).
    Recorded next to P2-D's prompt hash at the freeze (the prompt hash covers cards/roles/attack_kb)."""
    import blake3
    src = "".join(inspect.getsource(f) for f in (unit_text_from_records, incident_t0, CardBuilder))
    return blake3.blake3(src.encode()).hexdigest()[:16]


def iter_cards(ds: str, items: Sequence[tuple[str, int]], count_tokens=None, budget: int = 220,
               stats: dict | None = None) -> Iterator[tuple[tuple[str, int], str]]:
    """((incident, cand_id), card) in the given (deterministic) order; one CardBuilder per incident.
    ``stats`` (optional dict) collects incidents rendered without packet text (logged by the caller)."""
    cur, cb = None, None
    for inc, cid in items:
        if inc != cur:
            cb = CardBuilder(ds, inc, count_tokens, budget)
            cur = inc
            if stats is not None:
                stats["incidents"] = stats.get("incidents", 0) + 1
                if not cb.has_text:
                    stats["incidents_without_text"] = stats.get("incidents_without_text", 0) + 1
        yield (inc, int(cid)), cb.card(int(cid))


# ----------------------------------------------------------------------------- LLM groups
def llm_root(ds: str) -> Path:
    return results_root() / "llm" / ds


# Priority order (DESIGN §8.2): CF-TON (train band -> cal/test band) -> CF-TON-H -> B5 kappa = 1 arm ->
# SHOULD (IoT-23, CF-GOTHAM-T / Gotham-real, kappa = 0.2 extensions). (dataset, group, scorer, role, kind)
LLM_GROUPS: tuple[tuple[str, str, str | None, str, str], ...] = (
    ("cf_ton", "train_band", "S_TON", "train", "band"),
    ("cf_ton", "caltest_band", "S_TON", "caltest", "band"),
    ("cf_ton_h", "train_band", "S_TON_H", "train", "band"),
    ("cf_ton_h", "caltest_band", "S_TON_H", "caltest", "band"),
    ("cf_ton", "b5_arm", None, "caltest", "all"),
    ("iot23", "band", "S_CF", "test", "band"),
    ("cf_gotham", "train_band", "S_GOT", "train", "band"),
    ("cf_gotham", "caltest_band", "S_GOT", "caltest", "band"),
    ("cf_gotham_t", "caltest_band", "S_GOT", "caltest", "band"),
    ("gotham_real", "band", "S_TON", "test", "band"),
    ("cf_ton", "train_band_k020", "S_TON", "train", "band020"),
    ("cf_ton", "caltest_band_k020", "S_TON", "caltest", "band020"),
    ("cf_ton_h", "train_band_k020", "S_TON_H", "train", "band020"),
    ("cf_ton_h", "caltest_band_k020", "S_TON_H", "caltest", "band020"),
    ("cf_ton", "train_band_scf", "S_CF", "train", "band"),          # S_CF fusion (IoT-23, SHOULD): its own band
    ("cf_gotham", "train_band_scf", "S_CF", "train", "band"),
    # post-PREREG S_CF2 (CF-TON + cf_gotham_v2 train): its own bands, deduplicated per dataset as usual
    ("iot23", "band_scf2", "S_CF2", "test", "band"),
    ("cf_ton", "train_band_scf2", "S_CF2", "train", "band"),
    ("cf_gotham_v2", "train_band_scf2", "S_CF2", "train", "band"),
)
MUST_GROUPS = (("cf_ton", "train_band"), ("cf_ton", "caltest_band"), ("cf_ton_h", "train_band"),
               ("cf_ton_h", "caltest_band"), ("cf_ton", "b5_arm"))
B5_N_INCIDENTS = 15


def b5_incidents(ds: str = "cf_ton", n: int = B5_N_INCIDENTS) -> list[str]:
    """The kappa = 1 (B5) arm: n cal/test-pool incidents drawn label-free with the benchmark seed."""
    pool = list_incidents(ds, "caltest")
    if len(pool) <= n:
        return pool
    idx = np.random.default_rng(BENCHMARK_SEED).choice(len(pool), size=n, replace=False)
    return sorted(pool[int(i)] for i in idx)


def load_band(scorer: str, kind: str = "band"):
    from certichain.scoring.cascade import Band
    d = models_dir(scorer)
    if kind == "band":
        return Band.load(d / "band.json")
    if kind == "band020":
        grid = json.loads((d / "bands.json").read_text())["grid"]
        return Band.from_json(grid["0.20"])
    raise ValueError(kind)


def band_members(ds: str, scorer: str, incs: Sequence[str], band) -> list[tuple[str, int]]:
    """(incident, cand_id) of in-band candidates by the scorer's s_stat (training incidents: OOF s_stat)."""
    out = []
    for inc in incs:
        p = scores_path(ds, f"{scorer}.stat", inc)
        if not p.exists():
            raise FileNotFoundError(f"{p} (run scripts/08 --score for {scorer} first)")
        df = pl.read_parquet(p).sort("cand_id")
        m = band.contains(df["s"].to_numpy())
        out += [(inc, int(c)) for c in df["cand_id"].to_numpy()[m]]
    return out


def group_spec(ds: str, group: str) -> tuple[str | None, str, str]:
    for d, g, sc, role, kind in LLM_GROUPS:
        if d == ds and g == group:
            return sc, role, kind
    raise KeyError(f"unknown LLM group {ds}/{group}")


def _group_candidates(ds: str, group: str) -> list[tuple[str, int]]:
    sc, role, kind = group_spec(ds, group)
    if kind == "all":
        out = []
        for inc in b5_incidents(ds):
            out += [(inc, int(c)) for c in _raw_cands(ds, inc)["cand_id"].to_list()]
        return out
    incs = list_incidents(ds, role) if role != "test" else list_incidents(ds)
    return band_members(ds, sc, incs, load_band(sc, kind))


def plan_group(ds: str, group: str, *, write: bool = True) -> tuple[list[tuple[str, int]], dict]:
    """Frozen item list of a scoring group: its candidates minus every candidate of EARLIER groups of the
    same dataset (set-based, independent of scoring progress), sorted by (incident, cand_id). The list is
    written once to llm/<ds>/<group>/items.parquet and re-used (a different recomputation raises)."""
    gdir = llm_root(ds) / group
    ip = gdir / "items.parquet"
    fp = _plan_fingerprint(ds, group)
    if ip.exists():
        df = pl.read_parquet(ip)
        items = list(zip(df["incident"].to_list(), [int(c) for c in df["cand_id"].to_list()]))
        plan = json.loads((gdir / "plan.json").read_text())
        if plan.get("fingerprint") != fp:
            raise RuntimeError(f"stale LLM plan {ip}: scorer/band/candidates changed since it was frozen "
                               f"({plan.get('fingerprint')} != {fp}); move the group directory away to re-plan")
        return items, plan
    earlier: set[tuple[str, int]] = set()
    for d, g, *_ in LLM_GROUPS:
        if d == ds and g == group:
            break
        if d == ds:
            earlier |= set(_group_candidates(ds, g)) if _group_ready(ds, g) else set()
    cand = _group_candidates(ds, group)
    n_all = len(cand)
    items = sorted(set(cand) - earlier)
    sc, role, kind = group_spec(ds, group)
    n_ds = sum(_raw_cands(ds, i).height for i in sorted({i for i, _ in cand})) if cand else 0
    plan = {"dataset": ds, "group": group, "scorer": sc, "role": role, "kind": kind, "fingerprint": fp,
            "n_items": len(items),
            "n_group_candidates": n_all, "n_dedup_earlier": n_all - len(items),
            "n_incidents": len({i for i, _ in items}), "n_candidates_of_those_incidents": n_ds,
            "kappa_realised": (n_all / n_ds) if n_ds else None, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if write:
        atomic_write_parquet(pl.DataFrame({"incident": [i for i, _ in items], "cand_id": [c for _, c in items]},
                                          schema={"incident": pl.Utf8, "cand_id": pl.Int64}), ip)
        atomic_write_json(gdir / "plan.json", plan)
    return items, plan


def _plan_fingerprint(ds: str, group: str) -> str:
    """Scorer model + band files + the dataset's scripts/07 summary (code fingerprint) of a group's plan."""
    import blake3
    sc, role, kind = group_spec(ds, group)
    h = blake3.blake3()
    if sc is not None:
        for f in ("lgbm_stat.txt", "band.json", "bands.json"):
            h.update(str(file_blake3(models_dir(sc) / f)).encode())
    s7 = interim_dir(ds) / "summary_07.json"
    if s7.exists():
        h.update(json.dumps(json.loads(s7.read_text()).get("code_fp")).encode())
    h.update(f"{ds}|{group}|{role}|{kind}|{B5_N_INCIDENTS}|{BENCHMARK_SEED}".encode())
    return h.hexdigest()[:16]


def _group_ready(ds: str, group: str) -> bool:
    """A group can be planned once its scorer's stats scores exist (b5 arm: always)."""
    sc, role, kind = group_spec(ds, group)
    if kind == "all":
        return bool(list_incidents(ds, "caltest"))
    try:
        load_band(sc, kind)
    except (FileNotFoundError, KeyError):
        return False
    return True


def read_llm(ds: str, model_tag: str, prompt_hash: str, groups: Sequence[str] | None = None) -> pl.DataFrame:
    """All LLM shards of a dataset for one (model, prompt hash), over every scoring group (or the given
    ones), de-duplicated on (incident, cand_id)."""
    gl = "*" if groups is None else None
    pats = [str(llm_root(ds) / gl / model_tag / prompt_hash / "part-*.parquet")] if gl else \
        [str(llm_root(ds) / g / model_tag / prompt_hash / "part-*.parquet") for g in groups]
    files = sorted(f for p in pats for f in glob.glob(p))
    if not files:
        return pl.DataFrame(schema={"incident": pl.Utf8, "cand_id": pl.Int64, "s_llm": pl.Float32})
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    return df.unique(["incident", "cand_id"], keep="first", maintain_order=True)


def llm_lookup(ds: str, model_tag: str, prompt_hash: str) -> dict[str, dict[int, float]]:
    df = read_llm(ds, model_tag, prompt_hash)
    out: dict[str, dict[int, float]] = {}
    for inc, cid, s in zip(df["incident"].to_list(), df["cand_id"].to_list(), df["s_llm"].to_list()):
        out.setdefault(inc, {})[int(cid)] = float(s)
    return out


def run_groups(model_id: str, prefix: str, groups: Sequence[tuple[str, str]], *, frozen_hash: str | None = None,
               job: str = "p2_llm", batch_size: int | None = None, log=print, scorer=None, count_tokens=None,
               check_memory: bool = True) -> list[dict]:
    """Score the given (dataset, group)s in order with ONE model under the GPU lock (MLXScorer.run shards).
    ``scorer`` / ``count_tokens``: a pre-built MLXScorer and token counter (tests; no lock, no loading)."""
    import contextlib

    from certichain.llm.prompts import prompt_hash
    ph = prompt_hash(prefix)
    if frozen_hash is not None and ph != frozen_hash:
        raise RuntimeError(f"prompt hash {ph} != frozen {frozen_hash}: the prompt or card renderer changed")
    plans = [(ds, g, *plan_group(ds, g)) for ds, g in groups]
    if count_tokens is None:
        from certichain.features.cards import token_counter
        count_tokens = token_counter(model_id)
    counter = count_tokens
    out = []
    if scorer is None:
        from certichain.utils.gpu_lock import gpu_lock
        lock = gpu_lock("P2", job)
    else:
        lock = contextlib.nullcontext()
    with lock:
        if scorer is None:
            from certichain.llm.mlx_scorer import MLXScorer, ScorerConfig
            sc = MLXScorer(model_id, prefix, ScorerConfig(batch_size=batch_size))
        else:
            sc = scorer
        try:
            for ds, g, items, plan in plans:
                t0 = time.perf_counter()
                st: dict = {}
                if items:
                    sc.run(iter_cards(ds, items, counter, 220, st), llm_root(ds) / g, check_memory=check_memory)
                dt = time.perf_counter() - t0
                rec = {"dataset": ds, "group": g, "model_id": model_id, "model_revision": sc.revision,
                       "model_tag": sc.model_tag, "prompt_hash": sc.prompt_hash, "n_items": len(items),
                       "wall_s": round(dt, 1), "cand_per_s": (len(items) / dt) if dt > 0 and items else None,
                       "incidents_rendered": st.get("incidents", 0),
                       "incidents_without_packet_text": st.get("incidents_without_text", 0),
                       "n_truncated_total": sc.n_truncated, "plan": plan,
                       "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                atomic_write_json(llm_root(ds) / g / f"run_{sc.model_tag}_{sc.prompt_hash}.json", rec)
                log(f"[llm] {ds}/{g}: {len(items)} cands in {dt / 60:.1f} min"
                    + (f" ({len(items) / dt:.2f} cand/s)" if dt > 0 and items else ""))
                out.append(rec)
        finally:
            if scorer is None:
                sc.close()
    return out


# ----------------------------------------------------------------------------- certification inputs
def _gt_paths(transitions: Sequence[Sequence[str]], edge_pos: dict, max_paths: int = 2000) -> list[list[int]]:
    """Root -> leaf paths of the episode transition DAG as candidate-position sequences (-1 = uncoverable)."""
    succ: dict[str, list[str]] = {}
    indeg: dict[str, int] = {}
    nodes = set()
    for a, b in transitions:
        succ.setdefault(a, []).append(b)
        indeg[b] = indeg.get(b, 0) + 1
        nodes |= {a, b}
    roots = sorted(n for n in nodes if indeg.get(n, 0) == 0)
    out: list[list[int]] = []

    def walk(n, path, seen):
        if len(out) >= max_paths:
            return
        nxt = sorted(succ.get(n, []))
        if not nxt:
            if path:
                out.append(list(path))
            return
        for m in nxt:
            if m in seen:
                continue
            path.append(int(edge_pos.get((n, m), -1)))
            walk(m, path, seen | {m})
            path.pop()
    for r in roots:
        walk(r, [], {r})
    return out


def _unit_devices(units: pl.DataFrame) -> tuple[np.ndarray, dict]:
    """Device of a unit = its internal endpoint: src if the direction starts internal, else the first dst."""
    devs = []
    for src, dsts, dc in zip(units["src"].to_list(), units["dsts"].to_list(), units["dir_class"].to_list()):
        dc = dc or ""
        if dc.startswith("int") or not dsts:
            devs.append(src or "")
        else:
            devs.append(dsts[0])
    ids: dict[str, int] = {}
    arr = np.array([ids.setdefault(d, len(ids)) for d in devs], dtype=np.int64)
    return arr, ids


def build_incident(ds: str, inc: str, scores: np.ndarray, *, with_chains: bool = True,
                   extra_strata: dict | None = None):
    """certify.losses.Incident for one incident of the current view (GT side: needs label access).
    ``scores`` are aligned with load_cands(ds, inc) (sorted by cand_id); every array is positional."""
    from certichain.certify.losses import Incident
    cands = load_cands(ds, inc)
    lab = load_labels(ds, inc)
    if lab.height != cands.height or (lab.height and not (lab["cand_id"] == cands["cand_id"]).all()):
        raise RuntimeError(f"{ds}:{inc}: labels / candidates mismatch")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size != cands.height:
        raise RuntimeError(f"{ds}:{inc}: {scores.size} scores vs {cands.height} candidates")
    pos = {int(c): i for i, c in enumerate(cands["cand_id"].to_list())}
    y = lab["y"].to_numpy().astype(np.int8)
    tr = load_transitions(ds, inc)
    st = load_stats(ds, inc)["transition_stats"]
    gtj = load_gt_json(ds, inc)
    gtu = load_gt_units(ds, inc)
    units = load_units(ds, inc)
    N = cands.height
    trans = np.full(N, -1, dtype=np.int64)
    if tr.height:
        obs = tr.filter((pl.col("kind") == "obs") & pl.col("cand_id").is_not_null())
        for j, c in enumerate(obs["cand_id"].to_list()):
            p = pos.get(int(c))
            if p is not None and y[p] == 1:
                trans[p] = j
        edge_pos = {(a, b): pos[int(c)] for a, b, c, k in zip(tr["src_ep"].to_list(), tr["dst_ep"].to_list(),
                                                              tr["cand_id"].to_list(), tr["kind"].to_list())
                    if c is not None and k == "obs" and int(c) in pos}
    else:
        edge_pos = {}
    camps = (gtj.get("meta") or {}).get("campaigns") or {}
    decoy_camps = {c for c, d in camps.items() if (d or {}).get("kind", "main") != "main"}
    main_camps = {c for c, d in camps.items() if (d or {}).get("kind", "main") == "main"}
    ucamp = dict(zip(gtu["unit_id"].to_list(), gtu["campaign"].to_list())) if gtu.height else {}
    n_units = units.height
    decoy_unit = np.array([ucamp.get(i) in decoy_camps for i in range(n_units)], dtype=bool)
    dev, _ = _unit_devices(units)
    if camps:
        camp_units = [i for i in range(n_units) if ucamp.get(i) in main_camps]
    else:
        camp_units = [i for i in range(n_units) if ucamp.get(i) is not None]
    campaign_devices = sorted({int(dev[i]) for i in camp_units})
    paths = _gt_paths(gtj.get("transitions", []), edge_pos)
    L = max((len(p) for p in paths), default=0)
    strata = {"decoy_rich": len(decoy_camps) >= 2,
              "benign_only": (len(main_camps) == 0) if camps else (st.get("n_T", 0) == 0),
              "short_chain": L <= 2, "n_decoys": len(decoy_camps)}
    strata.update(extra_strata or {})
    kw = {}
    if with_chains:
        kw = {"u": cands["u"].to_numpy().astype(np.int64), "v": cands["v"].to_numpy().astype(np.int64),
              "unit_ts": units["ts"].to_numpy().astype(np.float64)}
    return Incident(incident_id=f"{ds}:{inc}", scores=scores, y=y, trans=trans,
                    bridge=lab["bridge"].fill_null(False).to_numpy().astype(bool), decoy_unit=decoy_unit,
                    unit_device=dev, campaign_devices=campaign_devices, gt_paths=paths,
                    n_trans=st.get("n_T"), n_trans_obs=st.get("n_T_obs"), n_trans_hid=st.get("n_T_hid"),
                    n_elink=st.get("n_E_link"), n_bridges=st.get("n_bridges"), strata=strata, **kw)


def view_transition_stats(ds: str, inc: str) -> dict:
    """n_T_obs, n_T_obs_in_C of the current view (r_CT under a candidate filter)."""
    tr = load_transitions(ds, inc)
    obs = tr.filter(pl.col("kind") == "obs") if tr.height else tr
    return {"n_T_obs": int(obs.height), "n_T_obs_in_C": int(obs["in_C"].sum()) if obs.height else 0}
