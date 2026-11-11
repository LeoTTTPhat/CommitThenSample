"""Per-incident loss tables (DESIGN §3.5 Alg. 4 steps 1-2, §3.6 Alg. 5, §6.2 counts).

Every rule family used here (BH on reference p-values over lambda, score thresholds over the
S_0-quantile grid, BH-iid, BY, Lee-Ren, B7, B8) selects a TOP-SCORE set of an incident's
candidates. A loss bank therefore stores, per incident, the candidates sorted by score
descending (CSR layout) with prefix counts, plus the selection size R of each family at each
grid index. Any count at any selection is one gather: count(R) = cum[off2[k] + R].

Tables [K_all, G] (G = |Lambda| = 61 for both families 'bh' and 'st'), as float64/int32 .npy:
  R, V (#Y != 1, strict false), A (#AMB), TP (#Y == 1), TR (transition hits, distinct),
  BR (bridge hits), DC (claims with a decoy endpoint), FD (non-campaign devices implicated by
  >= 1 false claim), fdp_strict, fdp_lenient, Lmono (running max of fdp_strict = CRC loss),
  fdx<gamma> (1{Lmono > gamma}; gamma in {0.2, 0.3}),
  chains (if endpoints given): chN (# maximal chains), chBad (# with a Y != 1 edge), chBad0
  (# with a Y = 0 edge), pfN / pfBad (prefix family size / # bad prefixes), ccr (# GT paths
  covered), chain_fdp, Lcmp (running max of max(fdp_strict, chain_fdp)), fwer (running max of
  1{chBad > 0}, the chain-RT loss).
Index parquet: one row per incident (row order = table row) with id, N, label counts and GT
denominators (|T*|, |T*_obs|, |T*_hid|, |E*_link|, #bridges, #GT paths, #non-campaign
devices), benign_only and free-form strata columns.

Labels: y in {1, 0, -1 (AMB)}. Strict FDP counts AMB as false (the calibration loss); lenient
excludes AMB from numerator and denominator (FDP_lenient <= FDP_strict, DESIGN §2.3).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import blake3
import numpy as np
import polars as pl

from .chains import J_DEFAULT, chain_stats, unit_positions, widest_chains
from .pvalues import ref_pvalues, sort_null_pool
from .rules import LAMBDA_GRID, bh_counts, bh_lam_star, score_order, st_counts, st_thresholds

FORMAT_VERSION = 1
GAMMAS = (0.2, 0.3)
FAMILIES = ("bh", "st")
COUNT_KEYS = ("V", "A", "TP", "TR", "BR", "DC", "FD")
CUM_OF = {"V": "c_false", "A": "c_amb", "TP": "c_tp", "TR": "c_trans", "BR": "c_bridge",
          "DC": "c_decoy", "FD": "c_fdev"}
CHAIN_KEYS = ("chN", "chBad", "chBad0", "pfN", "pfBad", "ccr")
META_INT = ("N", "n_true", "n_amb", "n_null0")
META_FLOAT = ("n_trans", "n_trans_obs", "n_trans_hid", "n_elink", "n_bridges", "n_gt_paths",
              "n_noncamp_dev")


# --------------------------------------------------------------------------- simple API (§7.2)
def fdp_curve(sel_sets: Sequence[np.ndarray], y, strict: bool = True) -> np.ndarray:
    """FDP of each selected index set. Strict: AMB counted false; lenient: AMB excluded."""
    y = np.asarray(y)
    out = np.empty(len(sel_sets))
    for i, s in enumerate(sel_sets):
        ys = y[np.asarray(s, dtype=np.int64)]
        if strict:
            out[i] = np.sum(ys != 1) / max(ys.size, 1)
        else:
            out[i] = np.sum(ys == 0) / max(int(np.sum(ys != -1)), 1)
    return out


def monotonize(curve: np.ndarray) -> np.ndarray:
    """L~(lambda) = max_{lambda' <= lambda} L(lambda') (CRC Cor. 1), along the last axis."""
    return np.maximum.accumulate(np.asarray(curve, dtype=np.float64), axis=-1)


# --------------------------------------------------------------------------- incident input
@dataclass
class Incident:
    """Per-incident arrays (candidate order is arbitrary; index = candidate id).

    scores [N] float; y [N] int in {1, 0, -1}; optional: trans [N] (transition id or -1; the
    cand -> transition map used for STR), bridge [N] bool, u/v [N] endpoint unit ids (u earlier),
    unit_ts [n_units] start times (else unit ids are the time order), decoy_unit [n_units] bool
    (UCR), decoy [N] bool (alternative per-candidate decoy flag), unit_device [n_units] int,
    campaign_devices (iterable of device ids), gt_paths (list of cand-id sequences, -1 = the
    transition's enabling pair is not in C), GT denominators and strata."""
    incident_id: str
    scores: np.ndarray
    y: np.ndarray
    trans: np.ndarray | None = None
    bridge: np.ndarray | None = None
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    unit_ts: np.ndarray | None = None
    decoy_unit: np.ndarray | None = None
    decoy: np.ndarray | None = None
    unit_device: np.ndarray | None = None
    campaign_devices: Sequence[int] | None = None
    gt_paths: list | None = None
    n_trans: int | None = None
    n_trans_obs: int | None = None
    n_trans_hid: int | None = None
    n_elink: int | None = None
    n_bridges: int | None = None
    strata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n = np.asarray(self.scores).size
        y = np.asarray(self.y)
        if y.shape != (n,):
            raise ValueError(f"{self.incident_id}: y shape {y.shape} != ({n},)")
        if not np.isin(y, (1, 0, -1)).all():
            raise ValueError(f"{self.incident_id}: labels must be in {{1, 0, -1}}")
        for name in ("trans", "bridge", "u", "v", "decoy"):
            a = getattr(self, name)
            if a is not None and np.asarray(a).shape != (n,):
                raise ValueError(f"{self.incident_id}: {name} must have shape ({n},)")
        if (self.u is None) != (self.v is None):
            raise ValueError(f"{self.incident_id}: give both u and v")
        if self.trans is not None:
            t = np.asarray(self.trans)
            if np.any((t >= 0) & (y != 1)):
                raise ValueError(f"{self.incident_id}: candidate mapped to a transition has y != 1")

    def hash_update(self, h) -> None:
        h.update(self.incident_id.encode())
        for name in ("scores", "y", "trans", "bridge", "u", "v", "unit_ts", "decoy_unit", "decoy",
                     "unit_device"):
            a = getattr(self, name)
            h.update(name.encode())
            if a is not None:
                a = np.ascontiguousarray(np.asarray(a))
                h.update(str(a.dtype).encode())
                h.update(a.tobytes())
        h.update(json.dumps([sorted(map(int, self.campaign_devices)) if self.campaign_devices is not None else None,
                             [list(map(int, p)) for p in self.gt_paths] if self.gt_paths is not None else None,
                             self.n_trans, self.n_trans_obs, self.n_trans_hid, self.n_elink, self.n_bridges,
                             {k: _jsonable(v) for k, v in sorted(self.strata.items())}],
                            default=str).encode())


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# --------------------------------------------------------------------------- per-incident work
def _first_occurrence_mask(keys_sorted_pos: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """mask[i] = True iff position i carries the first occurrence of its (valid) key."""
    n = keys_sorted_pos.size
    out = np.zeros(n, dtype=bool)
    pos = np.flatnonzero(valid)
    if pos.size:
        _, idx = np.unique(keys_sorted_pos[pos], return_index=True)
        out[pos[idx]] = True
    return out


def _cum(mask_or_counts: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(mask_or_counts, dtype=np.int64))).astype(np.int32)


def process_incident(inc: Incident, s0_sorted: np.ndarray, lam_grid: np.ndarray, st_grid: np.ndarray,
                     J: int = J_DEFAULT, chains: bool | str = "auto") -> dict[str, Any]:
    """All per-incident quantities for the bank (pure function; used by the parallel builder)."""
    inc.validate()
    scores = np.asarray(inc.scores, dtype=np.float64)
    y = np.asarray(inc.y, dtype=np.int8)
    N = scores.size
    order = score_order(scores)
    s_sorted = scores[order]
    p_sorted = ref_pvalues(s_sorted, s0_sorted)
    y_sorted = y[order]
    R_bh = bh_counts(bh_lam_star(p_sorted), lam_grid)
    R_st = st_counts(s_sorted, st_grid)

    false = y_sorted != 1
    amb = y_sorted == -1
    tp = y_sorted == 1
    out: dict[str, Any] = {
        "score": s_sorted, "p": p_sorted, "y": y_sorted, "cand": order.astype(np.int64),
        "c_false": _cum(false), "c_amb": _cum(amb), "c_tp": _cum(tp),
        "R_bh": R_bh.astype(np.int32), "R_st": R_st.astype(np.int32),
    }
    # transitions (distinct first hits)
    if inc.trans is not None:
        t_sorted = np.asarray(inc.trans, dtype=np.int64)[order]
        out["c_trans"] = _cum(_first_occurrence_mask(t_sorted, t_sorted >= 0))
        n_trans_inC = int(np.unique(t_sorted[t_sorted >= 0]).size)
    else:
        out["c_trans"] = np.zeros(N + 1, dtype=np.int32)
        n_trans_inC = None
    # bridges
    if inc.bridge is not None:
        b_sorted = np.asarray(inc.bridge, dtype=bool)[order] & tp
        out["c_bridge"] = _cum(b_sorted)
        n_bridges_inC = int(b_sorted.sum())
    else:
        out["c_bridge"] = np.zeros(N + 1, dtype=np.int32)
        n_bridges_inC = 0
    # decoy endpoints (UCR)
    if inc.decoy is not None:
        dflag = np.asarray(inc.decoy, dtype=bool)
    elif inc.decoy_unit is not None and inc.u is not None:
        du = np.asarray(inc.decoy_unit, dtype=bool)
        dflag = du[np.asarray(inc.u, dtype=np.int64)] | du[np.asarray(inc.v, dtype=np.int64)]
    else:
        dflag = np.zeros(N, dtype=bool)
    out["c_decoy"] = _cum(dflag[order])
    # device-level false implication
    n_noncamp = np.nan
    fd_counts = np.zeros(N, dtype=np.int64)
    if inc.unit_device is not None and inc.u is not None:
        dev = np.asarray(inc.unit_device, dtype=np.int64)
        camp = np.asarray(sorted(set(int(d) for d in (inc.campaign_devices or []))), dtype=np.int64)
        noncamp_unit = ~np.isin(dev, camp)
        n_noncamp = float(np.unique(dev[noncamp_unit]).size)
        us, vs = np.asarray(inc.u, dtype=np.int64)[order], np.asarray(inc.v, dtype=np.int64)[order]
        pos = np.concatenate([np.arange(N), np.arange(N)])
        devs = np.concatenate([dev[us], dev[vs]])
        ok = np.concatenate([false & noncamp_unit[us], false & noncamp_unit[vs]])
        pos, devs = pos[ok], devs[ok]
        if pos.size:
            srt = np.lexsort((pos, devs))
            first = np.ones(srt.size, dtype=bool)
            first[1:] = devs[srt][1:] != devs[srt][:-1]
            np.add.at(fd_counts, pos[srt][first], 1)
    out["c_fdev"] = _cum(fd_counts)

    # chains
    do_chains = (inc.u is not None) if chains == "auto" else bool(chains)
    G, Gs = lam_grid.size, st_grid.size
    if do_chains:
        if inc.u is None:
            raise ValueError(f"{inc.incident_id}: chains requested but no endpoints")
        u = np.asarray(inc.u, dtype=np.int64)
        v = np.asarray(inc.v, dtype=np.int64)
        n_units = int(len(inc.unit_ts)) if inc.unit_ts is not None else int(max(u.max(initial=-1), v.max(initial=-1)) + 1)
        upos = unit_positions(n_units, inc.unit_ts)
        us, vs = u[order], v[order]
        gt = inc.gt_paths or []
        stats: dict[int, tuple] = {0: (0, 0, 0, 0, 0, 0)}
        for R in np.unique(np.concatenate([R_bh, R_st])):
            R = int(R)
            if R in stats:
                continue
            ch = widest_chains(us[:R], vs[:R], s_sorted[:R], order[:R], upos, J)
            stats[R] = chain_stats(ch, y, gt)
        for fam, Rf in (("bh", R_bh), ("st", R_st)):
            arr = np.array([stats[int(r)] for r in Rf], dtype=np.int32).reshape(len(Rf), 6)
            for i, key in enumerate(CHAIN_KEYS):
                out[f"{key}_{fam}"] = arr[:, i]

    # meta
    n_true = int(tp.sum())
    def _f(x, fallback):
        return float(x) if x is not None else (float(fallback) if fallback is not None else np.nan)
    n_trans = _f(inc.n_trans, n_trans_inC)
    meta = {
        "incident_id": inc.incident_id, "N": N, "n_true": n_true, "n_amb": int(amb.sum()),
        "n_null0": int((y_sorted == 0).sum()),
        "n_trans": n_trans,
        "n_trans_obs": _f(inc.n_trans_obs, None if np.isnan(n_trans) else n_trans),
        "n_trans_hid": _f(inc.n_trans_hid, 0.0),
        "n_elink": _f(inc.n_elink, n_true),
        "n_bridges": _f(inc.n_bridges, n_bridges_inC),
        "n_gt_paths": float(len(inc.gt_paths)) if inc.gt_paths is not None else np.nan,
        "n_noncamp_dev": n_noncamp,
        "has_chains": bool(do_chains),
        "strata": {k: _jsonable(v) for k, v in inc.strata.items()},
    }
    meta["benign_only"] = bool(inc.strata.get("benign_only", n_true == 0 and not (n_trans > 0)))
    out["meta"] = meta
    return out


# --------------------------------------------------------------------------- the bank
@dataclass
class LossBank:
    """Cached loss tables + CSR prefix counts for one (dataset, scorer, rule) pool."""
    index: pl.DataFrame
    grids: dict[str, np.ndarray]
    tables: dict[str, np.ndarray]
    csr: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- basic accessors
    @property
    def n(self) -> int:
        return int(self.index.height)

    @property
    def has_chains(self) -> bool:
        return "chN_bh" in self.tables

    def t(self, name: str, family: str = "bh") -> np.ndarray:
        return self.tables[f"{name}_{family}"]

    def col(self, name: str) -> np.ndarray:
        """Index column as a float/bool numpy array (nulls -> NaN)."""
        s = self.index[name]
        if s.dtype == pl.Boolean:
            return s.to_numpy()
        return s.cast(pl.Float64).fill_null(np.nan).to_numpy()

    def gather(self, key: str, rows: np.ndarray, R: np.ndarray) -> np.ndarray:
        """cum-array lookup: value of prefix count ``key`` (e.g. 'c_tp') at selection size R."""
        rows = np.asarray(rows, dtype=np.int64)
        R = np.asarray(R, dtype=np.int64)
        return self.csr[key][self.csr["off2"][rows] + R]

    def counts(self, rows: np.ndarray, R: np.ndarray) -> dict[str, np.ndarray]:
        out = {"R": np.asarray(R, dtype=np.int64)}
        for k in COUNT_KEYS:
            out[k] = self.gather(CUM_OF[k], rows, R).astype(np.int64)
        return out

    def slice(self, row: int) -> slice:
        o = self.csr["offsets"]
        return slice(int(o[row]), int(o[row + 1]))

    def sizes(self) -> np.ndarray:
        o = self.csr["offsets"]
        return (o[1:] - o[:-1]).astype(np.int64)

    def cand_index(self, rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenated CSR positions of the candidates of ``rows`` (in order, duplicates allowed).
        Returns (idx, seg_start [len(rows)+1], within-incident rank 1..N)."""
        rows = np.asarray(rows, dtype=np.int64)
        o = self.csr["offsets"]
        lens = o[rows + 1] - o[rows]
        seg = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
        tot = int(seg[-1])
        rank = np.arange(tot, dtype=np.int64) - np.repeat(seg[:-1], lens)
        idx = np.repeat(o[rows], lens) + rank
        return idx, seg, rank + 1

    def scores(self, row: int) -> np.ndarray:
        return self.csr["score"][self.slice(row)]

    def labels(self, row: int) -> np.ndarray:
        return self.csr["y"][self.slice(row)]

    # ---- persistence
    def save(self, out_dir: Path | str, *, input_hash: str | None = None, overwrite: bool = False) -> Path:
        out_dir = Path(out_dir)
        tmp = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}-{time.time_ns()}")
        (tmp / "tables").mkdir(parents=True)
        (tmp / "csr").mkdir()
        files = {}
        for k, a in sorted(self.tables.items()):
            p = tmp / "tables" / f"{k}.npy"
            np.save(p, a)
            files[f"tables/{k}.npy"] = _file_b3(p)
        for k, a in sorted(self.csr.items()):
            p = tmp / "csr" / f"{k}.npy"
            np.save(p, a)
            files[f"csr/{k}.npy"] = _file_b3(p)
        for k, a in sorted(self.grids.items()):
            p = tmp / f"grid_{k}.npy"
            np.save(p, a)
            files[f"grid_{k}.npy"] = _file_b3(p)
        self.index.write_parquet(tmp / "index.parquet")
        files["index.parquet"] = _file_b3(tmp / "index.parquet")
        manifest = {"format_version": FORMAT_VERSION, "input_hash": input_hash, "n_incidents": self.n,
                    "grids": {k: int(v.size) for k, v in self.grids.items()},
                    "tables": {k: [list(a.shape), str(a.dtype)] for k, a in sorted(self.tables.items())},
                    "files_blake3": files, "meta": self.meta,
                    "created_unix": time.time()}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str))
        if out_dir.exists():
            if not overwrite:
                shutil.rmtree(tmp)
                raise FileExistsError(f"{out_dir} exists (pass overwrite=True)")
            old = out_dir.with_name(f".{out_dir.name}.old-{os.getpid()}-{time.time_ns()}")
            os.replace(out_dir, old)
            os.replace(tmp, out_dir)
            shutil.rmtree(old)
        else:
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, out_dir)
        return out_dir

    @classmethod
    def load(cls, path: Path | str, verify: bool = False) -> "LossBank":
        path = Path(path)
        man = json.loads((path / "manifest.json").read_text())
        if verify:
            for rel, h in man["files_blake3"].items():
                if _file_b3(path / rel) != h:
                    raise IOError(f"BLAKE3 mismatch for {rel}")
        tables = {p.stem: np.load(p) for p in sorted((path / "tables").glob("*.npy"))}
        csr = {p.stem: np.load(p) for p in sorted((path / "csr").glob("*.npy"))}
        grids = {p.stem[len("grid_"):]: np.load(p) for p in sorted(path.glob("grid_*.npy"))}
        index = pl.read_parquet(path / "index.parquet")
        meta = dict(man.get("meta", {}))
        meta["input_hash"] = man.get("input_hash")
        return cls(index=index, grids=grids, tables=tables, csr=csr, meta=meta)

    def subset(self, rows: Sequence[int]) -> "LossBank":
        """A new bank with the given incident rows (in that order; duplicates allowed)."""
        rows = np.asarray(rows, dtype=np.int64)
        tables = {k: a[rows] for k, a in self.tables.items()}
        o, o2 = self.csr["offsets"], self.csr["off2"]
        cand_parts = [np.arange(o[r], o[r + 1]) for r in rows]
        cum_parts = [np.arange(o2[r], o2[r + 1]) for r in rows]
        ci = np.concatenate(cand_parts) if cand_parts else np.empty(0, np.int64)
        cj = np.concatenate(cum_parts) if cum_parts else np.empty(0, np.int64)
        sizes = o[rows + 1] - o[rows]
        csr = {"offsets": np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)}
        csr["off2"] = csr["offsets"] + np.arange(rows.size + 1)
        for k, a in self.csr.items():
            if k in ("offsets", "off2"):
                continue
            csr[k] = a[cj] if k.startswith("c_") else a[ci]
        index = self.index[rows.tolist()].with_columns(pl.Series("row", np.arange(rows.size)))
        return LossBank(index=index, grids=dict(self.grids), tables=tables, csr=csr, meta=dict(self.meta))


def _file_b3(p: Path) -> str:
    h = blake3.blake3()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- assembly
def _derive_tables(R: np.ndarray, cnt: dict[str, np.ndarray], fam: str, gammas, tables: dict) -> None:
    Rf = R.astype(np.float64)
    V, A = cnt["V"].astype(np.float64), cnt["A"].astype(np.float64)
    tables[f"R_{fam}"] = R.astype(np.int32)
    for k in COUNT_KEYS:
        tables[f"{k}_{fam}"] = cnt[k].astype(np.int32)
    fs = V / np.maximum(Rf, 1.0)
    tables[f"fdp_strict_{fam}"] = fs
    tables[f"fdp_lenient_{fam}"] = (V - A) / np.maximum(Rf - A, 1.0)
    Lm = monotonize(fs)
    tables[f"Lmono_{fam}"] = Lm
    for g in gammas:
        tables[f"fdx{int(round(g * 100)):03d}_{fam}"] = (Lm > g + 1e-12).astype(np.float64)


def assemble_bank(results: Sequence[dict], lam_grid: np.ndarray, st_grid: np.ndarray,
                  gammas=GAMMAS, meta: dict | None = None) -> LossBank:
    K_all = len(results)
    sizes = np.array([r["score"].size for r in results], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    csr = {"offsets": offsets, "off2": offsets + np.arange(K_all + 1)}
    for k in ("score", "p", "y", "cand"):
        csr[k] = np.concatenate([r[k] for r in results]) if K_all else np.empty(0)
    for k in CUM_OF.values():
        csr[k] = np.concatenate([r[k] for r in results]).astype(np.int32) if K_all else np.empty(0, np.int32)
    bank_tables: dict[str, np.ndarray] = {}
    tmp = LossBank(index=pl.DataFrame({"row": np.arange(K_all)}), grids={}, tables={}, csr=csr)
    rows = np.arange(K_all)
    has_chains = K_all > 0 and all(r["meta"]["has_chains"] for r in results)
    for fam in FAMILIES:
        R = np.stack([r[f"R_{fam}"] for r in results]).astype(np.int64)
        cnt = tmp.counts(np.repeat(rows[:, None], R.shape[1], axis=1), R)
        _derive_tables(R, cnt, fam, gammas, bank_tables)
        if has_chains:
            for key in CHAIN_KEYS:
                bank_tables[f"{key}_{fam}"] = np.stack([r[f"{key}_{fam}"] for r in results]).astype(np.int32)
            cf = bank_tables[f"chBad_{fam}"] / np.maximum(bank_tables[f"chN_{fam}"], 1)
            bank_tables[f"chain_fdp_{fam}"] = cf
            bank_tables[f"Lcmp_{fam}"] = monotonize(np.maximum(bank_tables[f"fdp_strict_{fam}"], cf))
            bank_tables[f"fwer_{fam}"] = monotonize((bank_tables[f"chBad_{fam}"] > 0).astype(np.float64))
    # index
    metas = [r["meta"] for r in results]
    cols: dict[str, Any] = {"row": np.arange(K_all, dtype=np.int64),
                            "incident_id": [m["incident_id"] for m in metas]}
    for k in META_INT:
        cols[k] = np.array([m[k] for m in metas], dtype=np.int64)
    for k in META_FLOAT:
        cols[k] = np.array([m[k] for m in metas], dtype=np.float64)
    cols["benign_only"] = np.array([m["benign_only"] for m in metas], dtype=bool)
    strata_keys = sorted({k for m in metas for k in m["strata"]} - set(cols))
    index = pl.DataFrame(cols)
    for k in strata_keys:
        vals = [m["strata"].get(k) for m in metas]
        index = index.with_columns(pl.Series(k, vals, strict=False))
    index = index.with_columns([pl.col(k).fill_nan(None) for k in META_FLOAT])
    grids = {"bh": np.asarray(lam_grid, dtype=np.float64), "st": np.asarray(st_grid, dtype=np.float64)}
    return LossBank(index=index, grids=grids, tables=bank_tables, csr=csr, meta=dict(meta or {}))


# --------------------------------------------------------------------------- builder
_W: dict[str, Any] = {}


def _winit(s0, lam, st, J, chains):
    _W.update(s0=s0, lam=lam, st=st, J=J, chains=chains)


def _wrun(inc: Incident):
    return process_incident(inc, _W["s0"], _W["lam"], _W["st"], _W["J"], _W["chains"])


def bank_input_hash(incidents: Sequence[Incident], s0_sorted: np.ndarray, lam_grid, gammas, J, chains) -> str:
    h = blake3.blake3()
    h.update(json.dumps({"v": FORMAT_VERSION, "lam": np.asarray(lam_grid).tolist(), "gammas": list(gammas),
                         "J": J, "chains": str(chains)}).encode())
    h.update(np.ascontiguousarray(s0_sorted).tobytes())
    for inc in incidents:
        inc.hash_update(h)
    return h.hexdigest()


def build_loss_tables(incidents: Iterable[Incident], s0, out_dir: Path | str | None = None, *,
                      lam_grid=LAMBDA_GRID, gammas=GAMMAS, J: int = J_DEFAULT, chains: bool | str = "auto",
                      n_jobs: int = 1, overwrite: bool = False, meta: dict | None = None) -> LossBank:
    """Build (or load, if ``out_dir`` already holds a bank with the same input hash) a LossBank.

    ``s0``: reference null scores (training, cross-fitted). Writes ``out_dir`` atomically.
    ``n_jobs`` > 1 uses a process pool (<= 8 processes per the project rules).
    """
    incs = list(incidents)
    s0_sorted = sort_null_pool(s0)
    lam_grid = np.asarray(lam_grid, dtype=np.float64)
    st_grid = st_thresholds(s0_sorted, lam_grid)
    h = bank_input_hash(incs, s0_sorted, lam_grid, gammas, J, chains)
    if out_dir is not None:
        out_dir = Path(out_dir)
        man = out_dir / "manifest.json"
        if man.exists():
            old = json.loads(man.read_text()).get("input_hash")
            if old == h:
                return LossBank.load(out_dir)
            if not overwrite:
                raise FileExistsError(f"{out_dir} holds a bank with a different input hash; pass overwrite=True")
    n_jobs = max(1, min(int(n_jobs), 8))
    if n_jobs == 1 or len(incs) < 2 * n_jobs:
        results = [process_incident(inc, s0_sorted, lam_grid, st_grid, J, chains) for inc in incs]
    else:
        with ProcessPoolExecutor(n_jobs, initializer=_winit,
                                 initargs=(s0_sorted, lam_grid, st_grid, J, chains)) as ex:
            results = list(ex.map(_wrun, incs, chunksize=max(1, len(incs) // (8 * n_jobs))))
    m = {"n_s0": int(s0_sorted.size), "J": J, "gammas": list(gammas)}
    m.update(meta or {})
    bank = assemble_bank(results, lam_grid, st_grid, gammas, m)
    bank.meta["input_hash"] = h
    if out_dir is not None:
        bank.save(out_dir, input_hash=h, overwrite=True)
    return bank
