"""P2-X revision task 6 (review R3-M4, M5): Gotham cross-fit of scorer and calibration bank. Owner: P2-X.
POST-PREREG, EXPLORATORY.

Problem: the in-domain Gotham comparison used S_GOT2 (trained on CF-GOTHAM v2 TRAIN incidents) with the CF-GOTHAM-T
bank, which is built from the SAME train-pool episodes -> scorer / bank overlap (A2 violated).
Cross-fit: split the Gotham TRAIN-pool devices into two disjoint halves by the episode's victim device,
  half A = {192.168.20.10}               (35 library episodes; every template type incl. CoAP)
  half B = {192.168.17.10, 192.168.18.15} (43 library episodes; mirai infection + mirai-dos + merlin types),
generate (with P2-E's unchanged generator code, in this process) CF-GOTHAM-XA (100 TRAIN incidents from half A, the
training set of the cross-fitted scorer S_GOTXA) and CF-GOTHAM-XB (150 bank incidents from half B, templates of the
case study, as CF-GOTHAM-T), then P2-B's 07 / P2-F's 08 and 11 build the scorer and the bank; the Gotham-real case
study is re-run with (S_GOTXA, CF-GOTHAM-XB bank). Remaining overlap (stated): the train-pool background capture
(01-14 18:40-19:43) is shared by both halves (40-63 min windows cannot be cut from one 63-min capture), and decoys
D3 come from the same look-alike catalogue.

The split library is NOT written into P2-E's library directory: a subclass of EpisodeLibrary / BackgroundLibrary maps
the new pools 'trainA' / 'trainB' onto filtered views of the 'train' pool files.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ..io.paths import atomic_write_json, interim, results_root

HALF_A = ("192.168.20.10",)
HALF_B = ("192.168.17.10", "192.168.18.15")
VAR_A, VAR_B = "cf_gotham_xa", "cf_gotham_xb"

_COMMON = {
    "dataset": "gotham", "ip_plan": "subnets",
    "role_subnets": {"SCANNER": "192.168.0.100", "LOADER": "192.168.33.12", "DLSRV": "192.168.33.13",
                     "CNC": "192.168.33.10", "REPSRV": "192.168.33.11", "C2": "192.168.34.10",
                     "USCANNER": "192.168.35.10", "AMPPEER": "192.168.0.200", "A": "192.168.0.100"},
    "n_camp_p": [0.10, 0.55, 0.25, 0.10], "decoy_rate": 2.0, "decoy_kind_p": {"D1": 0.4, "D2": 0.2, "D3": 0.4},
    "evidence_loss": 0.0, "victim_is_seed_p": 0.5,
    "window": {"min_s": 2400, "margin_s": 900, "extra_s": 600, "max_s": 3780}, "max_template_resamples": 5,
    "ts_round": None, "W": 1800, "scope_rule": "alg1_step6:seed+N1+shared_external;F_infra=3;U_max=1000",
    "time_shift": "campaign_phase",
}
CFG_A = {**copy.deepcopy(_COMMON), "variant": VAR_A, "seed_variant": VAR_A, "splits": {"train": 100},
         "pool_of_split": {"train": "trainA"}, "templates": {"train": ["T-MIRAI", "T-MDOS", "T-MERLIN", "T-AMP"]},
         "d1": {"types": {"train": ["mirai_bf", "masscan", "udp_scan"]}, "binds": {"server": "A", "victim": "X"}}}
CFG_B = {**copy.deepcopy(_COMMON), "variant": VAR_B, "seed_variant": VAR_B, "splits": {"caltest": 150},
         "pool_of_split": {"caltest": "trainB"}, "templates": {"caltest": ["T-MIRAI", "T-MDOS", "T-MERLIN"]},
         "d1": {"types": ["mirai_bf", "masscan"], "binds": {"server": "A", "victim": "X"}}}


def split_tables(src_ds: str = "gotham") -> tuple[pl.DataFrame, dict[str, set[str]]]:
    from ..chainforge.episodes import library_dir
    e = pl.read_parquet(library_dir(src_ds) / "episodes.parquet")
    tr = e.filter(pl.col("pool") == "train")
    a = set(tr.filter(pl.col("victim").is_in(list(HALF_A)))["lib_eid"].to_list())
    b = set(tr.filter(pl.col("victim").is_in(list(HALF_B)))["lib_eid"].to_list())
    if a & b or len(a | b) != tr.height:
        raise RuntimeError("train-pool episodes are not partitioned by the device halves")
    pool = (pl.when(pl.col("lib_eid").is_in(list(a))).then(pl.lit("trainA"))
            .when(pl.col("lib_eid").is_in(list(b))).then(pl.lit("trainB")).otherwise(pl.col("pool")))
    return e.with_columns(pool.alias("pool")), {"trainA": a, "trainB": b}


def make_libraries(src_ds: str = "gotham"):
    from ..chainforge.background import BackgroundLibrary
    from ..chainforge.episodes import EpisodeLibrary, library_dir

    table, members = split_tables(src_ds)

    @dataclass
    class SplitEpisodeLibrary(EpisodeLibrary):
        members: dict = field(default_factory=dict)

        def _store(self, pool: str):
            if pool not in self._rec:
                src = "train" if pool in self.members else pool
                d = library_dir(self.ds)
                rec = pl.read_parquet(d / f"records_{src}.parquet")
                lab = pl.read_parquet(d / f"labels_{src}.parquet")
                if pool in self.members:
                    keep = list(self.members[pool])
                    rec = rec.filter(pl.col("lib_eid").is_in(keep))
                    lab = lab.filter(pl.col("lib_eid").is_in(keep))
                self._rec[pool] = {k[0]: g for k, g in rec.partition_by("lib_eid", as_dict=True).items()}
                self._lab[pool] = {k[0]: g for k, g in lab.partition_by("lib_eid", as_dict=True).items()}
            return self._rec[pool], self._lab[pool]

    class SplitBackgroundLibrary(BackgroundLibrary):
        def lookalikes(self, pool: str) -> pl.DataFrame:
            if pool in ("trainA", "trainB"):
                if pool not in self._la:
                    la = super().lookalikes("train")
                    self._la[pool] = la.with_columns(pl.lit(pool).alias("pool")) if la.height else la
                return self._la[pool]
            return super().lookalikes(pool)

    lib = SplitEpisodeLibrary(ds=src_ds, table=table, members=members)
    bg = SplitBackgroundLibrary.load(src_ds)
    info = copy.deepcopy(bg.info)
    for p in ("trainA", "trainB"):
        info["pools"][p] = dict(info["pools"]["train"])
    bg.info = info
    return lib, bg, members


def generate(variant: str, cfg: dict, split: str, n: int, lib, bg, force: bool = False) -> dict:
    """P2-E's generator (chainforge.generator.generate_incident / write_incident, unchanged) for one split."""
    from ..chainforge import generator as G
    done = fail = 0
    errs = []
    t0 = time.perf_counter()
    for idx in range(n):
        d = G.incident_dir(variant, split, idx)
        if all((d / f).exists() for f in ("records.parquet", "labels.parquet", "meta.json", "gt.json", "gen.json")) \
                and not force:
            done += 1
            continue
        try:
            out = G.generate_incident(variant, split, idx, cfg, lib, bg)
            # source-level disjointness (as scripts/05 check_disjoint): every spliced episode from the split's pool
            spec = out["spec"]
            for c in spec.campaigns + spec.decoys:
                for e in c.episodes.values():
                    if e.lib_eid and lib.row(e.lib_eid)["pool"] != spec.pool:
                        raise AssertionError(f"episode {e.lib_eid} not from pool {spec.pool}")
            summ = G.incident_summary(out)
            out["gen"]["summary"] = summ
            G.write_incident(out, d)
            done += 1
        except Exception as ex:  # noqa: BLE001
            fail += 1
            errs.append(f"{variant}/{split}/{idx}: {type(ex).__name__}: {ex}")
    res = {"variant": variant, "split": split, "n": n, "done": done, "failed": fail, "errors": errs[:10],
           "seconds": time.perf_counter() - t0}
    atomic_write_json(interim(variant, f"generation_{split}.json"), {**res, "config": cfg})
    return res


def reuse_check(members: dict) -> dict:
    """Which library episodes the generated incidents use (A only in XA, B only in XB)."""
    out = {}
    for variant in (VAR_A, VAR_B):
        used = set()
        for p in sorted((interim(variant) / "incidents").glob("*/*/gen.json")):
            used |= set(json.loads(p.read_text()).get("summary", {}).get("lib_eids") or [])
        out[variant] = {"n_used": len(used), "in_A": len(used & members["trainA"]), "in_B": len(used & members["trainB"])}
    return out


def gen_chunk(task: dict) -> dict:
    """Process-pool worker: build the split libraries in this process and generate a chunk of incidents."""
    lib, bg, _ = make_libraries()
    cfg = CFG_A if task["variant"] == VAR_A else CFG_B
    from ..chainforge import generator as G
    done, errs = 0, []
    for idx in task["idx"]:
        d = G.incident_dir(task["variant"], task["split"], idx)
        if all((d / f).exists() for f in ("records.parquet", "labels.parquet", "meta.json", "gt.json", "gen.json")):
            done += 1
            continue
        try:
            out = G.generate_incident(task["variant"], task["split"], idx, cfg, lib, bg)
            spec = out["spec"]
            for c in spec.campaigns + spec.decoys:
                for e in c.episodes.values():
                    if e.lib_eid and lib.row(e.lib_eid)["pool"] != spec.pool:
                        raise AssertionError(f"episode {e.lib_eid} not from pool {spec.pool}")
            out["gen"]["summary"] = G.incident_summary(out)
            G.write_incident(out, d)
            done += 1
        except Exception as ex:  # noqa: BLE001
            errs.append(f"{task['variant']}/{task['split']}/{idx}: {type(ex).__name__}: {ex}")
    return {"task": {k: v for k, v in task.items() if k != "idx"}, "n": len(task["idx"]), "done": done, "errors": errs}
