"""Natural single-capture streams (DESIGN §5.2, secondary setting, supplement only).

Each LOSO-eligible IoT-23 scenario alone, with its own benign units, at its natural prevalence (58-97 %
malicious units; mostly zero-payload scans), over the same 7.25-h window as its LP stream (10 min before the
first malicious payload connection, snapped to the source's epoch grid). No background, no remap, no thinning;
otherwise identical to streams/lp.py (StreamTables + features_pay.npy + cols_pay.npz + units.parquet).
Stream id: nat_iot23_<k>. Built by scripts/03b_build_streams.py (--kinds natural).
"""
from __future__ import annotations

from pathlib import Path

from .lp import STREAMS_DIR, build_stream
from .recipe import load_config, load_inventory, lp_folds, natural_spec

__all__ = ["natural_spec", "build_natural", "natural_ids"]


def natural_ids(cfg=None, inv=None) -> list[str]:
    cfg = cfg or load_config()
    inv = inv or load_inventory()
    return [natural_spec(cfg, inv, s).stream_id for s in lp_folds(cfg, inv)]


def build_natural(scenario: str, out_root: Path = STREAMS_DIR, force: bool = False, log=print) -> dict:
    cfg, inv = load_config(), load_inventory()
    return build_stream(natural_spec(cfg, inv, scenario), out_root=out_root, force=force, log=log)
