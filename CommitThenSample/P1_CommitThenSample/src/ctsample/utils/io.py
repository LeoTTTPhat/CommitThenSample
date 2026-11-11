"""Atomic, resumable I/O helpers (DESIGN §7.2).

Rules: every output is written to ``<path>.tmp`` and moved into place with
``os.replace``; a run directory is complete iff it contains ``DONE``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

P1_ROOT = Path(__file__).resolve().parents[3]          # .../P1_CommitThenSample
PROJECT_ROOT = P1_ROOT.parent                          # .../IoTJ-SI26-Oct
DATA = PROJECT_ROOT / "data"
RESULTS = P1_ROOT / "results"


def _tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def atomic_write_parquet(path: Path, table, compression: str = "zstd") -> None:
    """Write a pyarrow Table / pandas DataFrame / polars DataFrame atomically."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(table, pa.Table):
        if hasattr(table, "to_arrow"):          # polars
            table = table.to_arrow()
        else:                                   # pandas
            table = pa.Table.from_pandas(table, preserve_index=False)
    tmp = _tmp(path)
    pq.write_table(table, tmp, compression=compression)
    os.replace(tmp, path)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def exists_ok(path: Path, force: bool = False) -> bool:
    """True if ``path`` exists and we should skip recomputing it."""
    return (not force) and Path(path).exists()


def mark_done(run_dir: Path, info: dict | None = None) -> None:
    atomic_write_json(Path(run_dir) / "DONE", info or {})


def is_done(run_dir: Path) -> bool:
    return (Path(run_dir) / "DONE").exists()


def load_npy(path: Path, mmap: bool = True) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None)
