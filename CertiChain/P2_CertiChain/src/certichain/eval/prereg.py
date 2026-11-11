"""Pre-registration file (DESIGN §6.4): results/PREREG_2026-10-21.json, written BEFORE any
calibration/test run; its BLAKE3 is logged in plans.md.

write_prereg(path, cfg) writes canonical JSON (sorted keys, 2-space indent, UTF-8, trailing
newline) atomically and returns the BLAKE3 hex digest of the file bytes. A pre-registration is
immutable: rewriting an existing file with DIFFERENT content raises unless overwrite=True
(identical content is a no-op that returns the same hash).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import blake3

REQUIRED_KEYS = ("config_hashes", "data_manifest_sha256", "scorer_hashes", "prompt_hashes", "alpha_grid",
                 "lambda_grid", "K_per_dataset", "split_seeds", "endpoints", "holm_families", "fallback_framing")


def canonical_bytes(cfg: dict[str, Any]) -> bytes:
    return (json.dumps(cfg, sort_keys=True, indent=2, ensure_ascii=False, default=_default) + "\n").encode("utf-8")


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def blake3_file(path: Path | str) -> str:
    return blake3.blake3(Path(path).read_bytes()).hexdigest()


def write_prereg(path: Path | str, cfg: dict[str, Any], *, overwrite: bool = False, strict: bool = False) -> str:
    """Write the pre-registration and return its BLAKE3. ``strict`` requires REQUIRED_KEYS."""
    if strict:
        missing = [k for k in REQUIRED_KEYS if k not in cfg]
        if missing:
            raise ValueError(f"prereg missing keys: {missing}")
    path = Path(path)
    data = canonical_bytes(cfg)
    if path.exists():
        if path.read_bytes() == data:
            return blake3.blake3(data).hexdigest()
        if not overwrite:
            raise FileExistsError(f"{path} exists with different content (pre-registrations are immutable)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return blake3.blake3(data).hexdigest()


def verify_prereg(path: Path | str, expected_blake3: str) -> bool:
    return blake3_file(path) == expected_blake3
