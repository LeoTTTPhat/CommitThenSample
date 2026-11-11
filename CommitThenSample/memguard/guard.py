"""In-process helpers for experiments launched under memguard.

    from memguard.guard import scale, workers, check_memory

``scale()``/``workers()`` read the hints the supervisor exports (they are halved
after every memory-triggered restart, so a restarted job runs "slower and
smaller"). ``check_memory()`` lets long loops stop *before* the hard limit:
it raises ``MemoryBudgetExceeded`` once the process tree passes the soft limit
(90% of the budget). Scripts should catch it, flush checkpoints and
``sys.exit(75)`` -- the supervisor treats exit code 75 as a memory stop and
restarts the job with halved resources after a cooldown.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memguard import GB, tree_bytes  # noqa: E402

LIMIT_GB = float(os.environ.get("MEMGUARD_LIMIT_GB", "200"))
SOFT_FRACTION = 0.9
EXIT_MEMORY = 75


class MemoryBudgetExceeded(RuntimeError):
    pass


def scale() -> float:
    return float(os.environ.get("MEMGUARD_SCALE", "1.0"))


def workers(default: int = 16) -> int:
    return max(1, min(default, int(os.environ.get("MEMGUARD_WORKERS", str(default)))))


def attempt() -> int:
    return int(os.environ.get("MEMGUARD_ATTEMPT", "1"))


def usage_gb() -> float:
    return tree_bytes(os.getpid()) / GB


def check_memory(soft_fraction: float = SOFT_FRACTION) -> float:
    use = usage_gb()
    if use > soft_fraction * LIMIT_GB:
        raise MemoryBudgetExceeded(f"process tree uses {use:.1f} GB > {soft_fraction:.0%} of {LIMIT_GB:.0f} GB")
    return use
