"""memguard glue for P2 (DESIGN §8.1).

* Every P2 heavy job is launched from the project root with a 90 GB budget::

      python memguard/memguard.py run --name p2_<job> --limit-gb 90 --workers 16 -- \
          .venv/bin/python -u P2_CertiChain/scripts/<script>.py ...

  which exports MEMGUARD_LIMIT_GB=90, so ``check_memory()`` stops at 81 GB (soft).
* ``check_memory()`` measures the *caller's* process tree, so call it from the DRIVER
  loop only (never inside pool workers).
* Pool workers use ``worker_check(every)``: a per-worker soft budget of
  limit/workers() GB on their own footprint; a breach raises MemoryBudgetExceeded,
  the driver flushes shards and exits 75 (memguard restarts with halved resources).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_MG = str(Path(__file__).resolve().parents[4] / "memguard")
if _MG not in sys.path:
    sys.path.insert(0, _MG)

from guard import (  # noqa: E402
    EXIT_MEMORY,
    LIMIT_GB,
    MemoryBudgetExceeded,
    attempt,
    check_memory,
    scale,
    usage_gb,
    workers,
)
from memguard import GB, footprint_bytes  # noqa: E402

__all__ = ["EXIT_MEMORY", "LIMIT_GB", "MemoryBudgetExceeded", "attempt", "check_memory",
           "scale", "usage_gb", "workers", "worker_budget_gb", "worker_check", "guarded"]


def worker_budget_gb(default_workers: int = 16) -> float:
    return LIMIT_GB / max(1, workers(default=default_workers))


_calls = {"n": 0}


def worker_check(every: int = 64, default_workers: int = 16) -> None:
    """Cheap per-worker soft budget check (own footprint only), every ``every`` calls."""
    _calls["n"] += 1
    if _calls["n"] % every:
        return
    use = footprint_bytes(os.getpid()) / GB
    if use > 0.9 * worker_budget_gb(default_workers):
        raise MemoryBudgetExceeded(f"worker {os.getpid()} uses {use:.1f} GB")


def guarded(loop_body, flush=lambda: None, default_workers: int = 16):
    try:
        return loop_body(scale=scale(), workers=workers(default=default_workers))
    except MemoryBudgetExceeded as e:
        print(f"[mem] soft limit reached: {e}; flushing and exiting {EXIT_MEMORY}", flush=True)
        flush()
        sys.exit(EXIT_MEMORY)
