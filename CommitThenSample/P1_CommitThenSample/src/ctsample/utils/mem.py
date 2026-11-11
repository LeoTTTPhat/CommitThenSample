"""memguard bridge (DESIGN §7.3).

The memguard directory is put on sys.path directly (NOT the project root: memguard/
has no __init__.py and the project root would shadow memguard.py). Verified with
``.venv/bin/python -I``.

Launch heavy scripts from the project root as::

    python memguard/memguard.py run --name p1_<stage> --workers 24 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/<script>.py ...

``--workers 24`` is required (memguard exports MEMGUARD_WORKERS=16 by default) and
``-u`` is required (``-I`` ignores PYTHONUNBUFFERED).
"""
from __future__ import annotations

import sys
from pathlib import Path

_MG = str(Path(__file__).resolve().parents[4] / "memguard")
if _MG not in sys.path:
    sys.path.insert(0, _MG)

from guard import (  # noqa: E402
    EXIT_MEMORY,
    MemoryBudgetExceeded,
    attempt,
    check_memory,
    scale,
    usage_gb,
    workers,
)

__all__ = ["EXIT_MEMORY", "MemoryBudgetExceeded", "attempt", "check_memory", "scale",
           "usage_gb", "workers", "guarded"]


def guarded(loop_body, flush=lambda: None, default_workers: int = 24):
    """Run ``loop_body(scale=, workers=)``; on a soft memory stop flush and exit 75."""
    try:
        return loop_body(scale=scale(), workers=workers(default=default_workers))
    except MemoryBudgetExceeded as e:
        print(f"[mem] soft limit reached: {e}; flushing and exiting {EXIT_MEMORY}", flush=True)
        flush()
        sys.exit(EXIT_MEMORY)
