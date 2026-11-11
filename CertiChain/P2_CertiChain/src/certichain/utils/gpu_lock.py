"""Project-wide GPU lock (DESIGN §8.1 item 4), shared by P1 and P2.

Every MLX/MPS job takes an exclusive ``fcntl.flock`` on ``memguard/run/gpu.lock``
before loading a model, and writes a JSON note {project, job, pid, started}.

    from certichain.utils.gpu_lock import gpu_lock
    with gpu_lock("P2", "p2_llm_qwen7b"):
        ...load model, score...
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parents[4] / "memguard" / "run" / "gpu.lock"


@contextlib.contextmanager
def gpu_lock(project: str, job: str, wait: bool = True, poll_s: float = 10.0):
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise
                time.sleep(poll_s)
        note = {"project": project, "job": job, "pid": os.getpid(),
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(note).encode())
        yield note
    finally:
        try:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def holder() -> dict | None:
    """Best-effort read of the current lock note (for status displays)."""
    try:
        txt = LOCK_PATH.read_text().strip()
        return json.loads(txt) if txt else None
    except (OSError, json.JSONDecodeError):
        return None
