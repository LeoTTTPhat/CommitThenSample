"""Project, data and results paths + atomic writers (DESIGN §7.4).

Layout (everything derived lives under ``P2_CertiChain/results/``; ``data/`` is read-only)::

    results/interim/<ds>/records/...      RECORD_SCHEMA parquet (no label columns, ever)
    results/interim/<ds>/labels/...       LABEL_SCHEMA parquet (quarantined; see io/labels.py)
    results/manifests/sha256_cache.json   cached sha256 of source files (io/hashing.py)

``CERTICHAIN_RESULTS`` (env) overrides the results root -- used by tests so they never
touch the real results tree.

All writers are atomic: write to ``<name>.tmp-<pid>`` in the same directory, fsync, then
``os.replace`` -- a crashed or memguard-killed job never leaves a half-written output that
a resumed run would mistake for a finished one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# src/certichain/io/paths.py -> parents: io, certichain, src, P2_CertiChain, <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[4]
P2_ROOT = PROJECT_ROOT / "P2_CertiChain"
CONFIG_DIR = P2_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"

# --- raw datasets (UNTRUSTED, read as data only) -------------------------------------
IOT23_SMALL = DATA_ROOT / "iot23" / "small"
IOT23_PCAP = DATA_ROOT / "iot23" / "pcap"
TON_ROOT = DATA_ROOT / "ton_iot"
TON_PROCESSED = TON_ROOT / "Processed_datasets" / "Processed_Network_dataset"
TON_BRO = TON_ROOT / "Raw_datasets" / "network_data" / "Network_dataset_Bro"
TON_MANIFEST = TON_ROOT / "MANIFEST.md"
GOTHAM_ROOT = DATA_ROOT / "gotham2025"
GOTHAM_PROCESSED = GOTHAM_ROOT / "processed"
GOTHAM_MIRAI_PCAPS = GOTHAM_ROOT / "raw" / "malicious" / "mirai-infection"
GOTHAM_METADATA = GOTHAM_ROOT / "github_labeller" / "metadata"
GOTHAM_MANIFEST = GOTHAM_ROOT / "MANIFEST.md"


def results_root() -> Path:
    env = os.environ.get("CERTICHAIN_RESULTS")
    return Path(env).resolve() if env else P2_ROOT / "results"


def interim(ds: str, *parts: str) -> Path:
    """``results/interim/<ds>/<parts...>``."""
    return results_root().joinpath("interim", ds, *parts)


def records_dir(ds: str) -> Path:
    return interim(ds, "records")


def labels_dir(ds: str) -> Path:
    return interim(ds, "labels")


def manifests_dir() -> Path:
    return results_root() / "manifests"


def rel(path: str | os.PathLike) -> str:
    """Path relative to the project root (posix), used for evidence pointers ``src_file``.

    Paths outside the project root (test fixtures in tmp dirs) are returned absolute.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_rel(relpath: str) -> Path:
    """Inverse of :func:`rel`."""
    p = Path(relpath)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- atomic writers
def _tmp(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{os.getpid()}")


def _fsync_replace(tmp: Path, path: Path) -> None:
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        f.write(data)
    _fsync_replace(tmp, path)
    return path


def atomic_write_text(path: str | os.PathLike, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def _json_default(o: Any):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int | None = 1) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=indent, default=_json_default, sort_keys=False) + "\n")


def atomic_write_parquet(df, path: str | os.PathLike, compression: str = "zstd", **kw) -> Path:
    """Write a polars DataFrame atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    df.write_parquet(tmp, compression=compression, **kw)
    _fsync_replace(tmp, path)
    return path


def atomic_sink_parquet(lf, path: str | os.PathLike, compression: str = "zstd", **kw) -> Path:
    """Stream a polars LazyFrame to parquet atomically (bounded memory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    lf.sink_parquet(tmp, compression=compression, **kw)
    _fsync_replace(tmp, path)
    return path


def log(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_rss_gb() -> float:
    """Peak RSS of the calling process (macOS ru_maxrss is in bytes, Linux in KiB)."""
    import resource
    import sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


def run_pool(fn, tasks: list, workers: int, poll_s: float = 5.0, on_result=None, label=lambda t: str(t)):
    """Run ``fn(task)`` for every task in a spawn process pool of <= ``workers`` processes.

    Memory protocol (DESIGN §8.1): the DRIVER loop calls ``check_memory()`` every
    ``poll_s``; pool workers call ``worker_check()`` themselves. On MemoryBudgetExceeded
    (driver or worker) or a killed worker (BrokenProcessPool) the pool is terminated and
    the process exits 75, so memguard restarts the (resumable) job with halved workers.
    Other task exceptions are collected and returned as ``errors`` (the job continues).
    """
    import multiprocessing as mp
    import sys
    import traceback
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    from certichain.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory

    results, errors = [], []
    if not tasks:
        return results, errors
    ex = ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("spawn"))
    futs = {ex.submit(fn, t): t for t in tasks}
    pending = set(futs)

    def _terminate():
        procs = list(getattr(ex, "_processes", {}).values())
        ex.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass

    try:
        while pending:
            done, pending = wait(pending, timeout=poll_s, return_when=FIRST_COMPLETED)
            check_memory()
            for f in done:
                t = futs[f]
                try:
                    r = f.result()
                except (MemoryBudgetExceeded, BrokenProcessPool):
                    raise
                except Exception as e:  # noqa: BLE001
                    tb = traceback.format_exc()
                    log(f"task {label(t)} FAILED: {e!r}\n{tb}")
                    errors.append({"task": label(t), "error": repr(e), "traceback": tb})
                else:
                    results.append(r)
                    if on_result is not None:
                        on_result(r)
    except (MemoryBudgetExceeded, BrokenProcessPool) as e:
        log(f"[mem] {type(e).__name__}: {e}; terminating pool and exiting {EXIT_MEMORY} (resumable)")
        _terminate()
        sys.exit(EXIT_MEMORY)
    ex.shutdown(wait=True)
    return results, errors


def clean_tmp(directory: str | os.PathLike) -> int:
    """Remove stale ``*.tmp-<pid>`` files left by a killed run (call at job start)."""
    n = 0
    d = Path(directory)
    if not d.exists():
        return 0
    for p in d.rglob("*.tmp-*"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n
