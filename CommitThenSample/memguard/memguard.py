#!/usr/bin/env python3
"""memguard: keep this project under a hard RAM budget (default 200 GB).

Two complementary mechanisms:

1. ``run``   -- launch an experiment under supervision. The supervisor polls the
                physical memory footprint of the whole process tree (macOS
                ``ri_phys_footprint``, which also counts Metal/MPS/MLX buffers;
                falls back to RSS). If the job alone, or the project as a whole,
                crosses the limit, the job is stopped (SIGTERM -> SIGKILL), the
                supervisor cools down until memory is released, and the job is
                restarted *slowly*: the scale/worker hints exported to the job are
                halved on every restart (MEMGUARD_SCALE, MEMGUARD_WORKERS) and the
                restart waits for a cooldown. Jobs are expected to be resumable
                (skip work whose outputs already exist).

2. ``watch`` -- a background watchdog that sums the footprint of *every* Python
                process belonging to this project (cwd or cmdline under the project
                root) and kills the largest one if the project total exceeds the
                limit. This is the backstop for ad-hoc scripts that were not
                launched through ``run``.

``status`` prints the current project memory usage.

Usage:
  python memguard/memguard.py run --name p1_main -- python P1/code/run_all.py
  python memguard/memguard.py watch &        # backstop daemon
  python memguard/memguard.py status

Events are appended to memguard/events.jsonl; per-run logs go to memguard/logs/.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent  # project root
GUARD_DIR = Path(__file__).resolve().parent
RUN_DIR = GUARD_DIR / "run"
LOG_DIR = GUARD_DIR / "logs"
EVENTS = GUARD_DIR / "events.jsonl"
GB = 1024 ** 3
DEFAULT_LIMIT_GB = float(os.environ.get("MEMGUARD_LIMIT_GB", "200"))


# --------------------------------------------------------------------------- #
# Physical footprint (macOS proc_pid_rusage) with RSS fallback
# --------------------------------------------------------------------------- #
class _RusageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


_libproc = None
if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RusageInfoV2)]
        _libproc.proc_pid_rusage.restype = ctypes.c_int
    except OSError:
        _libproc = None


def footprint_bytes(pid: int) -> int:
    """Physical footprint of one process (bytes); RSS if footprint is unavailable."""
    if _libproc is not None:
        info = _RusageInfoV2()
        if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(info)) == 0:
            return int(info.ri_phys_footprint)
    try:
        return psutil.Process(pid).memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def tree_bytes(pid: int) -> int:
    try:
        procs = [psutil.Process(pid)] + psutil.Process(pid).children(recursive=True)
    except (psutil.Error, OSError, SystemError):
        return 0
    return sum(footprint_bytes(p.pid) for p in procs)


def _is_project_python(p: psutil.Process) -> bool:
    try:
        cmd = p.cmdline()
        if not cmd or "python" not in os.path.basename(cmd[0]).lower():
            return False
        if "memguard.py" in " ".join(cmd):
            return False  # never count/kill the guard itself
        root = str(ROOT)
        if any(root in c for c in cmd):
            return True
        return str(p.cwd()).startswith(root)
    except (psutil.Error, OSError, SystemError):
        return False


def project_processes() -> list[tuple[psutil.Process, int]]:
    out = []
    for p in psutil.process_iter():
        if _is_project_python(p):
            out.append((p, footprint_bytes(p.pid)))
    return out


def log_event(kind: str, **kw) -> None:
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event": kind, **kw}
    with EVENTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[memguard] {rec}", flush=True)


def kill_tree(pid: int, grace: float = 15.0) -> None:
    try:
        parent = psutil.Process(pid)
        procs = parent.children(recursive=True) + [parent]
    except (psutil.Error, OSError, SystemError):
        return
    for p in procs:
        try:
            p.send_signal(signal.SIGTERM)
        except (psutil.Error, OSError, SystemError):
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace)
    for p in alive:
        try:
            p.kill()
        except (psutil.Error, OSError, SystemError):
            pass
    psutil.wait_procs(alive, timeout=5)


def wait_for_release(target_free_gb: float, max_wait: float) -> None:
    """Block until the project footprint has fallen and the system has headroom."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        proj = sum(b for _, b in project_processes()) / GB
        avail = psutil.virtual_memory().available / GB
        if proj < 5 and avail >= target_free_gb:
            return
        time.sleep(5)


# --------------------------------------------------------------------------- #
# Registry of supervised runs (lets concurrent supervisors share one budget)
# --------------------------------------------------------------------------- #
def registry_total(exclude: str | None = None) -> float:
    total, now = 0.0, time.time()
    for f in RUN_DIR.glob("*.json"):
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("name") == exclude or now - rec.get("ts", 0) > 15:
            continue
        if psutil.pid_exists(rec.get("pid", -1)):
            total += rec.get("usage_gb", 0.0)
    return total


def cmd_run(a: argparse.Namespace) -> int:
    RUN_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("memguard run: missing command", file=sys.stderr)
        return 2
    limit = a.limit_gb
    scale, workers = 1.0, a.workers
    reg = RUN_DIR / f"{a.name}.json"
    logf = (LOG_DIR / f"{a.name}.log").open("a")
    attempt = 0
    while True:
        attempt += 1
        env = dict(os.environ, MEMGUARD_ATTEMPT=str(attempt), MEMGUARD_SCALE=f"{scale:.4f}",
                   MEMGUARD_WORKERS=str(max(1, workers)), MEMGUARD_LIMIT_GB=str(limit),
                   PYTHONUNBUFFERED="1")
        log_event("start", name=a.name, attempt=attempt, scale=scale, workers=workers, cmd=" ".join(cmd))
        logf.write(f"\n===== attempt {attempt} scale={scale} workers={workers} {time.ctime()} =====\n")
        logf.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=a.cwd or os.getcwd(),
                                start_new_session=True)
        peak, killed = 0.0, False
        while proc.poll() is None:
            use = tree_bytes(proc.pid) / GB
            peak = max(peak, use)
            reg.write_text(json.dumps({"name": a.name, "pid": proc.pid, "usage_gb": use, "ts": time.time()}))
            others = registry_total(exclude=a.name)
            if use > limit or (use + others > limit and use >= others):
                log_event("limit_exceeded", name=a.name, attempt=attempt, usage_gb=round(use, 1),
                          others_gb=round(others, 1), limit_gb=limit)
                kill_tree(proc.pid)
                killed = True
                break
            time.sleep(a.interval)
        rc = proc.wait()
        reg.unlink(missing_ok=True)
        log_event("exit", name=a.name, attempt=attempt, rc=rc, killed=killed, peak_gb=round(peak, 1))
        memory_exit = killed or rc == 75 or (rc is not None and rc < 0 and -rc in (signal.SIGKILL, signal.SIGTERM))
        if rc == 0 and not killed:
            return 0
        if not memory_exit or attempt > a.max_restarts:
            return rc if rc else 1
        # stop -> cool down -> restart slowly with halved resources
        scale = max(scale / 2.0, a.min_scale)
        workers = max(1, workers // 2)
        log_event("cooldown", name=a.name, seconds=a.cooldown, next_scale=scale, next_workers=workers)
        time.sleep(a.cooldown)
        wait_for_release(target_free_gb=limit, max_wait=600)


def cmd_watch(a: argparse.Namespace) -> int:
    RUN_DIR.mkdir(exist_ok=True)
    (RUN_DIR / "watch.pid").write_text(str(os.getpid()))
    log_event("watch_start", limit_gb=a.limit_gb, pid=os.getpid())
    while True:
        try:  # the watchdog must never die (psutil on macOS can raise SystemError)
            procs = project_processes()
            total = sum(b for _, b in procs) / GB
            if total > a.limit_gb and procs:
                victim, vb = max(procs, key=lambda x: x[1])
                try:
                    cmdline = " ".join(victim.cmdline())[:200]
                except (psutil.Error, OSError, SystemError):
                    cmdline = "?"
                log_event("watch_kill", total_gb=round(total, 1), victim_pid=victim.pid,
                          victim_gb=round(vb / GB, 1), cmd=cmdline)
                kill_tree(victim.pid)
                time.sleep(a.cooldown)
        except Exception as e:  # noqa: BLE001
            try:
                log_event("watch_error", err=repr(e)[:300])
            except Exception:  # noqa: BLE001
                pass
        time.sleep(a.interval)


def cmd_status(_: argparse.Namespace) -> int:
    procs = project_processes()
    vm = psutil.virtual_memory()
    print(f"system: total={vm.total / GB:.0f} GB  available={vm.available / GB:.0f} GB  used%={vm.percent}")
    print(f"project python processes: {len(procs)}  footprint total={sum(b for _, b in procs) / GB:.1f} GB "
          f"(limit {DEFAULT_LIMIT_GB:.0f} GB)")
    for p, b in sorted(procs, key=lambda x: -x[1])[:15]:
        try:
            print(f"  pid={p.pid:<7} {b / GB:7.2f} GB  {' '.join(p.cmdline())[:110]}")
        except (psutil.Error, OSError, SystemError):
            pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="sub", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--limit-gb", type=float, default=DEFAULT_LIMIT_GB)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--max-restarts", type=int, default=4)
    r.add_argument("--cooldown", type=float, default=90.0)
    r.add_argument("--min-scale", type=float, default=0.0625)
    r.add_argument("--interval", type=float, default=2.0)
    r.add_argument("--cwd", default=None)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    w = sub.add_parser("watch")
    w.add_argument("--limit-gb", type=float, default=DEFAULT_LIMIT_GB)
    w.add_argument("--interval", type=float, default=3.0)
    w.add_argument("--cooldown", type=float, default=30.0)
    sub.add_parser("status")
    a = ap.parse_args()
    return {"run": cmd_run, "watch": cmd_watch, "status": cmd_status}[a.sub](a)


if __name__ == "__main__":
    sys.exit(main())
