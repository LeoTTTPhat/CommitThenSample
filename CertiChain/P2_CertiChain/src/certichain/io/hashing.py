"""Hashing helpers: cached SHA-256 of source files, BLAKE3 helpers, ordering ``tiehash``.

* ``sha256_file(path)`` -- SHA-256 of a (large) source file, cached in
  ``results/manifests/sha256_cache.json`` keyed by the project-relative path and validated
  by (size, mtime_ns). The cache is shared by concurrent workers (fcntl lock + atomic
  replace). Evidence pointers (``src_sha256``) always come from here.
* ``blake3_hex`` / ``blake3_file`` / ``tree_blake3`` -- for run manifests and claim records.
* ``tiehash(src, sport, dst, dport, proto)`` -- EXACTLY the rule in io/schema.py:
  first 8 bytes of BLAKE3(f"{src}|{sport}|{dst}|{dport}|{proto}") read as u64
  little-endian. Python f-string semantics: ints print in decimal, a missing port (None)
  prints as ``None``. ``tiehash_series`` is the vectorised version (one BLAKE3 call per
  distinct 5-tuple) and returns identical values.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import blake3

from certichain.io.paths import atomic_write_json, manifests_dir, rel, resolve_rel

_CHUNK = 8 << 20


# --------------------------------------------------------------------------- sha256 (cached)
def _cache_path() -> Path:
    return manifests_dir() / "sha256_cache.json"


def _sha256_raw(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_file(path: str | os.PathLike, use_cache: bool = True) -> str:
    """SHA-256 hex of ``path`` (cached by project-relative path, size and mtime_ns)."""
    p = Path(path).resolve()
    st = p.stat()
    key = rel(p)
    if not use_cache:
        return _sha256_raw(p)
    cache_file = _cache_path()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    lock = cache_file.with_suffix(".lock")
    # read under a shared lock
    with open(lock, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_SH)
        cache = _load(cache_file)
        fcntl.flock(lf, fcntl.LOCK_UN)
    ent = cache.get(key)
    if ent and ent.get("size") == st.st_size and ent.get("mtime_ns") == st.st_mtime_ns:
        return ent["sha256"]
    digest = _sha256_raw(p)
    record_sha256(p, digest)
    return digest


def record_sha256(path: str | os.PathLike, digest: str) -> None:
    """Store a digest computed elsewhere (e.g. during a structure scan) in the shared cache."""
    p = Path(path).resolve()
    st = p.stat()
    cache_file = _cache_path()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    lock = cache_file.with_suffix(".lock")
    with open(lock, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        cache = _load(cache_file)  # re-read: other workers may have added entries
        cache[rel(p)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        atomic_write_json(cache_file, dict(sorted(cache.items())), indent=1)
        fcntl.flock(lf, fcntl.LOCK_UN)


def cached_sha256(path: str | os.PathLike) -> str | None:
    """Cached digest if still valid (same size and mtime), else None (never hashes)."""
    p = Path(path).resolve()
    st = p.stat()
    ent = _load(_cache_path()).get(rel(p))
    if ent and ent.get("size") == st.st_size and ent.get("mtime_ns") == st.st_mtime_ns:
        return ent["sha256"]
    return None


def _load(cache_file: Path) -> dict:
    try:
        return json.loads(cache_file.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


_HEX64 = re.compile(r"\b[0-9a-f]{64}\b")


def manifest_sha256(manifest: str | os.PathLike, name: str) -> str | None:
    """sha256 listed for ``name`` (a file name or relative path) in a dataset MANIFEST.md.

    Returns the first 64-hex token on the first line that mentions ``name`` with a
    non-alphanumeric boundary after it (so ``..._1.csv`` does not match ``..._10.csv``).
    """
    pat = re.compile(re.escape(name) + r"(?![\w.-])")
    try:
        with open(manifest, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if pat.search(line):
                    m = _HEX64.search(line)
                    if m:
                        return m.group(0)
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- BLAKE3
def blake3_bytes(data: bytes | str) -> bytes:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return blake3.blake3(data).digest()


def blake3_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return blake3.blake3(data).hexdigest()


def blake3_file(path: str | os.PathLike) -> str:
    h = blake3.blake3()
    with open(path, "rb") as f:
        while True:
            b = f.read(_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tree_blake3(paths: Iterable[str | os.PathLike], suffixes: tuple[str, ...] = (".py", ".yaml", ".yml")) -> str:
    """BLAKE3 over (relative path, file BLAKE3) of every file under ``paths`` (sorted)."""
    items = []
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(q for q in root.rglob("*") if q.is_file())
        for q in files:
            if "__pycache__" in q.parts or (suffixes and q.suffix not in suffixes):
                continue
            items.append(f"{rel(q)}\t{blake3_file(q)}")
    return blake3_hex("\n".join(sorted(items)))


# --------------------------------------------------------------------------- tiehash
def tiehash(src, sport, dst, dport, proto) -> int:
    """u64 LE of the first 8 bytes of BLAKE3(f"{src}|{sport}|{dst}|{dport}|{proto}")."""
    s = f"{src}|{sport}|{dst}|{dport}|{proto}"
    return int.from_bytes(blake3.blake3(s.encode("utf-8")).digest()[:8], "little")


def tiehash_series(df, src="src", sport="sport", dst="dst", dport="dport", proto="proto"):
    """Vectorised tiehash for a polars DataFrame -> ``pl.Series`` (UInt64) named "tiehash".

    Null ports format as ``None`` (Python f-string semantics), identical to :func:`tiehash`.
    """
    import polars as pl

    def fmt(c):
        return pl.col(c).cast(pl.Utf8).fill_null("None")

    key = pl.concat_str([fmt(src), fmt(sport), fmt(dst), fmt(dport), fmt(proto)], separator="|")
    keys = df.select(key.alias("_k"))["_k"]
    uniq = keys.unique()
    vals = [int.from_bytes(blake3.blake3(k.encode("utf-8")).digest()[:8], "little") for k in uniq.to_list()]
    lut = pl.DataFrame({"_k": uniq, "tiehash": pl.Series(vals, dtype=pl.UInt64)})
    out = pl.DataFrame({"_k": keys}).join(lut, on="_k", how="left", maintain_order="left")
    return out["tiehash"]


def verify_sha256_against(path: str | os.PathLike, expected: str | None) -> bool | None:
    """True/False if an expected digest is given, None otherwise."""
    if not expected:
        return None
    return sha256_file(path) == expected


def config_hash(cfg) -> str:
    """BLAKE3 (16 hex) of the canonical JSON of a config object (resume checks)."""
    return blake3_hex(json.dumps(cfg, sort_keys=True, default=str))[:16]


def write_run_manifest(job: str, config, data_files: dict[str, str] | None = None,
                       extra: dict | None = None) -> Path:
    """``results/manifests/run_<timestamp>_<job>.json``: BLAKE3 of the P2 src/scripts/configs
    tree, the config, and the sha256 of the data files used (DESIGN §7.4)."""
    import time

    from certichain.io.paths import P2_ROOT

    tree = tree_blake3([P2_ROOT / "src" / "certichain", P2_ROOT / "scripts", P2_ROOT / "configs"])
    rec = {"job": job, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "src_tree_blake3": tree,
           "config": config, "config_hash": config_hash(config), "data_sha256": data_files or {}}
    if extra:
        rec.update(extra)
    path = manifests_dir() / f"run_{time.strftime('%Y%m%d-%H%M%S')}_{job}.json"
    atomic_write_json(path, rec)
    return path


__all__ = ["sha256_file", "record_sha256", "cached_sha256", "manifest_sha256", "blake3_bytes", "blake3_hex",
           "blake3_file", "tree_blake3", "tiehash", "tiehash_series", "verify_sha256_against", "resolve_rel",
           "config_hash", "write_run_manifest"]
