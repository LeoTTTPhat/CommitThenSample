#!/usr/bin/env python3
"""Resumable, multi-connection HTTP downloader for large public datasets.

    python tools/dl.py URL OUTFILE [--conns 8] [--chunk-mb 64] [--min-free-gb 150]

* Follows redirects (Hugging Face xet CDN, Kaggle signed GCS URLs, SharePoint).
* Splits the file into chunks fetched in parallel with HTTP Range requests and
  written in place (os.pwrite); progress is kept in OUTFILE.parts.json, so an
  interrupted download resumes where it stopped. Re-run the same command.
* Re-resolves the URL when a signed redirect expires (HTTP 403/410).
* Refuses to start if free disk would drop below --min-free-gb.
* Falls back to a single streaming GET when the server ignores Range.
Exit code 0 = complete, 3 = not enough disk, 1 = error.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

UA = {"User-Agent": "Mozilla/5.0 (research dataset download)"}


def resolve(sess: requests.Session, url: str) -> tuple[str, int | None, bool]:
    r = sess.get(url, headers={**UA, "Range": "bytes=0-0"}, allow_redirects=True, stream=True, timeout=60)
    final = r.url
    size, ranged = None, False
    if r.status_code == 206:
        cr = r.headers.get("Content-Range", "")
        if "/" in cr and cr.split("/")[-1].isdigit():
            size = int(cr.split("/")[-1])
            ranged = True
    elif r.status_code == 200:
        cl = r.headers.get("Content-Length")
        size = int(cl) if cl and cl.isdigit() else None
    else:
        r.raise_for_status()
    r.close()
    return final, size, ranged


def single_stream(sess, url, out):
    tmp = out + ".partial"
    with sess.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for blk in r.iter_content(1 << 20):
                f.write(blk)
    os.replace(tmp, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--conns", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--min-free-gb", type=float, default=150)
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and not os.path.exists(out + ".parts.json"):
        print(f"[dl] already complete: {out}")
        return 0
    sess = requests.Session()
    final, size, ranged = resolve(sess, a.url)
    free_gb = shutil.disk_usage(os.path.dirname(out)).free / 1e9
    need_gb = (size or 0) / 1e9
    if free_gb - need_gb < a.min_free_gb:
        print(f"[dl] refusing: free {free_gb:.0f} GB - need {need_gb:.1f} GB < {a.min_free_gb} GB", file=sys.stderr)
        return 3
    if not ranged or size is None:
        print(f"[dl] server does not support ranges; streaming {a.url}")
        single_stream(sess, final, out)
        return 0

    part = out + ".download"
    state_f = out + ".parts.json"
    chunk = a.chunk_mb << 20
    n = (size + chunk - 1) // chunk
    done: set[int] = set()
    if os.path.exists(state_f) and os.path.exists(part):
        st = json.load(open(state_f))
        if st.get("size") == size:
            done = set(st.get("done", []))
    else:
        with open(part, "wb") as f:
            f.truncate(size)
    json.dump({"url": a.url, "size": size, "done": sorted(done)}, open(state_f, "w"))
    lock = threading.Lock()
    url_box = [final]
    fd = os.open(part, os.O_WRONLY)
    t0, got = time.time(), [len(done) * chunk]

    def fetch(i: int) -> int:
        lo, hi = i * chunk, min(size, (i + 1) * chunk) - 1
        for attempt in range(8):
            try:
                with requests.get(url_box[0], headers={**UA, "Range": f"bytes={lo}-{hi}"}, stream=True, timeout=120) as r:
                    if r.status_code in (403, 410) and attempt < 7:
                        with lock:
                            url_box[0] = resolve(requests.Session(), a.url)[0]
                        continue
                    if r.status_code != 206:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    off = lo
                    for blk in r.iter_content(1 << 20):
                        os.pwrite(fd, blk, off)
                        off += len(blk)
                    if off != hi + 1:
                        raise RuntimeError(f"short chunk {i}: {off - lo} of {hi - lo + 1}")
                return i
            except Exception as e:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
                last = e
        raise RuntimeError(f"chunk {i} failed: {last}")

    todo = [i for i in range(n) if i not in done]
    print(f"[dl] {os.path.basename(out)}: {size / 1e9:.2f} GB, {len(todo)}/{n} chunks to fetch, {a.conns} conns")
    with ThreadPoolExecutor(a.conns) as ex:
        futs = [ex.submit(fetch, i) for i in todo]
        for k, fu in enumerate(as_completed(futs)):
            i = fu.result()
            with lock:
                done.add(i)
                got[0] += chunk
                if True:
                    json.dump({"url": a.url, "size": size, "done": sorted(done)}, open(state_f, "w"))
                    rate = (got[0] - 0) / max(1e-9, time.time() - t0) / 1e6
                    if k % 16 == 0 or len(done) == n:
                        print(f"[dl] {len(done)}/{n} chunks  ~{rate:.1f} MB/s", flush=True)
    os.close(fd)
    os.replace(part, out)
    os.remove(state_f)
    print(f"[dl] complete: {out} ({size / 1e9:.2f} GB) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
