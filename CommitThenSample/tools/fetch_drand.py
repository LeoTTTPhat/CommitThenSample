#!/usr/bin/env python3
"""Fetch a block of consecutive public drand rounds (League of Entropy default/mainnet chain).

    python tools/fetch_drand.py OUTDIR --start 2570006 --count 3000 [--workers 8]

Writes OUTDIR/chain_info.json (from /info), OUTDIR/chains.json, OUTDIR/rounds.jsonl (sorted by round:
round, randomness, signature, previous_signature, plus derived UTC time) and prints verification:
randomness == sha256(signature) and previous_signature(r) == signature(r-1) (chained scheme).
Endpoints are tried in order: api.drand.sh, api2.drand.sh, drand.cloudflare.com.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

HOSTS = ["https://api.drand.sh", "https://api2.drand.sh", "https://drand.cloudflare.com"]
UA = {"User-Agent": "research-beacon-fetch/1.0 (IoTJ-SI26 project; polite, <=8 connections)"}
_local = threading.local()


def sess() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(UA)
    return _local.s


def get_json(path: str, tries: int = 6, delay: float = 0.1) -> tuple[dict, str]:
    err = None
    for i in range(tries):
        host = HOSTS[min(i // 2, len(HOSTS) - 1)]
        try:
            r = sess().get(host + path, timeout=30)
            if r.status_code == 200:
                time.sleep(delay)  # politeness
                return r.json(), host
            err = f"HTTP {r.status_code}"
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
        except (requests.RequestException, ValueError) as e:
            err = repr(e)
        time.sleep(min(30, 1.5 * 2 ** i))
    raise RuntimeError(f"{path}: {err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    workers = min(8, a.workers)
    os.makedirs(a.outdir, exist_ok=True)

    info, host = get_json("/info")
    chains, _ = get_json("/chains")
    json.dump(info, open(os.path.join(a.outdir, "chain_info.json"), "w"), indent=1)
    json.dump(chains, open(os.path.join(a.outdir, "chains.json"), "w"), indent=1)
    print(f"[drand] info from {host}: hash={info['hash']} period={info['period']} genesis={info['genesis_time']}")
    chash = info["hash"]

    rounds = list(range(a.start, a.start + a.count))
    out: dict[int, dict] = {}
    hosts_used: dict[str, int] = {}
    t0 = time.time()

    def fetch(r: int):
        d, h = get_json(f"/{chash}/public/{r}")
        return r, d, h

    with ThreadPoolExecutor(workers) as ex:
        for r, d, h in ex.map(fetch, rounds):
            out[r] = d
            hosts_used[h] = hosts_used.get(h, 0) + 1
    print(f"[drand] fetched {len(out)} rounds in {time.time() - t0:.1f} s via {hosts_used}")

    period, genesis = info["period"], info["genesis_time"]
    bad_rand = bad_link = 0
    with open(os.path.join(a.outdir, "rounds.jsonl"), "w") as f:
        for r in rounds:
            d = out[r]
            assert d["round"] == r, (d["round"], r)
            ts = genesis + (r - 1) * period
            if hashlib.sha256(bytes.fromhex(d["signature"])).hexdigest() != d["randomness"]:
                bad_rand += 1
            if r - 1 in out and out[r - 1]["signature"] != d.get("previous_signature"):
                bad_link += 1
            f.write(json.dumps({"round": r, "randomness": d["randomness"], "signature": d["signature"],
                                "previous_signature": d.get("previous_signature"), "time_unix": ts,
                                "time_utc": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
                    + "\n")
    print(f"[drand] verify: randomness!=sha256(sig): {bad_rand}; broken prev-signature links: {bad_link}")
    return 0 if bad_rand == 0 and bad_link == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
