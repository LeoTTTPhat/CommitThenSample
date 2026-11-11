"""S3b: build the LP / training / natural streams as StreamTables (DESIGN §5.2; recipes in config/streams.yaml).

    results/streams/<stream_id>/{s,pay,epoch,device,b,hit_*,label,incident,content,epoch_ptr}.npy, meta.json, DONE
                                features_pay.npy (payload units x 38, int32), units.parquet (test streams)
    results/streams/index.json   one row per stream (kind, fold, p, realised prevalence, sizes)

Run under memguard from the project root (resumable: complete streams have DONE and are skipped)::

    .venv/bin/python memguard/memguard.py run --name p1_s3b --workers 2 -- \
        .venv/bin/python -I -u P1_CommitThenSample/scripts/03b_build_streams.py [--kinds lp_iot23,train_lp,lp_edge,natural]
        [--only <substring>] [--force]

Workers: min(--workers, MEMGUARD_WORKERS); the big training streams (~11 M units) need ~15 GB each.

Recipe variants (rev1 sensitivity, review R3-M1(a); config/streams.yaml zero_payload_variants)::

    ... 03b_build_streams.py --variant zq [--only 7-1] [--workers 2]

builds lp_iot23_<k>_p<ppm>_<variant> and trainlp_<j>_p<ppm>_<variant> (seed host's pure zero-payload connections
thinned, streams/lp.thin_zero_payload) and indexes them in results/streams/index_<variant>.json ONLY (--index);
index.json is never touched by a variant run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctsample.utils.io import RESULTS, atomic_write_json, is_done  # noqa: E402
from ctsample.utils.mem import EXIT_MEMORY, MemoryBudgetExceeded, check_memory, workers  # noqa: E402


def _init_worker(ppid: int) -> None:
    """Pool workers exit as soon as the parent dies (no orphans after a kill or a memguard restart)."""
    import os
    import threading

    def watch():
        while True:
            if os.getppid() != ppid:
                os._exit(1)
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()


def _build_one(spec_json: dict, force: bool) -> dict:
    from ctsample.streams.lp import build_stream
    from ctsample.streams.recipe import Foreground, StreamSpec
    d = dict(spec_json)
    d["foregrounds"] = [Foreground(**f) for f in d["foregrounds"]]
    spec = StreamSpec(**d)
    try:
        meta = build_stream(spec, force=force, write_units=(spec.kind != "train_lp"),
                            log=lambda m: print(m, flush=True))
        ex = meta["extra"]
        return {"stream_id": spec.stream_id, "kind": spec.kind, "fold": spec.fold, "p": spec.p, "ok": True,
                "n_units": meta["n_units"], "n_epochs": meta["n_epochs"], "n_devices": meta["n_devices"],
                "n_payload_units": ex["n_payload_units"], "n_mal_payload_units": ex["n_mal_payload_units"],
                "realised_prevalence": ex["realised_prevalence"], "q": ex["q"],
                "payload_units_per_epoch": ex["payload_units_per_epoch"], "s_all_over_s_pay": ex["s_all_over_s_pay"],
                "n_incidents": ex["n_incidents"], "provisional_tables": ex["provisional_tables"],
                "suricata_ready": ex["suricata_ready"], "build_s": ex["build_s"]}
    except MemoryBudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return {"stream_id": spec.stream_id, "ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="lp_iot23,train_lp,lp_edge,natural")
    ap.add_argument("--only", default=None, help="substring filter on stream ids (comma-separated)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--variant", default=None, help="recipe variant (config zero_payload_variants), e.g. zq, z0")
    ap.add_argument("--index", default=None, help="index file name under results/streams (default index.json; "
                                                  "index_<variant>.json with --variant)")
    a = ap.parse_args()
    from ctsample.streams.recipe import ZERO_VARIANT_KINDS, all_specs, load_config, load_inventory, variant_specs
    cfg, inv = load_config(), load_inventory()
    if a.variant:
        specs = variant_specs(cfg, inv, a.variant, kinds=[k for k in a.kinds.split(",") if k in ZERO_VARIANT_KINDS])
    else:
        specs = all_specs(cfg, inv, kinds=tuple(a.kinds.split(",")))
    index_name = a.index or ("index.json" if not a.variant else f"index_{a.variant}.json")
    if a.variant and index_name == "index.json":
        raise SystemExit("variant streams must not be indexed in index.json (experiments discover streams there)")
    if a.only:
        keys = a.only.split(",")
        specs = [s for s in specs if any(k in s.stream_id for k in keys)]
    root = RESULTS / "streams"
    todo = [s for s in specs if a.force or not is_done(root / s.stream_id)]
    # big training streams first (they bound the wall time)
    todo.sort(key=lambda s: (s.kind != "train_lp", s.stream_id))
    nw = max(1, min(a.workers, workers(default=a.workers)))
    print(f"[s3b] {len(specs)} streams, {len(todo)} to build, {nw} workers", flush=True)
    t0 = time.time()
    results, failed = {}, []
    if todo:
        import os
        ex = ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn"), initializer=_init_worker,
                                 initargs=(os.getpid(),))
        try:
            pending = {ex.submit(_build_one, s.to_json(), a.force): s.stream_id for s in todo}
            while pending:
                done, _ = wait(set(pending), timeout=30, return_when=FIRST_COMPLETED)
                for f in done:
                    sid = pending.pop(f)
                    r = f.result()
                    results[sid] = r
                    if not r["ok"]:
                        failed.append(sid)
                        print(f"[s3b] FAILED {sid}: {r['error']}\n{r['traceback']}", flush=True)
                    else:
                        print(f"[s3b] done {sid} ({len(results)}/{len(todo)}, {time.time() - t0:.0f}s)", flush=True)
                check_memory()
        except MemoryBudgetExceeded as e:
            print(f"[mem] {e}; exiting {EXIT_MEMORY}", flush=True)
            ex.shutdown(wait=False, cancel_futures=True)
            return EXIT_MEMORY
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    # index over every complete stream
    idx = []
    for s in specs:
        d = root / s.stream_id
        if not is_done(d):
            continue
        m = json.loads((d / "meta.json").read_text())
        e = m["extra"]
        idx.append({"stream_id": s.stream_id, "kind": s.kind, "fold": s.fold, "p": s.p, "n_units": m["n_units"],
                    "n_epochs": m["n_epochs"], "n_devices": m["n_devices"], "n_payload_units": e["n_payload_units"],
                    "n_mal_payload_units": e["n_mal_payload_units"], "realised_prevalence": e["realised_prevalence"],
                    "q": e["q"], "payload_units_per_epoch": e["payload_units_per_epoch"],
                    "s_all_over_s_pay": e["s_all_over_s_pay"], "n_incidents": e["n_incidents"],
                    "provisional_tables": e["provisional_tables"], "suricata_ready": e["suricata_ready"],
                    "queries": [q["name"] for q in m["queries"]]})
        if a.variant:
            z = e.get("zero_thin") or {}
            zp = (z.get("parts") or [{}])[0]
            idx[-1].update({"variant": a.variant, "base_stream_id": z.get("base_stream_id"),
                            "models_root": f"models_{a.variant}", "zero_keep_prob": z.get("keep_prob"),
                            "n_seed_zero_units_before": zp.get("n_seed_zero_units_before"),
                            "n_seed_zero_units_after": zp.get("n_seed_zero_units_after"),
                            "n_pure_zero_conns_before": zp.get("n_pure_zero_conns_before"),
                            "n_pure_zero_conns_after": zp.get("n_pure_zero_conns_after")})
    old = {}
    ip = root / index_name
    if ip.exists():
        old = {r["stream_id"]: r for r in json.loads(ip.read_text()).get("streams", [])}
    old.update({r["stream_id"]: r for r in idx})
    atomic_write_json(ip, {"streams": sorted(old.values(), key=lambda r: r["stream_id"]),
                           "updated": time.strftime("%Y-%m-%d %H:%M:%S %z")})
    print(f"[s3b] finished: {len(results) - len(failed)} built, {len(failed)} failed, {time.time() - t0:.0f}s",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
