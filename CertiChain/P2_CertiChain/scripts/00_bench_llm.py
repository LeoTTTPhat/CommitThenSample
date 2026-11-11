#!/usr/bin/env python3
"""LLM scorer throughput benchmark (DESIGN §3.4.3 "Throughput plan", §8.2 S5).

Measures cand/s of the cached-prefix scorer for B = 1 (prefix cached, one candidate per forward)
and B ∈ {8, 16, 32} (prefix KV tiled, right-padded suffixes), plus an uncached B = 1 reference
(full prefix+suffix forward), on realistic synthetic evidence cards (≈ 200-token suffixes) and the
real prefix (Appendix A task + 40-entry ATT&CK glossary + 4 [PLACEHOLDER] few-shot cards).
One model per process. Run from the project root under memguard:

    .venv/bin/python memguard/memguard.py run --name p2_bench_qwen7b --limit-gb 90 --workers 4 -- \
        .venv/bin/python -u P2_CertiChain/scripts/00_bench_llm.py --model qwen7b

Writes P2_CertiChain/results/bench/llm_<model_tag>.json (atomic).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "qwen7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "llama3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "qwen3moe": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
}

# ----------------------------------------------------------------------------- synthetic cards
_TYPES = [  # (type, proto/port service, hints)
    ("BRUTE_TELNET", "tcp/23 telnet", "T1110.001"), ("SCAN_H", "tcp/23,2323 telnet", "T1595.001, T1046"),
    ("INTERACTIVE_SESSION", "tcp/23 telnet", "T1021, T1059"), ("HTTP_DOWNLOAD", "tcp/80 http", "T1105"),
    ("C2_PERIODIC", "tcp/48101 other_tcp", "T1571, T1071"), ("FLOOD_SYN", "tcp/80 http", "T0814, T1498"),
    ("DNS_ANOMALY", "udp/53 dns", "T1568, T1071.004"), ("MQTT_PUBSUB", "tcp/1883 mqtt", "-"),
    ("BENIGN_GENERIC_TCP", "tcp/8080 http", "-"), ("REPORT_SMALL", "tcp/48101 other_tcp", "T1041"),
    ("RTSP_STREAM", "tcp/554 rtsp", "-"), ("TLS_SESSION", "tcp/443 https", "-"), ("SCAN_V", "tcp 1,024 ports", "T1046"),
    ("BRUTE_SSH", "tcp/22 ssh", "T1110.001"), ("WEB_ATTACK", "tcp/80 http", "T1190"),
]
_ROLES = ["camera,rtsp,telnet", "client", "mqtt-broker,mqtt", "web", "camera,rtsp", "dns", "plc,modbus", "telnet,web"]
_URIS = ["GET /bins/mirai.arm7", "GET /shell?cd+/tmp;wget+http", "POST /cgi-bin/luci/;stok=/locale", "GET /index.html",
         "GET /setup.cgi?next_file=netgear.cfg", "GET /bins.sh", "GET /api/v1/telemetry?id=7731"]
_UAS = ["Wget/1.21", "Hello, World", "Mozilla/5.0 (X11; Linux x86_64)", "python-requests/2.31", "curl/8.4.0"]
_NAMES = ["cnc.example-botnet.net", "xn--80ak6aa92e.example", "time.nist.gov", "q8z1kfp0a7.example",
          "api.vendor-cloud.example", "update.cam-firmware.example"]


def _bytes(rng) -> str:
    v = float(np.exp(rng.uniform(np.log(60), np.log(5e6))))
    return f"{v:.0f}B" if v < 1e3 else (f"{v / 1e3:.1f}kB" if v < 1e6 else f"{v / 1e6:.1f}MB")


def _dur(rng) -> str:
    s = float(np.exp(rng.uniform(np.log(0.3), np.log(3600))))
    return f"{s:.1f}s" if s < 10 else (f"{s:.0f}s" if s < 60 else f"{int(s // 60)}m{int(s % 60):02d}s")


def _host(rng, internal: bool) -> str:
    if rng.random() < 0.15:
        return f"INFRA{rng.integers(1, 4)}"
    return f"H{rng.integers(1, 40)}[{_ROLES[rng.integers(len(_ROLES))]}]" if internal else f"X{rng.integers(1, 25)}[ext]"


def _unit_line(rng, tag: str, t: int) -> str:
    typ, svc, hints = _TYPES[rng.integers(len(_TYPES))]
    n = int(np.exp(rng.uniform(0, np.log(5000))))
    fail = int(rng.uniform(0, 100))
    src = _host(rng, rng.random() < 0.5)
    dst = (f"{rng.integers(9, 256)} {'int' if rng.random() < 0.5 else 'ext'} hosts in {rng.integers(1, 40)} /24s"
           if typ.startswith(("SCAN", "SWEEP")) else _host(rng, rng.random() < 0.7))
    conn = (f"1 conn, {'established' if fail < 50 else 'failed'}" if n == 1 else
            f"{n:,} conns; {fail}% failed (S0/REJ), {int(n * (100 - fail) / 100):,} established")
    line = (f"{tag}: t=+{t // 60:02d}:{t % 60:02d} dur {_dur(rng)} | {src} -> {dst} {svc} | {conn} | "
            f"out {_bytes(rng)} in {_bytes(rng)} | type {typ} | hints {hints}")
    r = rng.random()
    if r < 0.35:
        line += f" | http ⟦{_URIS[rng.integers(len(_URIS))]}⟧ ua ⟦{_UAS[rng.integers(len(_UAS))]}⟧"
    elif r < 0.55:
        line += f" | dns ⟦{_NAMES[rng.integers(len(_NAMES))]}⟧"
    elif r < 0.65:
        line += f" | sni ⟦{_NAMES[rng.integers(len(_NAMES))]}⟧"
    return line


def synthetic_cards(n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    rel = ["same victim {h}; different source", "same source {h}", "pivot: U's target {h} is V's source",
           "U's targets include V's target {h}; different source", "shared infrastructure INFRA2; different source",
           "same source {h}; same victim X3"]
    cards = []
    for _ in range(n):
        tu = int(rng.integers(0, 3000))
        tv = tu + int(rng.integers(1, 1800))
        gap = int(rng.integers(1, 900))
        h = f"H{rng.integers(1, 40)}"
        cards.append("Incident clock starts at t0.\n" + _unit_line(rng, "U", tu) + "\n" + _unit_line(rng, "V", tv)
                     + f"\nRelation: V starts {gap}s after U ends; {rel[rng.integers(len(rel))].format(h=h)}.")
    return cards


_UTYPES = [  # (type, svc_class, dport, proto, dir, n_records range, fail range, flags)
    ("BRUTE_TELNET", "telnet", 23, "tcp", "ext->int", (20, 2000), (0.6, 0.99), 0),
    ("SCAN_H", "telnet", 23, "tcp", "ext->int", (200, 6000), (0.9, 1.0), 0),
    ("SCAN_H", "telnet", 23, "tcp", "int->ext", (200, 6000), (0.9, 1.0), 0),
    ("INTERACTIVE_SESSION", "telnet", 23, "tcp", "ext->int", (1, 3), (0.0, 0.0), 0),
    ("HTTP_DOWNLOAD", "http", 80, "tcp", "int->ext", (1, 4), (0.0, 0.0), 5),
    ("C2_PERIODIC", "other_tcp", 48101, "tcp", "int->ext", (5, 300), (0.0, 0.2), 0),
    ("FLOOD_SYN", "http", 80, "tcp", "int->ext", (500, 50000), (0.9, 1.0), 0),
    ("DNS_ANOMALY", "dns", 53, "udp", "int->ext", (20, 400), (0.0, 0.3), 24),
    ("DNS_ACTIVITY", "dns", 53, "udp", "int->ext", (1, 30), (0.0, 0.1), 0),
    ("MQTT_PUBSUB", "mqtt", 1883, "tcp", "int->int", (2, 200), (0.0, 0.1), 0),
    ("RTSP_STREAM", "rtsp", 554, "tcp", "int->int", (1, 10), (0.0, 0.1), 0),
    ("TLS_SESSION", "https", 443, "tcp", "int->ext", (1, 40), (0.0, 0.1), 0),
    ("REPORT_SMALL", "report48101", 48101, "tcp", "int->ext", (1, 1), (0.0, 0.0), 0),
    ("WEB_ATTACK", "http", 80, "tcp", "ext->int", (3, 100), (0.0, 0.5), 2),
    ("BENIGN_GENERIC_TCP", "other_tcp", 8080, "tcp", "int->int", (1, 50), (0.0, 0.5), 0),
]


def rendered_cards(n: int, seed: int, count_tokens) -> list[str]:
    """Cards from features/cards.py on synthetic incidents (budget enforced with the model tokenizer)."""
    from certichain.features import cards as C
    rng = np.random.default_rng(seed + 1)
    out: list[str] = []
    while len(out) < n:
        t0 = 1.7e9
        ints = [f"192.168.{rng.integers(0, 3)}.{i}" for i in rng.choice(250, 30, replace=False) + 2]
        exts = [f"203.0.{rng.integers(0, 255)}.{i}" for i in rng.choice(250, 25, replace=False) + 2]
        infra = exts[:3]
        units, ts = [], 0.0
        for uid in range(80):
            typ, svc, port, proto, dc, nr, fr, flags = _UTYPES[rng.integers(len(_UTYPES))]
            ts += float(rng.exponential(45))
            src = ints[rng.integers(len(ints))] if dc.startswith("int") else exts[rng.integers(len(exts))]
            pool = ints if dc.endswith("int") else exts
            nd = int(rng.integers(9, 256)) if typ == "SCAN_H" else 1
            dsts = [pool[i] for i in rng.choice(len(pool), min(nd, len(pool)), replace=False)]
            if typ == "SCAN_H":
                dsts = [f"{'192.168' if dc.endswith('int') else '198.18'}.{rng.integers(0, 4)}.{i}" for i in range(nd)]
            n_rec = int(rng.integers(nr[0], nr[1] + 1))
            dur = float(np.exp(rng.uniform(np.log(0.5), np.log(900))))
            u = {"unit_id": uid, "ts": t0 + ts, "te": t0 + ts + dur, "dur": dur, "src": src, "dsts": dsts,
                 "n_dst": len(dsts), "fanout": len(dsts) >= 8, "svc_class": svc, "dir_class": dc, "proto": proto,
                 "dports": [port] if typ != "SCAN_H" or rng.random() < 0.6 else [23, 2323], "type": typ,
                 "n_dports": 1, "n_records": n_rec, "obytes": float(np.exp(rng.uniform(4, 15))),
                 "rbytes": float(np.exp(rng.uniform(3, 16))), "fail_frac": float(rng.uniform(*fr)),
                 "start_cv": float(rng.uniform(0.01, 1.5)) if n_rec >= 5 else float("nan"), "content_flags": flags,
                 "dns_names": [_NAMES[rng.integers(len(_NAMES))]] if svc == "dns" else [],
                 "http_hosts": ["fw.vendor.example"] if svc == "http" and rng.random() < 0.5 else [],
                 "tls_snis": [_NAMES[rng.integers(len(_NAMES))]] if svc == "https" else [],
                 "infra": [h for h in [src, *dsts[:1]] if h in infra]}
            u["n_dports"] = len(u["dports"])
            units.append(u)
        text = {u["unit_id"]: {"uri": [_URIS[rng.integers(len(_URIS))]], "ua": [_UAS[rng.integers(len(_UAS))]]}
                for u in units if u["svc_class"] == "http" and rng.random() < 0.7}
        ctx = C.build_context(units, t0=t0, unit_text=text)
        for _ in range(64):                      # candidate pairs sharing an endpoint, V after U
            i, j = sorted(rng.choice(len(units), 2, replace=False))
            u, v = units[i], units[j]
            if not ({u["src"], *u["dsts"]} & {v["src"], *v["dsts"]}):
                continue
            cand = {"lag_start": v["ts"] - u["ts"], "lag_gap": v["ts"] - u["te"], "overlap": v["ts"] <= u["te"],
                    "rel_mask": 0}
            out.append(C.render_card(u, v, cand, ctx, count_tokens=count_tokens))
            if len(out) >= n:
                break
    return out


# ----------------------------------------------------------------------------- bench
def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="repo id or alias: " + ", ".join(ALIASES))
    ap.add_argument("--batch-sizes", default="1,8,16,32")
    ap.add_argument("--n", type=int, default=512, help="cards per batched run")
    ap.add_argument("--n-b1", type=int, default=128, help="cards for the B=1 run")
    ap.add_argument("--n-uncached", type=int, default=16, help="cards for the uncached B=1 reference (0: skip)")
    ap.add_argument("--no-invariant", action="store_true", help="disable fused-QKV/padded-router invariance patch")
    ap.add_argument("--no-pad-router", action="store_true", help="keep the MoE router unpadded")
    ap.add_argument("--min-rows", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cards", choices=["rendered", "template"], default="rendered",
                    help="rendered: features/cards.py on synthetic incidents (≤ 220-token budget); "
                         "template: older free-text generator (longer, up to the 256 cap)")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--out", default=str(ROOT / "results" / "bench"))
    a = ap.parse_args()

    import mlx.core as mx

    from certichain.llm import prompts
    from certichain.llm.mlx_scorer import MLXScorer, ScorerConfig, model_tag
    from certichain.utils.gpu_lock import gpu_lock, holder
    from certichain.utils.mem import usage_gb

    model_id = ALIASES.get(a.model, a.model)
    tag = model_tag(model_id) + (f"_{a.tag}" if a.tag else "")
    sizes = [int(x) for x in a.batch_sizes.split(",") if x]
    prefix = prompts.build_prefix()
    res: dict = {"model_id": model_id, "host": platform.node(), "loadavg_start": [round(x, 1) for x in os.getloadavg()],
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv[1:],
                 "prompt_hash": prompts.prompt_hash(prefix), "placeholder_fewshot": prompts.uses_placeholders()}
    print(f"[bench] {model_id}: batch sizes {sizes}, n={a.n}, n_b1={a.n_b1}", flush=True)
    with gpu_lock("P2", f"p2_bench_{tag}"):
        res["gpu_lock"] = holder()
        cfg = ScorerConfig(batch_size=max(sizes), invariant=not a.no_invariant, pad_router=not a.no_pad_router,
                           min_rows=a.min_rows)
        sc = MLXScorer(model_id, prefix, cfg)
        res.update(revision=sc.revision, load_s=round(sc.load_s, 2), prefix_tokens=sc.n_prefix,
                   tail_tokens=len(sc.tail_ids), prefix_prefill_s=round(sc.prefix_s, 3), tail_text=sc.tail_text,
                   answer_ids=sc.answer_ids, answer_ntok=sc.answer_ntok, split_consistent=sc.split_consistent,
                   head_check=sc.head_check, invariance=sc.invariance, mem_limits_gb=sc.mem_limits,
                   mlx=mx.__version__, device=mx.device_info().get("device_name"))
        n_cards = max(a.n, a.n_b1, a.n_uncached, max(sizes))
        if a.cards == "rendered":
            cnt = lambda t: len(sc._encode(t))  # noqa: E731  (the scoring model's own tokenizer)
            cards = rendered_cards(n_cards, a.seed, cnt)
        else:
            cards = synthetic_cards(n_cards, a.seed)
        res["cards"] = a.cards
        res["card_examples"] = cards[:3]
        ntok = np.array([len(sc.suffix_ids(c)[0]) for c in cards])
        res["suffix_tokens"] = {"mean": round(float(ntok.mean()), 1), "min": int(ntok.min()), "max": int(ntok.max()),
                                "p50": float(np.median(ntok)), "card_only_mean": round(float(ntok.mean()) - len(sc.tail_ids)
                                                                                     - len(sc._encode("\n" + prompts.QUESTION)), 1)}
        print(f"[bench] loaded in {sc.load_s:.1f}s; prefix {sc.n_prefix} tok (prefill {sc.prefix_s:.2f}s); "
              f"suffix tokens mean {ntok.mean():.1f} [{ntok.min()}, {ntok.max()}]; invariance {sc.invariance}", flush=True)
        sc.score(cards[:max(sizes)], batch_size=max(sizes))          # warm-up (kernel compilation)
        sc.score(cards[:2], batch_size=1)
        runs, ref = {}, None
        for B in sizes:
            n = a.n_b1 if B == 1 else a.n
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            r = sc.score(cards[:n], batch_size=B)
            wall = time.perf_counter() - t0
            if B == 1:
                ref = r
            run = {"B": B, "n": n, "wall_s": round(wall, 3), "cand_per_s": round(n / wall, 3),
                   "suffix_tok_per_s": round(float(r.n_tok.sum()) / wall, 1),
                   "ms_per_cand_mean": round(float(r.ms.mean()), 2),
                   "mlx_peak_gib": round(mx.get_peak_memory() / 2 ** 30, 2), "footprint_gb": round(usage_gb(), 2),
                   "loadavg": [round(x, 1) for x in os.getloadavg()]}
            if ref is not None:
                m = min(len(ref.s_llm), n)
                run["max_abs_ds_vs_B1"] = float(np.max(np.abs(r.s_llm[:m] - ref.s_llm[:m])))
            runs[str(B)] = run
            print(f"[bench] B={B:3d}: {n} cands in {wall:.2f}s = {n / wall:.2f} cand/s "
                  f"({run['suffix_tok_per_s']:.0f} suffix tok/s); MLX peak {run['mlx_peak_gib']} GiB; "
                  f"footprint {run['footprint_gb']} GB; max|ds| vs B=1 {run.get('max_abs_ds_vs_B1')}", flush=True)
        res["runs"] = runs
        if a.n_uncached:
            t0 = time.perf_counter()
            ru = sc.score_uncached(cards[:a.n_uncached])
            wall = time.perf_counter() - t0
            m = min(a.n_uncached, len(ref.s_llm)) if ref is not None else 0
            res["uncached_B1"] = {"n": a.n_uncached, "wall_s": round(wall, 3), "cand_per_s": round(a.n_uncached / wall, 3),
                                  "max_abs_ds_vs_cached": float(np.max(np.abs(ru.s_llm[:m] - ref.s_llm[:m]))) if m else None}
            print(f"[bench] uncached B=1: {a.n_uncached / wall:.2f} cand/s; max|ds| vs cached "
                  f"{res['uncached_B1']['max_abs_ds_vs_cached']}", flush=True)
        s = (ref if ref is not None else r).s_llm
        lz = (ref if ref is not None else r)
        res["s_llm_synthetic"] = {"mean": round(float(s.mean()), 3), "sd": round(float(s.std()), 3),
                                  "frac_pos": round(float((s > 0).mean()), 3),
                                  "answer_mass_mean": round(float(np.mean(np.exp(
                                      np.logaddexp.reduce(lz.logits, axis=1) - lz.logz))), 4)}
        best = max(runs.values(), key=lambda x: x["cand_per_s"])
        res["best"] = {"B": best["B"], "cand_per_s": best["cand_per_s"]}
        res["mlx_peak_gib_overall"] = round(mx.get_peak_memory() / 2 ** 30, 2)
        res["footprint_gb_end"] = round(usage_gb(), 2)
        sc.close()
    res["loadavg_end"] = [round(x, 1) for x in os.getloadavg()]
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out = Path(a.out) / f"llm_{tag}.json"
    _atomic_json(out, res)
    print(f"[bench] best {best['cand_per_s']:.2f} cand/s at B={best['B']}; wrote {out}", flush=True)
    print("RESULT " + json.dumps({"model": model_id, "best": res["best"],
                                  "runs": {k: v["cand_per_s"] for k, v in runs.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
