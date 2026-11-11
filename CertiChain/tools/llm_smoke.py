#!/usr/bin/env python3
"""Smoke-test one MLX LLM from the local Hugging Face cache (run it under memguard).

    python memguard/memguard.py run --name smoke_<short> --max-restarts 0 -- \
        python tools/llm_smoke.py mlx-community/Qwen2.5-7B-Instruct-4bit

Reports: load time, prompt-processing tok/s on a ~600-token chat prompt, decode tok/s for 16 new
tokens (greedy), and next-token log-probabilities of ' Yes' / ' No' (and 'Yes' / 'No') after a
yes/no question, computed with one forward pass of model(ids). The last line printed is
``RESULT {json}`` for machine parsing. Uses the HF cache only (HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "memguard"))
try:
    from guard import usage_gb  # process-tree physical footprint (counts Metal buffers)
except Exception:  # pragma: no cover
    usage_gb = lambda: float("nan")  # noqa: E731

GB = 1024 ** 3

SYSTEM = ("You are a network-forensics assistant for an IoT security operations centre. You read flow "
          "records and IDS alerts and answer precisely and briefly.")
LOG_LINES = [
    "ts=2026-10-18T12:00:{s:02d}Z uid=C{u:05d} src=192.168.1.{a} sport={p} dst=192.168.1.20 dport=80 proto=tcp "
    "service=http duration=0.{d:03d} orig_bytes={ob} resp_bytes={rb} conn_state=SF history=ShADadFf",
    "ts=2026-10-18T12:01:{s:02d}Z uid=C{u:05d} src=203.0.113.{a} sport={p} dst=192.168.1.30 dport=23 proto=tcp "
    "service=telnet duration=1.{d:03d} orig_bytes={ob} resp_bytes={rb} conn_state=S1 history=ShAD",
]
QUESTION = ("The camera at 192.168.1.30 received repeated telnet logins from 203.0.113.50 using the "
            "password 'xc3511', and Suricata raised 'ET MALWARE Linux.Mirai Login Attempt (xc3511)'. "
            "Is this traffic consistent with a Mirai-style brute-force attack? Answer Yes or No.")


def build_long_prompt(tokenizer, target: int = 600) -> tuple[str, int]:
    lines, i = [], 0
    while True:
        tmpl = LOG_LINES[i % 2]
        lines.append(tmpl.format(s=i % 60, u=10000 + i, a=10 + i % 200, p=40000 + i, d=(37 * i) % 1000,
                                 ob=100 + 13 * i, rb=900 + 71 * i))
        user = ("Summarise the following connection log in one sentence and point out anything suspicious.\n\n"
                + "\n".join(lines))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n >= target:
            return prompt, n
        i += 1


def single_token(tokenizer, s: str) -> tuple[int, int]:
    ids = tokenizer.encode(s, add_special_tokens=False)
    return ids[0], len(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--mem-limit-gb", type=float, default=60.0)
    ap.add_argument("--wired-limit-gb", type=float, default=60.0)
    ap.add_argument("--prompt-tokens", type=int, default=600)
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    info = mx.device_info()
    mx.set_memory_limit(int(a.mem_limit_gb * GB))
    wired = int(min(a.wired_limit_gb * GB, info["max_recommended_working_set_size"]))
    try:
        mx.set_wired_limit(wired)
    except Exception as e:  # wired limit must not exceed the OS limit
        print(f"[smoke] set_wired_limit failed: {e}", flush=True)
        wired = None
    res: dict = {"repo": a.repo, "mlx": mx.__version__, "device": info.get("device_name"),
                 "loadavg_start": [round(x, 1) for x in os.getloadavg()],
                 "memory_limit_gb": a.mem_limit_gb, "wired_limit_gb": (wired / GB if wired else None)}
    print(f"[smoke] {a.repo} on {info.get('device_name')} mlx={mx.__version__}", flush=True)

    t0 = time.perf_counter()
    model, tokenizer = load(a.repo)  # lazy=False -> weights are materialised here
    mx.eval(model.parameters())
    res["load_s"] = round(time.perf_counter() - t0, 2)
    res["active_gb_after_load"] = round(mx.get_active_memory() / GB, 2)
    res["footprint_gb_after_load"] = round(usage_gb(), 2)
    print(f"[smoke] loaded in {res['load_s']} s, active={res['active_gb_after_load']} GB", flush=True)

    # warm-up (kernel compilation) on a short prompt
    warm = tokenizer.apply_chat_template([{"role": "user", "content": "Say hi."}], tokenize=False,
                                         add_generation_prompt=True)
    for _ in stream_generate(model, tokenizer, warm, max_tokens=4):
        pass

    prompt, n_prompt = build_long_prompt(tokenizer, a.prompt_tokens)
    runs = []
    for rep in range(a.repeats):  # greedy, identical runs; the machine may be shared, so keep median + best
        last, text = None, ""
        t1 = time.perf_counter()
        for r in stream_generate(model, tokenizer, prompt, max_tokens=a.new_tokens):
            last = r
            text += r.text
        runs.append((last.prompt_tps, last.generation_tps, time.perf_counter() - t1))
        print(f"[smoke] run {rep + 1}: prompt {last.prompt_tokens} tok @ {last.prompt_tps:.1f} tok/s; decode "
              f"{last.generation_tokens} tok @ {last.generation_tps:.1f} tok/s; wall {runs[-1][2]:.2f} s", flush=True)
    pp, dd = sorted(x[0] for x in runs), sorted(x[1] for x in runs)
    res.update(prompt_tokens=last.prompt_tokens, new_tokens=last.generation_tokens, repeats=a.repeats,
               prompt_tps=round(pp[len(pp) // 2], 1), prompt_tps_best=round(pp[-1], 1),
               decode_tps=round(dd[len(dd) // 2], 1), decode_tps_best=round(dd[-1], 1),
               gen_wall_s=[round(x[2], 3) for x in runs], sample_output=text)
    print(f"[smoke] median prompt {res['prompt_tps']} tok/s (best {res['prompt_tps_best']}); median decode "
          f"{res['decode_tps']} tok/s (best {res['decode_tps_best']}); output={text!r}", flush=True)

    # yes/no next-token log-probabilities (single forward pass, no cache)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    qprompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"
    ids = tokenizer.encode(qprompt, add_special_tokens=False)
    t2 = time.perf_counter()
    logits = model(mx.array(ids)[None])[0, -1].astype(mx.float32)
    logp = logits - mx.logsumexp(logits)
    mx.eval(logp)
    res["yesno_forward_s"] = round(time.perf_counter() - t2, 3)
    out = {}
    for key, s in [("sp_yes", " Yes"), ("sp_no", " No"), ("yes", "Yes"), ("no", "No")]:
        tid, ntok = single_token(tokenizer, s)
        out[key] = {"token": s, "id": tid, "n_subtokens": ntok, "logp": round(logp[tid].item(), 4)}
    top = mx.argsort(-logp)[:5].tolist()
    res["yesno"] = out
    res["yesno_prompt_tokens"] = len(ids)
    res["logp_yes_minus_no"] = round(out["sp_yes"]["logp"] - out["sp_no"]["logp"], 4)
    res["logp_yes_minus_no_nospace"] = round(out["yes"]["logp"] - out["no"]["logp"], 4)
    res["top5_next"] = [(tokenizer.decode([t]), round(logp[t].item(), 3)) for t in top]
    res["mlx_peak_gib"] = round(mx.get_peak_memory() / GB, 2)  # whole run incl. load
    res["footprint_gb_end"] = round(usage_gb(), 2)
    res["loadavg_end"] = [round(x, 1) for x in os.getloadavg()]
    print(f"[smoke] logP(' Yes')-logP(' No') = {res['logp_yes_minus_no']}  (no-space variant "
          f"{res['logp_yes_minus_no_nospace']}); top5 = {res['top5_next']}", flush=True)
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
