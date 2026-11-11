"""MLX LLM scorer: cached system prefix + right-padded batched suffixes (DESIGN §3.4.3, Alg. 3).

Input of one candidate = [chat-template head + system PREFIX + user header]  (KV cached ONCE)
                         [card + QUESTION]                                    (user message)
                         [user end + assistant header + "Answer:"]            (prefill)
One forward pass per batch: the prefix KV is broadcast along the batch axis, suffixes are
right-padded, the causal mask (aligned to the end of the prefix) keeps padded positions out
of every real position's attention, and the final hidden state is gathered at each row's last
real position. Logits for ' Yes', 'Yes', ' No', 'No' are computed in fp32 from that hidden
state and the dequantised output-embedding rows (the model's own bf16 logits are quantised to
~1/8 nat at |logit| ≈ 20, which would create ties in s_LLM), and

    s_LLM = logsumexp(ℓ' Yes', ℓ'Yes') − logsumexp(ℓ' No', ℓ'No')      (fp32)

Greedy/deterministic: there is no sampling at all (a single forward pass per batch).

Usage (always under memguard and the GPU lock)::

    with gpu_lock("P2", "p2_llm_qwen7b"):
        sc = MLXScorer("mlx-community/Qwen2.5-7B-Instruct-4bit", prompts.build_prefix())
        sc.run(((inc, cid), card) for ...), "P2_CertiChain/results/llm/CF-TON")
        sc.close()
"""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

from certichain.llm import prompts  # noqa: E402

GiB = 1024 ** 3
LOGIT_COLS = ("logit_sYes", "logit_Yes", "logit_sNo", "logit_No")
SENTINEL = "CERTICHAINSUFFIXSENTINEL7F3A"
SCHEMA_VERSION = 1


@dataclass
class ScorerConfig:
    batch_size: int | None = None          # None -> max(4, int(32 * memguard scale()))  (DESIGN §3.10)
    max_suffix_tokens: int = 256           # hard cap on suffix tokens (card+question+tail); cards target ≤ 220
    shard_size: int = 4096
    sort_by_length: bool = True            # batch similar lengths together inside a shard
    mem_limit_gb: float | None = None      # None -> 16 (≤ 7B) / 32 (30B-A3B)
    cache_limit_gb: float | None = None    # None -> 4 / 8
    revision: str | None = None            # None -> refs/main of the local HF cache entry
    date_string: str = "26 Jul 2024"       # pins the date line of the Llama-3 chat template
    require_gpu_lock: bool = True
    prompt_hash: str | None = None         # None -> prompts.prompt_hash(prefix)
    suffix_template: str = prompts.SUFFIX_TEMPLATE
    prefill: str = prompts.PREFILL
    answers: tuple[str, ...] = prompts.ANSWERS
    prefill_chunk: int = 4096              # prefix prefill chunk (tokens)
    pad_id: int | None = None              # None -> tokenizer pad/eos id (value is irrelevant: right padding)
    invariant: bool = True                 # fused q/k/v (+ padded MoE router): row results independent of batch
    pad_router: bool = True                # MoE only; ~+18% compute on Qwen3-30B-A3B
    min_rows: int = 160                    # every forward has B * L_pad >= min_rows (right padding)
    extra_meta: dict = field(default_factory=dict)


@dataclass
class ScoreResult:
    logits: np.ndarray       # [n, len(answers)] fp32, order = cfg.answers
    s_llm: np.ndarray        # [n] fp32
    logz: np.ndarray         # [n] fp32: log Z over the vocabulary (bf16 head; answer entries fp32) - diagnostic
    n_tok: np.ndarray        # [n] int32 suffix tokens fed (after truncation)
    ms: np.ndarray           # [n] fp32 amortised wall ms per candidate (batch ms / batch size)
    truncated: np.ndarray    # [n] bool


# ----------------------------------------------------------------------------- helpers
def model_tag(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1].lower()


def is_big_model(model_id: str) -> bool:
    m = model_id.lower()
    return any(x in m for x in ("30b", "32b", "a3b", "14b", "72b"))


def resolve_revision(model_id: str) -> str:
    """Commit hash of refs/main in the local HF cache (no network)."""
    p = Path(model_id)
    if p.exists():
        return f"local:{p.resolve()}"
    from huggingface_hub import constants
    ref = Path(constants.HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip()


def _assert_gpu_lock_held() -> None:
    import fcntl

    from certichain.utils.gpu_lock import LOCK_PATH, holder
    note = holder()
    if not note or int(note.get("pid", -1)) != os.getpid():
        raise RuntimeError("MLXScorer: take certichain.utils.gpu_lock.gpu_lock(...) before loading a model "
                           f"(lock note: {note})")
    fd = os.open(LOCK_PATH, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        raise RuntimeError("MLXScorer: GPU lock note names this pid but the flock is not held")
    except BlockingIOError:
        pass
    finally:
        os.close(fd)


def set_mlx_limits(model_id: str, mem_gb: float | None = None, cache_gb: float | None = None) -> tuple[float, float]:
    big = is_big_model(model_id)
    mem_gb = mem_gb if mem_gb is not None else (32.0 if big else 16.0)
    cache_gb = cache_gb if cache_gb is not None else (8.0 if big else 4.0)
    mx.set_memory_limit(int(mem_gb * GiB))
    mx.set_cache_limit(int(cache_gb * GiB))
    return mem_gb, cache_gb


class _TiledPrefixCache:
    """Per-layer cache holding the shared prefix K/V [1, H, P, D]; broadcast along the batch."""

    def __init__(self, keys: mx.array, values: mx.array):
        self.k, self.v = keys, values
        self.offset = keys.shape[2]          # RoPE offset of the first suffix token

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B = keys.shape[0]
        k = self.k if B == 1 else mx.broadcast_to(self.k, (B, *self.k.shape[1:]))
        v = self.v if B == 1 else mx.broadcast_to(self.v, (B, *self.v.shape[1:]))
        return mx.concatenate([k, keys], axis=2), mx.concatenate([v, values], axis=2)

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(N, offset=self.offset, return_array=return_array, window_size=window_size)


# ----------------------------------------------------------------------------- shape invariance
# MLX's 4-bit/8-bit GEMM picks a different (split-K) kernel when the output grid is small, so a
# row's result depends on the number of rows M in the call (measured on the M3 Ultra, MLX 0.32.4):
#   N ≥ 4608 (any K tested): identical for every M ≥ 33;   N = 2048..4096: identical for M ≥ 80..144;
#   N ≤ 1024 (k_proj/v_proj, MoE router): differs up to M = 256..2048.
# This made B = 1 differ from B = 32 by up to 1e-2 in s_LLM. Fixes (mathematically exact):
#   * q/k/v projections fused into ONE quantized matmul (N = 4608/5120) - no extra compute;
#   * MoE router (N = 128, 8-bit) zero-padded to N = 4608 - ~+18% compute on Qwen3-30B-A3B;
#   * every forward has ≥ min_rows rows (B·L_pad ≥ 160, right padding).
INVARIANT_N = 4608


class _Holder:
    """Plain attribute holder (nn.Module.__setattr__ would register arrays as parameters)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class FusedQKV(nn.Module):
    """q_proj, k_proj and v_proj as ONE quantized matmul (q_proj(x) computes all three)."""

    def __init__(self, q, k, v):
        super().__init__()
        self.weight = mx.concatenate([q["weight"], k["weight"], v["weight"]], axis=0)
        self.scales = mx.concatenate([q["scales"], k["scales"], v["scales"]], axis=0)
        if q.get("biases") is not None:
            self.biases = mx.concatenate([q["biases"], k["biases"], v["biases"]], axis=0)
        if "bias" in q:
            self.bias = mx.concatenate([q["bias"], k["bias"], v["bias"]], axis=0)
        self.group_size, self.bits, self.mode = q.group_size, q.bits, getattr(q, "mode", "affine")
        nq, nk = q["scales"].shape[0], k["scales"].shape[0]
        self.split_at = (nq, nq + nk)
        self._st = _Holder(x=None, kv=None)

    def _all(self, x):
        y = mx.quantized_matmul(x, self["weight"], scales=self["scales"], biases=self.get("biases"),
                                transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode)
        if "bias" in self:
            y = y + self["bias"]
        return mx.split(y, list(self.split_at), axis=-1)

    def __call__(self, x):                       # called as q_proj(x)
        q, k, v = self._all(x)
        self._st.x, self._st.kv = x, (k, v)
        return q

    def part(self, x, j):                        # k_proj(x) -> j = 1, v_proj(x) -> j = 2
        st = self._st
        if st.x is x and st.kv is not None:
            out = st.kv[j - 1]
            if j == 2:
                st.x, st.kv = None, None
            return out
        return self._all(x)[j]


class QKVPart(nn.Module):
    def __init__(self, src: FusedQKV, j: int):
        super().__init__()
        self._src, self._j = _Holder(m=src), j

    def __call__(self, x):
        return self._src.m.part(x, self._j)


class PaddedQLinear(nn.Module):
    """Quantized linear with zero rows appended so N ≥ INVARIANT_N; returns the first N outputs."""

    def __init__(self, lin, n_pad: int):
        super().__init__()
        n = lin["scales"].shape[0]
        self.n_out = n
        pad = n_pad - n
        z = lambda a: mx.zeros((pad, a.shape[1]), a.dtype)  # noqa: E731
        self.weight = mx.concatenate([lin["weight"], z(lin["weight"])])
        self.scales = mx.concatenate([lin["scales"], z(lin["scales"])])
        if lin.get("biases") is not None:
            self.biases = mx.concatenate([lin["biases"], z(lin["biases"])])
        if "bias" in lin:
            self.bias = lin["bias"]
        self.group_size, self.bits, self.mode = lin.group_size, lin.bits, getattr(lin, "mode", "affine")

    def __call__(self, x):
        y = mx.quantized_matmul(x, self["weight"], scales=self["scales"], biases=self.get("biases"),
                                transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode)
        y = y[..., : self.n_out]
        return y + self["bias"] if "bias" in self else y


def make_shape_invariant(model, pad_router: bool = True) -> dict:
    """Patch attention q/k/v (fused) and MoE routers (padded) in place; returns a report."""
    n_fused = n_router = 0
    skipped = set()
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        qkv = [attn.get(n) if attn is not None else None for n in ("q_proj", "k_proj", "v_proj")]
        if all(isinstance(m, nn.QuantizedLinear) for m in qkv) and \
                len({(m.group_size, m.bits, getattr(m, "mode", "affine"), m["weight"].shape[1]) for m in qkv}) == 1:
            src = FusedQKV(*qkv)
            attn.q_proj, attn.k_proj, attn.v_proj = src, QKVPart(src, 1), QKVPart(src, 2)
            n_fused += 1
        elif attn is not None:
            skipped.add("attention:" + type(qkv[0]).__name__)
        mlp = getattr(layer, "mlp", None)
        gate = mlp.get("gate") if isinstance(mlp, nn.Module) else None
        if pad_router and isinstance(gate, nn.QuantizedLinear) and gate["scales"].shape[0] < INVARIANT_N \
                and hasattr(mlp, "switch_mlp"):
            mlp.gate = PaddedQLinear(gate, INVARIANT_N)
            n_router += 1
    mx.eval(model.parameters())
    return {"fused_qkv_layers": n_fused, "padded_router_layers": n_router, "skipped": sorted(skipped)}


def _head(model):
    """(callable hidden->vocab logits, layer holding the output-embedding rows)."""
    lm = getattr(model, "lm_head", None)
    if lm is not None:
        return lm, lm
    emb = model.model.embed_tokens
    return emb.as_linear, emb


def _rows_fp32(layer, ids: Sequence[int]) -> tuple[mx.array, mx.array | None]:
    idx = mx.array(list(ids), dtype=mx.int32)
    if hasattr(layer, "scales"):                     # QuantizedLinear / QuantizedEmbedding
        biases = layer.get("biases", None) if hasattr(layer, "get") else getattr(layer, "biases", None)
        W = mx.dequantize(layer.weight[idx], layer.scales[idx], None if biases is None else biases[idx],
                          group_size=layer.group_size, bits=layer.bits,
                          mode=getattr(layer, "mode", "affine"), dtype=mx.float32)
    else:
        W = layer.weight[idx].astype(mx.float32)
    b = layer["bias"][idx].astype(mx.float32) if ("bias" in layer and layer.get("bias") is not None) else None
    return W, b


def _key(k: Any) -> tuple[str, int]:
    if isinstance(k, (tuple, list)):
        return str(k[0]), int(k[1])
    return "", int(k)


def iter_chunks(it: Iterable, n: int) -> Iterator[list]:
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# ----------------------------------------------------------------------------- scorer
class MLXScorer:
    def __init__(self, model_id: str, prefix_text: str, cfg: ScorerConfig | None = None):
        cfg = cfg or ScorerConfig()
        if cfg.require_gpu_lock:
            _assert_gpu_lock_held()
        self.mem_limits = set_mlx_limits(model_id, cfg.mem_limit_gb, cfg.cache_limit_gb)
        from mlx_lm import load
        revision = cfg.revision or resolve_revision(model_id)
        t0 = time.perf_counter()
        model, tokenizer = load(model_id, revision=None if revision.startswith("local:") else revision)
        mx.eval(model.parameters())
        self.load_s = time.perf_counter() - t0
        self._init(model, tokenizer, prefix_text, cfg, model_id, revision)

    @classmethod
    def from_model(cls, model, tokenizer, prefix_text: str, cfg: ScorerConfig | None = None,
                   model_id: str = "local/test-model", revision: str = "test") -> "MLXScorer":
        """Wrap an already-built model (tests: tiny random MLX models; no lock, no loading)."""
        self = cls.__new__(cls)
        self.mem_limits = (None, None)
        self.load_s = 0.0
        self._init(model, tokenizer, prefix_text, cfg or ScorerConfig(require_gpu_lock=False), model_id, revision)
        return self

    # ------------------------------------------------------------------ init
    def _init(self, model, tokenizer, prefix_text, cfg, model_id, revision):
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.model_id, self.revision, self.model_tag = model_id, revision, model_tag(model_id)
        self.prefix_text = prefix_text
        self.prompt_hash = cfg.prompt_hash or prompts.prompt_hash(
            prefix_text, cfg.suffix_template, cfg.prefill, cfg.answers)
        self.inner = model.model
        self._head_fn, head_layer = _head(model)
        self.n_truncated = 0
        self.invariance = make_shape_invariant(model, cfg.pad_router) if cfg.invariant else {}

        # chat template split: [head][user content][tail + prefill]
        msgs = [{"role": "system", "content": prefix_text}, {"role": "user", "content": SENTINEL}]
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 date_string=cfg.date_string)
        if rendered.count(SENTINEL) != 1:
            raise ValueError("chat template did not render the user sentinel exactly once")
        i = rendered.index(SENTINEL)
        self.head_text = rendered[:i]
        self.tail_text = rendered[i + len(SENTINEL):] + cfg.prefill
        self.prefix_ids = self._encode(self.head_text)
        self.tail_ids = self._encode(self.tail_text)
        self.n_prefix = len(self.prefix_ids)

        # answer tokens (single tokens expected; else first sub-token, recorded)
        self.answer_ids, self.answer_ntok = [], []
        for s in cfg.answers:
            ids = self._encode(s)
            self.answer_ids.append(int(ids[0]))
            self.answer_ntok.append(len(ids))
        if len(set(self.answer_ids)) != len(self.answer_ids):
            raise ValueError(f"answer strings map to duplicate token ids: {self.answer_ids}")
        if any(n != 1 for n in self.answer_ntok):
            print(f"[llm] WARNING: multi-token answers {dict(zip(cfg.answers, self.answer_ntok))}; "
                  "using the first sub-token", flush=True)
        self.W, self.b = _rows_fp32(head_layer, self.answer_ids)
        pad = cfg.pad_id
        if pad is None:
            pad = getattr(tokenizer, "pad_token_id", None)
            if pad is None:
                pad = getattr(tokenizer, "eos_token_id", None) or 0
        self.pad_id = int(pad)

        # tokenisation of the split pieces must equal tokenisation of the joint string
        probe = self.suffix_text("Incident clock starts at t0.\nU: t=+00:01 | V: t=+00:02")
        joint = self._encode(self.head_text + probe + self.tail_text)
        self.split_consistent = joint == self.prefix_ids + self._encode(probe) + self.tail_ids

        t0 = time.perf_counter()
        self.prefix_kv = self._prefill_prefix()
        self.prefix_s = time.perf_counter() - t0
        self.head_check = self._check_head()

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _prefill_prefix(self):
        from mlx_lm.models.cache import KVCache, make_prompt_cache
        cache = make_prompt_cache(self.model)
        if not all(type(c) is KVCache for c in cache):
            raise NotImplementedError(f"{self.model_id}: only plain KVCache models are supported "
                                      f"(got {sorted({type(c).__name__ for c in cache})})")
        ids = mx.array(self.prefix_ids, dtype=mx.int32)[None]
        step = max(1, int(self.cfg.prefill_chunk))
        for s in range(0, ids.shape[1], step):
            self.inner(ids[:, s:s + step], cache=cache)
            mx.eval([c.state for c in cache])
        kv = [(mx.contiguous(c.keys[:, :, :c.offset, :]), mx.contiguous(c.values[:, :, :c.offset, :]))
              for c in cache]
        mx.eval(kv)
        del cache
        return kv

    def _check_head(self) -> dict:
        """model(x) == head(inner(x)) on a short probe; fp32 rows == model head at the answer ids."""
        ids = mx.array(self.prefix_ids[: min(16, self.n_prefix)], dtype=mx.int32)[None]
        full = self.model(ids)[0, -1]
        hs = self.inner(ids)
        via = self._head_fn(hs)[0, -1]                 # same shapes as model(ids) -> same kernels
        mine = self._logits4(hs[0, -1:])[0]
        mx.eval(full, via, mine)
        same = bool(mx.array_equal(full, via).item())
        if not same:
            raise RuntimeError(f"{self.model_id}: model(x) != head(inner(x)); unsupported architecture")
        ref = np.array(via.astype(mx.float32))[self.answer_ids]
        diff = float(np.max(np.abs(ref - np.array(mine))))
        tol = 0.05 + 0.02 * float(np.max(np.abs(ref)))
        if diff > tol:
            raise RuntimeError(f"{self.model_id}: fp32 answer rows disagree with the model head ({diff:.3f} > {tol:.3f})")
        return {"model_eq_head_inner": same, "max_abs_fp32_vs_model_logit": round(diff, 5)}

    # ------------------------------------------------------------------ text -> ids
    def suffix_text(self, card: str) -> str:
        return self.cfg.suffix_template.format(card=card.rstrip("\n"))

    def suffix_ids(self, card: str) -> tuple[list[int], bool]:
        body = self._encode(self.suffix_text(card))
        cap = self.cfg.max_suffix_tokens - len(self.tail_ids)
        trunc = len(body) > cap
        if trunc:
            body = body[:cap]
        return body + self.tail_ids, trunc

    # ------------------------------------------------------------------ forward
    def _logits4(self, h_last: mx.array) -> mx.array:
        # row-wise multiply + reduce (not a GEMM): the result of a row never depends on B
        out = (h_last.astype(mx.float32)[:, None, :] * self.W[None, :, :]).sum(axis=-1)
        return out + self.b if self.b is not None else out

    def _logz(self, h_last: mx.array, l4: mx.array) -> mx.array:
        """log Z over the vocabulary: bf16 head logits, with the answer entries replaced by their fp32 values."""
        full = self._head_fn(h_last).astype(mx.float32)
        if getattr(self, "_ans_mask", None) is None or self._ans_mask.shape[0] != full.shape[-1]:
            m = np.zeros(full.shape[-1], bool)
            m[self.answer_ids] = True
            self._ans_mask = mx.array(m)
        rest = mx.logsumexp(mx.where(self._ans_mask, -mx.inf, full), axis=-1)
        return mx.logaddexp(rest, mx.logsumexp(l4, axis=-1))

    def _forward(self, batch: Sequence[Sequence[int]]):
        B = len(batch)
        lens = np.fromiter((len(x) for x in batch), dtype=np.int64, count=B)
        L = max(int(lens.max()), -(-int(self.cfg.min_rows) // B))          # ceil(min_rows / B)
        tok = np.full((B, L), self.pad_id, dtype=np.int32)
        for r, x in enumerate(batch):
            tok[r, : len(x)] = x
        cache = [_TiledPrefixCache(k, v) for k, v in self.prefix_kv]
        h = self.inner(mx.array(tok), cache=cache)                         # [B, L, hidden]
        h_last = h[mx.arange(B), mx.array(lens - 1)]                       # last real position
        l4 = self._logits4(h_last)                                         # [B, 4] fp32
        logz = self._logz(h_last, l4)
        k = len(self.cfg.answers) // 2
        s = mx.logsumexp(l4[:, :k], axis=-1) - mx.logsumexp(l4[:, k:], axis=-1)
        mx.eval(l4, s, logz)
        return np.array(l4), np.array(s), np.array(logz)

    def default_batch_size(self) -> int:
        if self.cfg.batch_size:
            return int(self.cfg.batch_size)
        try:
            from certichain.utils.mem import scale
            sc = scale()
        except Exception:
            sc = 1.0
        return max(4, int(32 * sc))

    def score_ids(self, ids: Sequence[Sequence[int]], batch_size: int | None = None,
                  sort: bool | None = None) -> ScoreResult:
        n = len(ids)
        B = batch_size or self.default_batch_size()
        sort = self.cfg.sort_by_length if sort is None else sort
        order = sorted(range(n), key=lambda i: (len(ids[i]), i)) if sort else list(range(n))
        na = len(self.cfg.answers)
        logits = np.zeros((n, na), np.float32)
        s = np.zeros(n, np.float32)
        logz = np.zeros(n, np.float32)
        ms = np.zeros(n, np.float32)
        for st in range(0, n, B):
            idx = order[st:st + B]
            t0 = time.perf_counter()
            l4, sb, zb = self._forward([ids[i] for i in idx])
            dt = (time.perf_counter() - t0) * 1000.0 / len(idx)
            logits[idx], s[idx], logz[idx], ms[idx] = l4, sb, zb, dt
        n_tok = np.fromiter((len(x) for x in ids), dtype=np.int32, count=n)
        return ScoreResult(logits, s, logz, n_tok, ms, np.zeros(n, bool))

    def score(self, cards: Sequence[str], batch_size: int | None = None, sort: bool | None = None) -> ScoreResult:
        enc = [self.suffix_ids(c) for c in cards]
        res = self.score_ids([e[0] for e in enc], batch_size, sort)
        res.truncated = np.array([e[1] for e in enc], dtype=bool)
        self.n_truncated += int(res.truncated.sum())
        return res

    def score_uncached(self, cards: Sequence[str]) -> ScoreResult:
        """Reference path (tests): full forward of prefix+suffix, no KV cache, one at a time."""
        k = len(self.cfg.answers) // 2
        out_l, out_s, out_z, ntok, ms, tr = [], [], [], [], [], []
        for c in cards:
            sfx, t = self.suffix_ids(c)
            t0 = time.perf_counter()
            h = self.inner(mx.array(self.prefix_ids + sfx, dtype=mx.int32)[None])[0, -1:]
            l4 = self._logits4(h)
            z = self._logz(h, l4)
            s = mx.logsumexp(l4[:, :k], axis=-1) - mx.logsumexp(l4[:, k:], axis=-1)
            mx.eval(l4, s, z)
            ms.append((time.perf_counter() - t0) * 1000.0)
            out_l.append(np.array(l4)[0]); out_s.append(float(np.array(s)[0])); out_z.append(float(np.array(z)[0]))
            ntok.append(len(sfx)); tr.append(t)
        return ScoreResult(np.array(out_l, np.float32), np.array(out_s, np.float32), np.array(out_z, np.float32),
                           np.array(ntok, np.int32), np.array(ms, np.float32), np.array(tr, bool))

    # ------------------------------------------------------------------ shards
    def metadata(self) -> dict:
        import mlx_lm
        return {
            "schema_version": SCHEMA_VERSION, "model_id": self.model_id, "model_revision": self.revision,
            "model_tag": self.model_tag, "prompt_hash": self.prompt_hash,
            "prefix_tokens": self.n_prefix, "tail_tokens": len(self.tail_ids),
            "prefix_render_blake3": __import__("blake3").blake3(self.head_text.encode()).hexdigest()[:16],
            "answers": list(self.cfg.answers), "answer_ids": self.answer_ids, "answer_ntok": self.answer_ntok,
            "max_suffix_tokens": self.cfg.max_suffix_tokens, "batch_size": self.default_batch_size(),
            "sort_by_length": self.cfg.sort_by_length, "date_string": self.cfg.date_string,
            "split_tokenization_consistent": self.split_consistent, "head_check": self.head_check,
            "logit_precision": "fp32 from bf16 final hidden state x dequantised output rows",
            "invariance": self.invariance, "min_rows": self.cfg.min_rows,
            "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
            "placeholder_fewshot": "[PLACEHOLDER]" in self.prefix_text, **self.cfg.extra_meta,
        }

    def shard_dir(self, out_dir: str | os.PathLike) -> Path:
        return Path(out_dir) / self.model_tag / self.prompt_hash

    def run(self, items: Iterable[tuple[Any, str]], out_dir: str | os.PathLike, *, check_memory: bool = True,
            log_every: int = 1) -> Path:
        """Score ``items`` = iterable of (key, card) with key = cand_id or (incident, cand_id).

        Writes ``<out_dir>/<model_tag>/<prompt_hash>/part-NNNNN.parquet`` (4,096 candidates each,
        atomic). Resumable: a complete shard whose keys match the chunk is skipped; a mismatching
        shard raises (never mix orders or prompts). ``items`` must be deterministic in order.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        mem_check = None
        if check_memory:
            try:
                from certichain.utils.mem import check_memory as mem_check
            except Exception:
                mem_check = None
        out = self.shard_dir(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = self.metadata()
        n_done = n_skip = 0
        t_start = time.perf_counter()
        for k, chunk in enumerate(iter_chunks(items, self.cfg.shard_size)):
            path = out / f"part-{k:05d}.parquet"
            keys = [_key(x[0]) for x in chunk]
            if path.exists():
                tbl = pq.read_table(path, columns=["incident", "cand_id"])
                old = list(zip(tbl.column(0).to_pylist(), tbl.column(1).to_pylist()))
                md = pq.read_metadata(path).metadata or {}
                same = (old == keys and md.get(b"prompt_hash", b"").decode() == self.prompt_hash
                        and md.get(b"model_revision", b"").decode() == self.revision)
                if not same:
                    raise RuntimeError(f"{path}: existing shard does not match this chunk/prompt/revision")
                n_skip += len(chunk)
                continue
            t0 = time.perf_counter()
            res = self.score([x[1] for x in chunk])
            dt = time.perf_counter() - t0
            cols = {"incident": pa.array([a for a, _ in keys], pa.string()),
                    "cand_id": pa.array([b for _, b in keys], pa.int64())}
            names = LOGIT_COLS if tuple(self.cfg.answers) == tuple(prompts.ANSWERS) else \
                tuple("logit_" + "".join(ch for ch in x.strip() if ch.isalnum()) for x in self.cfg.answers)
            for j, c in enumerate(names):
                cols[c] = pa.array(res.logits[:, j], pa.float32())
            cols.update(s_llm=pa.array(res.s_llm, pa.float32()), n_tok=pa.array(res.n_tok, pa.int32()),
                        ms=pa.array(res.ms, pa.float32()), logz=pa.array(res.logz, pa.float32()),
                        truncated=pa.array(res.truncated, pa.bool_()))
            shard_meta = dict(meta, shard=k, n=len(chunk), wall_s=round(dt, 3),
                              suffix_tokens_sum=int(res.n_tok.sum()), n_truncated=int(res.truncated.sum()),
                              created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            tbl = pa.table(cols).replace_schema_metadata({
                "certichain.llm": json.dumps(shard_meta), "model_revision": self.revision,
                "prompt_hash": self.prompt_hash, "model_id": self.model_id})
            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            pq.write_table(tbl, tmp)
            os.replace(tmp, path)
            n_done += len(chunk)
            if log_every and k % log_every == 0:
                el = time.perf_counter() - t_start
                print(f"[llm] {self.model_tag} shard {k:05d}: {len(chunk)} cands {dt:.1f}s "
                      f"({len(chunk) / dt:.2f} cand/s); scored {n_done}, skipped {n_skip}, {el / 60:.1f} min",
                      flush=True)
            if mem_check is not None:
                mem_check()
        manifest = dict(meta, n_scored=n_done, n_skipped=n_skip, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        wall_s=round(time.perf_counter() - t_start, 1), n_truncated_total=self.n_truncated)
        tmp = out / f"manifest.json.tmp{os.getpid()}"
        tmp.write_text(json.dumps(manifest, indent=1))
        os.replace(tmp, out / "manifest.json")
        return out

    def close(self) -> None:
        for a in ("prefix_kv", "model", "inner", "_head_fn", "W", "b"):
            if hasattr(self, a):
                delattr(self, a)
        gc.collect()
        mx.clear_cache()


def read_scores(shard_dir: str | os.PathLike):
    """Concatenate all shards of one <model_tag>/<prompt_hash> directory (polars DataFrame)."""
    import polars as pl
    files = sorted(Path(shard_dir).glob("part-*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()


def config_dict(cfg: ScorerConfig) -> dict:
    return asdict(cfg)
