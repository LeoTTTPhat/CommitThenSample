"""Microbenchmarks of the CtS primitives (DESIGN §3.11, E6), in µs per operation.

Run from P1_CommitThenSample (single process, < 1 min, < 200 MB):

    ../.venv/bin/python -I -m ctsample.bench.primitives [--out results/bench/p1a_primitives.json] [--quick]

Reports the median over ``repeats`` timing runs of: BLAKE3 frame hash (64/512/1500 B), frame_leaf,
unit_leaf, leaf0, Merkle root (n = 10², 10³, 10⁴, 10⁵ leaves; µs per tree and per leaf), ECVRF
prove/verify (pure Python), Ed25519 sign/verify (cryptography/OpenSSL), PRF word per unit, header
encode+hash, state commitment C^st (salt PRF + hash), AEAD seal of a 1,500 B frame record, integer
threshold per unit.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

from ..crypto import aead, ecvrf
from ..crypto import hashing as hs
from ..crypto.prf import selection_key, u_word
from ..crypto.signing import KeySet
from ..design import intpi
from ..utils.io import RESULTS, atomic_write_json


def _bench(fn, n: int, repeats: int) -> float:
    """Median µs per call of fn() over ``repeats`` runs of n calls."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        ts.append((time.perf_counter_ns() - t0) / n / 1000.0)
    return statistics.median(ts)


def run(quick: bool = False) -> dict:
    rep = 3 if quick else 5
    k = 0.2 if quick else 1.0
    out: dict = {"meta": {"python": platform.python_version(), "machine": platform.machine(),
                          "processor": platform.processor(), "quick": quick,
                          "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}}
    r = {}
    for size in (64, 512, 1500):
        fr = os.urandom(size)
        r[f"blake3_frame_{size}B"] = _bench(lambda: hs.frame_hash(fr), int(20000 * k), rep)
        r[f"frame_leaf_{size}B"] = _bench(lambda: hs.frame_leaf(1_672_531_200_000_000, 1, size, fr), int(20000 * k), rep)
    meta, meta0, root = os.urandom(128), os.urandom(56), os.urandom(32)
    r["unit_leaf"] = _bench(lambda: hs.unit_leaf(meta, root), int(20000 * k), rep)
    r["leaf0"] = _bench(lambda: hs.leaf0(meta0), int(20000 * k), rep)
    merkle = {}
    for n in (100, 1000, 10_000, 100_000):
        leaves = [os.urandom(32) for _ in range(n)]
        reps_n = max(1, int((200_000 // n) * k)) if n < 100_000 else 1
        t = _bench(lambda: hs.merkle_root(leaves), reps_n, rep)
        merkle[str(n)] = {"us_per_tree": t, "us_per_leaf": t / n}
    r["merkle_root"] = merkle
    keys = KeySet.from_master(b"bench")
    alpha = os.urandom(32 + 32 + 8 + 32)
    pi = ecvrf.prove(keys.sk_V, alpha)
    r["ecvrf_prove"] = _bench(lambda: ecvrf.prove(keys.sk_V, alpha), int(100 * k) or 20, rep)
    r["ecvrf_verify"] = _bench(lambda: ecvrf.verify(keys.pk_V, alpha, pi), int(100 * k) or 20, rep)
    msg = os.urandom(32)
    sig = keys.sk_G.sign(msg)
    r["ed25519_sign"] = _bench(lambda: keys.sk_G.sign(msg), int(5000 * k), rep)
    r["ed25519_verify"] = _bench(lambda: keys.sk_G.verify(sig, msg), int(5000 * k), rep)
    K = selection_key(ecvrf.proof_to_hash(pi))
    L = os.urandom(32)
    r["prf_u_word"] = _bench(lambda: u_word(K, 123, 4567, L), int(20000 * k), rep)
    from ..protocol.header import Header, StateOpening, state_salt
    op = StateOpening(salt=state_salt(keys.k_st, 5), Lam_A=1 << 40, Lam_S=1 << 30, D_prev=-3, floor_mode=0,
                      boost_set=(1, 7, 9), H_boost=os.urandom(32))
    h = Header(gw_id=os.urandom(16), e=5, t_e_us=1, T_us=30_000_000, n=100, n_pay=60, sum_s_pay=1, sum_s=2,
               R=os.urandom(32), H_store=os.urandom(32), H_theta=os.urandom(32), H_policy=os.urandom(32), tau_phi=4,
               C_st=op.commitment(), r_star=7, H_prev=os.urandom(32))
    r["header_encode_hash"] = _bench(lambda: h.digest(), int(20000 * k), rep)
    r["state_commitment"] = _bench(lambda: StateOpening(salt=state_salt(keys.k_st, 5), Lam_A=1 << 40, Lam_S=1 << 30,
                                                        D_prev=-3, floor_mode=0, boost_set=(1, 7, 9),
                                                        H_boost=op.H_boost).commitment(), int(20000 * k), rep)
    Ke = aead.epoch_key(os.urandom(32), 5)
    rec = os.urandom(1516)
    r["aead_seal_1516B"] = _bench(lambda: aead.seal(Ke, 5, 77, rec), int(5000 * k), rep)
    tphi, t16 = intpi.floor_tau("0.01"), intpi.t16_from_mix("0.2")
    LamA, LamS = intpi.lam_fix("10"), intpi.lam_fix("0.001")
    r["threshold_unit"] = _bench(lambda: intpi.threshold_unit(LamA, LamS, 50000, 9000, tphi, t16), int(20000 * k), rep)
    out["us"] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=RESULTS / "bench" / "p1a_primitives.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    res = run(quick=a.quick)
    atomic_write_json(a.out, res)
    for kname, v in res["us"].items():
        if isinstance(v, dict):
            for n, d in v.items():
                print(f"{kname}[n={n}]: {d['us_per_tree']:.1f} µs/tree, {d['us_per_leaf']:.3f} µs/leaf")
        else:
            print(f"{kname}: {v:.2f} µs")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
