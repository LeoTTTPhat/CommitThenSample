"""Tests for the Alg. 7 mimicry attacker (src/ctsample/adversary/mimicry.py; DESIGN Sec. 6.4, Prop. 4).

Fast unit tests of the manipulation grid, the metadata recomputation and the floor invariant run on synthetic
or light inputs. The end-to-end best-response invariants (attacked detection >= floor, miss <= (1-phi0)^k,
black-box weaker than white-box, splitting never beats the floor) run on the real LP-IoT23(34-1, 1%) stream
when its models/streams/designs are present (skipped otherwise)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ctsample.adversary import mimicry as MM
from ctsample.design import probs as P

RESULTS = Path(__file__).resolve().parents[1] / "results"
SID = "lp_iot23_34-1_p10000"
FOLD = "34-1"
_HAVE = (RESULTS / "streams" / SID / "DONE").exists() and (RESULTS / "models" / FOLD / "0" / "DONE").exists() \
    and (RESULTS / "designs" / SID / f"U6tau0.2_s0_bm_0.05.json").exists()


# ----------------------------------------------------------------------------- grid / manipulations
def test_grid_size_and_product():
    spec = MM.MimicrySpec()
    g = MM.manipulation_grid(spec)
    assert len(g) == spec.n_variants == 5 * 3 * 6 * 4 * 3 == 1080
    # the no-op base variant is present exactly once
    base = [m for m in g if m["pad"] == 1.0 and m["prefix"] == 0 and m["dport"] == "orig"
            and m["iat"] == 1 and m["split"] == 1]
    assert len(base) == 1


def test_small_grid_product():
    spec = MM.MimicrySpec(pad_factors=(1.0, 2.0), prefix_bytes=(0, 1024), dports=("orig", 80),
                          iat_factors=(1, 2), splits=(1, 2))
    assert spec.n_variants == 32 == len(MM.manipulation_grid(spec))


def test_prefix_pushes_first8_sizes():
    """A prefix of L >= 8*MSS bytes fills the committed first-8 sizes with prefix packets (content pushed out)."""
    from ctsample.units import features as F
    n = 3
    cols = {"pay": np.ones(n, bool), "proto": np.full(n, 6), "port_a": np.full(n, 5555),
            "port_b": np.full(n, 23), "device_id": np.zeros(n, np.int64), "first_ts": np.zeros(n, np.int64),
            "last_ts": np.full(n, 1000), "pk0": np.full(n, 4), "pk1": np.full(n, 2),
            "pb0": np.full(n, 400), "pb1": np.full(n, 100), "fb0": np.full(n, 500), "fb1": np.full(n, 150),
            "syn": np.zeros(n, np.int64), "synack": np.zeros(n, np.int64), "fin": np.zeros(n, np.int64),
            "rst": np.zeros(n, np.int64), "psh": np.full(n, 1), "cs": np.full(n, 2),
            "s": np.full(n, 700), "cont": np.zeros(n, np.int64), "fps": np.ones(n, np.int64),
            "sizes": np.tile(np.array([100, -40, 80, -20, 0, 0, 0, 0], np.int16), (n, 1)),
            "iats": np.tile(np.array([10, 12, 14, 16, 255, 255, 255, 255], np.uint8), (n, 1)),
            "ip_a": np.zeros((n, 16), np.uint8), "ip_b": np.zeros((n, 16), np.uint8),
            "flow_uid": np.zeros((n, 16), np.uint8)}
    cmask = np.array([True, False, True])
    manip = {"pad": 1.0, "prefix": 8 * 1448, "dport": "orig", "iat": 1, "split": 1}
    c2 = MM._apply_to_content(cols, cmask, manip, 1448)
    # content units: first-8 sizes are now the (positive) prefix packets, not the original content sizes
    assert np.all(c2["sizes"][0] == min(1448, 32767))
    assert np.all(c2["sizes"][2] == min(1448, 32767))
    # the non-content unit (index 1) is unchanged
    assert np.array_equal(c2["sizes"][1], cols["sizes"][1])
    # packet count and cost grew on the content units
    assert c2["pk0"][0] == 4 + 8 and c2["s"][0] > cols["s"][0]
    assert c2["pk0"][1] == cols["pk0"][1]


def test_dport_and_iat_manipulation():
    from ctsample.units import features as F
    n = 2
    cols = {"pay": np.ones(n, bool), "proto": np.full(n, 6), "port_a": np.full(n, 5555),
            "port_b": np.full(n, 23), "device_id": np.zeros(n, np.int64), "first_ts": np.zeros(n, np.int64),
            "last_ts": np.full(n, 1000), "pk0": np.full(n, 4), "pk1": np.full(n, 2),
            "pb0": np.full(n, 400), "pb1": np.full(n, 100), "fb0": np.full(n, 500), "fb1": np.full(n, 150),
            "syn": np.zeros(n, np.int64), "synack": np.zeros(n, np.int64), "fin": np.zeros(n, np.int64),
            "rst": np.zeros(n, np.int64), "psh": np.full(n, 1), "cs": np.full(n, 2),
            "s": np.full(n, 700), "cont": np.zeros(n, np.int64), "fps": np.ones(n, np.int64),
            "sizes": np.tile(np.array([100, -40, 80, -20, 0, 0, 0, 0], np.int16), (n, 1)),
            "iats": np.tile(np.array([10, 12, 255, 255, 255, 255, 255, 255], np.uint8), (n, 1)),
            "ip_a": np.zeros((n, 16), np.uint8), "ip_b": np.zeros((n, 16), np.uint8),
            "flow_uid": np.zeros((n, 16), np.uint8)}
    cmask = np.array([True, True])
    c2 = MM._apply_to_content(cols, cmask, {"pad": 1.0, "prefix": 0, "dport": 443, "iat": 4, "split": 1}, 1448)
    assert np.all(c2["port_b"] == 443)
    # IAT x4 adds 8*log2(4)=16 to present codes; missing (255) stays missing
    assert c2["iats"][0, 0] == min(10 + 16, 254) and c2["iats"][0, 2] == 255
    # the new dport bucket (443 = https) differs from telnet (23)
    assert F.dport_bucket(c2["proto"], c2["port_a"], c2["port_b"])[0] != \
        F.dport_bucket(cols["proto"], cols["port_a"], cols["port_b"])[0]


def test_attacked_pi_floor_invariant():
    """attacked_pi_cts never goes below the committed floor, for any utility and size; split raises it."""
    phi0 = 0.015
    cf = P.committed_floor(phi0)
    a_fix = np.array([0, 100, 1 << 20], dtype=np.int64)       # from zero utility to large
    s = np.array([210.0, 1400.0, 6000.0])
    pi1 = MM.attacked_pi_cts(a_fix, s, phi0, [300.0, 4e-5], "mixture", 0.2)
    assert np.all(pi1 >= cf - 1e-12)
    pi2 = MM.attacked_pi_cts(a_fix, s, phi0, [300.0, 4e-5], "mixture", 0.2, split=np.array([2, 2, 2]))
    assert np.all(pi2 >= pi1 - 1e-12)                          # 1-(1-pi)^2 >= pi


# ----------------------------------------------------------------------------- real-data best response
@pytest.mark.skipif(not _HAVE, reason="LP-IoT23(34-1) stream/models/designs not built")
def test_best_response_invariants_real():
    from ctsample.design.utility import UtilityBundle
    from ctsample.sim.tables import StreamTables
    T = StreamTables.load(RESULTS / "streams" / SID)
    cols_pay = dict(np.load(RESULTS / "streams" / SID / "cols_pay.npz"))
    b = UtilityBundle.load(RESULTS / "models" / FOLD / "0", attach_lgbm=True)
    Q = json.loads((RESULTS / "models" / FOLD / "0" / "Qnorm.json").read_text())["p10000"]
    dj = json.loads((RESULTS / "designs" / SID / "U6tau0.2_s0_bm_0.05.json").read_text())["spec"]
    cts = {"phi0": dj["phi0"], "lam": dj["lam"], "kind": dj["kind"], "tau_mix": dj["tau_mix"]}
    cf = P.committed_floor(dj["phi0"])
    devs = np.unique(np.asarray(T.device)[np.asarray(T.content, bool)])
    ctx = MM.load_device_context(T, cols_pay, devs)
    # a reduced grid keeps the test light while exercising every manipulation axis
    spec = MM.MimicrySpec(pad_factors=(1.0, 2.0), prefix_bytes=(0, 4096), dports=("orig", 443),
                          iat_factors=(1, 4), splits=(1, 2))
    sb = UtilityBundle.load(RESULTS / "models" / "1-1" / "0", attach_lgbm=True)
    sQ = json.loads((RESULTS / "models" / "1-1" / "0" / "Qnorm.json").read_text())["p10000"]
    br = MM.best_response(ctx, b, Q["active"], Q["W"], Q["Q2"], spec, cts_design=cts,
                          surrogate=(sb, sQ["active"], sQ["W"], sQ["Q2"]))
    # (1) the attacked detection never drops below the committed floor (Prop. 4(i),(iii))
    assert np.all(br["det_min"] >= cf - 1e-9)
    # (2) the attacker is never worse off than no manipulation
    assert np.all(br["det_min"] <= br["det_base"] + 1e-9)
    # (3) incident content-evidence miss <= (1 - phi0)^k under the white-box best response
    inc_c = ctx.incident[ctx.content]
    for i in np.unique(inc_c):
        m = inc_c == i
        miss = np.exp(np.log1p(-np.clip(br["det_min"][m], 0, 1)).sum())
        assert miss <= (1 - cf) ** int(m.sum()) * (1 + 1e-6)
    # (4) P_mal is minimised below the committed value
    assert np.all(br["pmal_min"] <= br["pmal_base"] + 1e-9)
    # (5) black-box transfer is no stronger than white-box (committed detection at the surrogate choice >= wb min)
    assert np.all(br["det_at_bb"] >= br["det_min"] - 1e-9)
    # (6) splitting never yields detection below the floor
    assert set(np.unique(br["split_min"]).tolist()).issubset(set(float(x) for x in spec.splits))
