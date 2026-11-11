"""design/treeeval.py: the integer forest equals the float64 reference within the LUT step on 10^6 rows,
the numpy traversal equals LightGBM's own traversal, and integer outputs are identical across two
processes and after a JSON round trip."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from ctsample.design import treeeval as te  # noqa: E402

N_TEST = 1_000_000


def _data(n, rng):
    """Integer features with mixed ranges (like units/features.py: codes, q8 values, sizes, counts)."""
    X = np.empty((n, 12), dtype=np.int32)
    X[:, 0] = rng.integers(0, 3, n)
    X[:, 1] = rng.integers(0, 16, n)
    X[:, 2] = rng.integers(0, 4, n)
    X[:, 3] = rng.integers(0, 2, n)
    X[:, 4] = rng.integers(0, 200, n)
    X[:, 5] = rng.integers(0, 160, n)
    X[:, 6] = rng.integers(-1500, 1500, n)
    X[:, 7] = rng.integers(0, 256, n)
    X[:, 8] = rng.integers(0, 100000, n)
    X[:, 9] = rng.integers(0, 65537, n)
    X[:, 10] = rng.integers(-5, 6, n)
    X[:, 11] = rng.integers(0, 13, n)
    z = (0.8 * (X[:, 1] == 0) + 0.002 * X[:, 5] - 0.0007 * np.abs(X[:, 6]) + 0.3 * X[:, 3] * (X[:, 0] == 1)
         + 0.00001 * X[:, 8] - 2.0 + 0.2 * np.sin(X[:, 4] / 9.0) - 0.15 * X[:, 10])
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(np.int32)
    return X, y


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    rng = np.random.default_rng(20261018)
    X, y = _data(60_000, rng)
    params = dict(objective="binary", num_leaves=31, learning_rate=0.05, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=1.0, is_unbalance=True, num_threads=8, seed=7, deterministic=True, verbose=-1)
    bst = lgb.train(params, lgb.Dataset(X.astype(np.float64), y, free_raw_data=False), num_boost_round=200)
    w1 = (y == 0).sum() / max((y == 1).sum(), 1)
    forest = te.compile_lgbm(bst, bias=-float(np.log(w1)))
    Xt, _ = _data(N_TEST, np.random.default_rng(99))
    d = tmp_path_factory.mktemp("forest")
    return bst, forest, Xt, d


def test_float_reference_is_lightgbm(model):
    bst, forest, Xt, _ = model
    sub = Xt[:200_000]
    raw_lgb = bst.predict(sub.astype(np.float64), raw_score=True)
    assert np.allclose(forest.raw_float(sub), raw_lgb + forest.bias_f, rtol=0, atol=1e-9)


def test_integer_within_lut_step(model):
    bst, forest, Xt, _ = model
    p_int = forest.predict_int(Xt)                        # LightGBM pred_leaf fast path (attached booster)
    assert p_int.dtype == np.int64 and p_int.min() >= 0 and p_int.max() <= 65535
    p_float = 1.0 / (1.0 + np.exp(-(bst.predict(Xt.astype(np.float64), raw_score=True) + forest.bias_f)))
    err = np.abs(p_int / 65536.0 - p_float)
    step = te.lut_max_step()
    T = forest.n_trees
    bound = 0.25 * (1.0 / 512 + (T + 1) * 2.0 ** -17) + 2.0 ** -16      # half bin + leaf rounding + LUT rounding
    assert step <= 0.25 / 256 + 2 ** -16
    assert err.max() <= min(step, bound) + 1e-12, (err.max(), step, bound)
    assert err.mean() < 0.25 * step


def test_numpy_traversal_equals_lightgbm_traversal(model):
    bst, forest, Xt, _ = model
    sub = Xt[:250_000]
    L_np = forest.leaves(sub)
    L_lgb = np.asarray(bst.predict(sub.astype(np.float64), pred_leaf=True), dtype=np.int64)
    assert np.array_equal(L_np, L_lgb)
    bare = te.IntForest.from_json(forest.to_json())       # no booster attached -> numpy traversal
    assert bare.booster is None
    assert np.array_equal(bare.predict_int(sub), forest.predict_int(sub))
    assert bare.theta_hash() == forest.theta_hash()


def test_identical_across_processes(model):
    bst, forest, Xt, d = model
    xp = d / "X.npy"
    np.save(xp, Xt)
    (d / "forest.json").write_text(forest.to_json())
    bst.save_model(str(d / "model.txt"))
    outp = d / "p_child.npy"
    code = (
        "import sys, numpy as np, lightgbm as lgb\n"
        "from ctsample.design.treeeval import IntForest\n"
        "d = sys.argv[1]\n"
        "f = IntForest.from_json(open(d + '/forest.json').read()).attach(lgb.Booster(model_file=d + '/model.txt'))\n"
        "np.save(d + '/p_child.npy', f.predict_int(np.load(d + '/X.npy')))\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(d)], check=True, timeout=600)
    p_child = np.load(outp)
    p_here = forest.predict_int(Xt)
    p_again = forest.predict_int(Xt)
    assert np.array_equal(p_child, p_here) and np.array_equal(p_here, p_again)


def test_constant_forest_and_lut():
    lut = te.sigmoid_lut()
    assert lut.dtype == np.uint16 and len(lut) == 65536
    assert np.all(np.diff(lut.astype(np.int64)) >= 0)
    assert lut[0] == 0 and lut[-1] == 65535 and abs(int(lut[32768]) - 32768) <= 64
    f = te.constant_forest(5, 0.01)
    X = np.zeros((10, 5), dtype=np.int32)
    p = f.predict_int(X)
    assert np.all(p == p[0]) and abs(p[0] / 65536 - 0.01) < 2 * te.lut_max_step()
    g = te.IntForest.from_dict(json.loads(f.to_json()))
    assert np.array_equal(g.predict_int(X), p)
