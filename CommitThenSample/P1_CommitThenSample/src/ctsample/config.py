"""Configuration (DESIGN §3.10 defaults, §7.2 run identity).

    cfg = load_yaml("config/experiments/e2.yaml")     # deep-merged over config/defaults.yaml
    h = cfg_hash(cfg)                                  # BLAKE3(canonical JSON of cfg || hash of src/)

Dataclasses: Policy, Budget, Grid, StreamSpec, ExpConfig. ``ExpConfig.raw`` keeps the merged dict
(every key, including ones without a dataclass field) so the hash covers everything.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import yaml

__all__ = ["Policy", "Budget", "Grid", "StreamSpec", "ExpConfig", "load_yaml", "load_defaults", "cfg_hash",
           "src_hash", "canonical_json", "DEFAULTS_PATH"]

P1_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = P1_ROOT / "config" / "defaults.yaml"
SRC_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Policy:
    gamma: float = 0.3
    tau_mix: float = 0.2
    floor: str = "constant"
    tier: str = "single"
    kappa: float = 1.0
    Wc: float = 30.0
    L: float = math.log(16)
    Dmax_mult: float = 200.0
    kappa_b: float = 5.0
    W_b: int = 20
    rolling_W: int = 120
    alpha: float = 0.05


@dataclass(frozen=True)
class Budget:
    rho: float = 0.05
    mode: str = "horizon"
    rho_grid: tuple = (0.01, 0.02, 0.05, 0.10)


@dataclass(frozen=True)
class Grid:
    K_lcs: int = 16
    lcs_decades: float = 2.0
    V_ref_epochs: int = 2880
    ucs_w0: float = 0.5
    ucs_eta0_frac: float = 0.999
    ucs_n_low: int = 15
    ucs_decades: float = 4.0


@dataclass(frozen=True)
class StreamSpec:
    name: str
    path: str | None = None
    stream_id: int = 0
    kind: str = ""
    folds: tuple = ()


@dataclass
class ExpConfig:
    name: str = "default"
    policy: Policy = field(default_factory=Policy)
    budget: Budget = field(default_factory=Budget)
    grid: Grid = field(default_factory=Grid)
    streams: list = field(default_factory=list)
    designs: list = field(default_factory=list)
    reps: int = 500
    raw: dict = field(default_factory=dict)

    def mc_spec(self, **overrides):
        """An MCSpec with this config's grid/alpha/MC defaults."""
        from .sim.montecarlo import MCSpec
        mc = self.raw.get("mc", {})
        kw = dict(reps=self.reps, alpha=self.policy.alpha, K_lcs=self.grid.K_lcs, lcs_decades=self.grid.lcs_decades,
                  ucs_n_low=self.grid.ucs_n_low, ucs_decades=self.grid.ucs_decades, ucs_w0=self.grid.ucs_w0,
                  ucs_eta0_frac=self.grid.ucs_eta0_frac, report_every=int(mc.get("report_every_epochs", 30)),
                  width_reps=int(mc.get("width_reps", 200)), dcap_mode=mc.get("dcap_mode", "normal"),
                  void_random_p=float(mc.get("void_random_p", 0.3)), withhold_p=float(mc.get("withhold_p", 0.5)))
        kw.update(overrides)
        return MCSpec(**kw)


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_defaults(path=DEFAULTS_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def _from_raw(raw: dict, name: str) -> ExpConfig:
    pol = raw.get("policy", {})
    ctl = pol.get("controller", {})
    bst = pol.get("boost", {})
    policy = Policy(gamma=float(pol.get("gamma", 0.3)), tau_mix=float(pol.get("tau_mix", 0.2)),
                    floor=pol.get("floor", "constant"), tier=pol.get("tier", "single"),
                    kappa=float(ctl.get("kappa", 1.0)), Wc=float(ctl.get("Wc_epochs", 30)),
                    L=float(ctl.get("L", math.log(16))), Dmax_mult=float(ctl.get("Dmax_mult", 200)),
                    kappa_b=float(bst.get("kappa_b", 5.0)), W_b=int(bst.get("W_b_epochs", 20)),
                    rolling_W=int(pol.get("rolling_window_epochs", 120)), alpha=float(raw.get("alpha", 0.05)))
    b = raw.get("budget", {})
    budget = Budget(rho=float(b.get("rho", 0.05)), mode=b.get("mode", "horizon"),
                    rho_grid=tuple(b.get("rho_grid", (0.01, 0.02, 0.05, 0.10))))
    cs = raw.get("cs", {})
    lcs, ucs = cs.get("lcs", {}), cs.get("ucs", {})
    grid = Grid(K_lcs=int(lcs.get("K", 16)), lcs_decades=float(lcs.get("decades", 2.0)),
                V_ref_epochs=int(lcs.get("V_ref_epochs", 2880)), ucs_w0=float(ucs.get("w0", 0.5)),
                ucs_eta0_frac=float(ucs.get("eta0_frac", 0.999)), ucs_n_low=int(ucs.get("n_low", 15)),
                ucs_decades=float(ucs.get("decades", 4.0)))
    streams = [StreamSpec(**s) if isinstance(s, dict) else StreamSpec(name=str(s)) for s in raw.get("streams", [])]
    mc = raw.get("mc", {})
    reps = int(raw.get("reps", mc.get("reps_default", 500)))
    return ExpConfig(name=raw.get("name", name), policy=policy, budget=budget, grid=grid, streams=streams,
                     designs=list(raw.get("designs", [])), reps=reps, raw=raw)


def load_yaml(path=None, defaults=DEFAULTS_PATH, overrides: dict | None = None) -> ExpConfig:
    """Deep-merge defaults.yaml <- experiment yaml (optional) <- overrides (optional)."""
    raw = load_defaults(defaults)
    name = "default"
    if path is not None:
        raw = _deep_merge(raw, yaml.safe_load(Path(path).read_text()) or {})
        name = Path(path).stem
    if overrides:
        raw = _deep_merge(raw, overrides)
    return _from_raw(raw, name)


def _jsonable(o):
    if is_dataclass(o):
        return _jsonable(asdict(o))
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float):
        return float(repr(o)) if math.isfinite(o) else str(o)
    if isinstance(o, Path):
        return str(o)
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return _jsonable(o.item())
        if isinstance(o, np.ndarray):
            return _jsonable(o.tolist())
    except ImportError:  # pragma: no cover
        pass
    return o


def canonical_json(obj) -> str:
    return json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _blake3():
    try:
        from blake3 import blake3
        return blake3()
    except ImportError:  # pragma: no cover - documented fallback
        import hashlib
        return hashlib.blake2b(digest_size=32)


def src_hash(root: Path = SRC_ROOT) -> str:
    """BLAKE3 over (relative path, bytes) of every .py file under src/ctsample (sorted)."""
    h = _blake3()
    for p in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(4, "little") + rel)
        data = p.read_bytes()
        h.update(len(data).to_bytes(8, "little") + data)
    return h.hexdigest()


def cfg_hash(cfg, include_src: bool = True, n: int = 16) -> str:
    """Run identity: BLAKE3(canonical JSON of cfg [|| src hash]); first n hex chars."""
    payload = cfg.raw if isinstance(cfg, ExpConfig) and cfg.raw else cfg
    h = _blake3()
    h.update(canonical_json(payload).encode())
    if include_src:
        h.update(b"|src|" + src_hash().encode())
    return h.hexdigest()[:n]
