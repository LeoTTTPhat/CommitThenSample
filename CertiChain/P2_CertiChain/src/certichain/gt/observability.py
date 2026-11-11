"""Transition-level observability (DESIGN §2.3, §5.5, Remark R1) and the STR helper.

  ρ_T     = |T*_hid| / |T*|
  r_{C,T} = |{(A→B) ∈ T*_obs : (enab(A,B), first(B)) ∈ C}| / |T*_obs|
  ceiling = (1 − ρ_T) · r_{C,T}   ( = |T*_obs ∩ C| / |T*| ;  STR ≤ ceiling per incident )
  STR     = |{(A→B) ∈ T*_obs : (enab(A,B), first(B)) ∈ Ŝ}| / |T*|
  CER_link = |Ŝ ∩ {Y=1}| / (|E*_link| + |T*_hid|)
The per-incident transition table comes from ``semantics.transition_table`` (kind obs|hid|bridge).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import polars as pl


def rho_T(stats: dict) -> float | None:
    return stats["n_T_hid"] / stats["n_T"] if stats.get("n_T") else None


def r_CT(stats: dict) -> float | None:
    return stats["n_T_obs_in_C"] / stats["n_T_obs"] if stats.get("n_T_obs") else None


def ceiling(stats: dict) -> float | None:
    r, rho = r_CT(stats), rho_T(stats)
    return None if (r is None or rho is None) else (1.0 - rho) * r


def candidate_transitions(trans: pl.DataFrame) -> pl.DataFrame:
    """STR helper: cand_id → (src_ep, dst_ep, kind) for candidates that are the enabling pair
    of an observable transition (kind obs) or of a bridge (kind bridge)."""
    return (trans.filter(pl.col("cand_id").is_not_null() & pl.col("kind").is_in(["obs", "bridge"]))
            .select("cand_id", "src_ep", "dst_ep", "kind"))


def str_metrics(selected: Iterable[int], trans: pl.DataFrame, labels: pl.DataFrame | None = None) -> dict:
    """STR, bridge recall and CER_link of a selected candidate set Ŝ (cand_ids) in one incident."""
    sel = set(int(c) for c in selected)
    obs = trans.filter(pl.col("kind") == "obs")
    hid = trans.filter(pl.col("kind") == "hid")
    br = trans.filter(pl.col("kind") == "bridge")
    n_T = obs.height + hid.height
    hit = sum(1 for c in obs["cand_id"].to_list() if c is not None and c in sel)
    bhit = sum(1 for c in br["cand_id"].to_list() if c is not None and c in sel)
    out = {"STR": hit / n_T if n_T else None, "n_T": n_T, "n_T_obs_selected": hit,
           "bridge_recall": bhit / br.height if br.height else None}
    if labels is not None:
        lab = labels.filter(pl.col("cand_id").is_in(list(sel)))
        tp = int((lab["y"] == 1).sum())
        n_e = int(obs["enab_u"].is_not_null().sum()) + int(br["enab_u"].is_not_null().sum())
        out["CER_link"] = tp / (n_e + hid.height) if (n_e + hid.height) else None
    return out


def fdp_strict(y_selected: Sequence[int]) -> float:
    """(V' + A) / max(R, 1): AMB counted as false."""
    y = np.asarray(y_selected)
    return float(((y != 1).sum()) / max(len(y), 1))


def fdp_lenient(y_selected: Sequence[int]) -> float:
    """V' / max(R − A, 1): AMB excluded."""
    y = np.asarray(y_selected)
    r_eff = len(y) - int((y == -1).sum())
    return float((y == 0).sum() / max(r_eff, 1))


def pooled(stats_list: list[dict]) -> dict:
    """Pooled (Σ-ratio) and per-incident-mean observability over incidents."""
    def ssum(k):
        return sum(s.get(k, 0) or 0 for s in stats_list)
    n_T, n_obs, n_hid, n_in = ssum("n_T"), ssum("n_T_obs"), ssum("n_T_hid"), ssum("n_T_obs_in_C")
    per_r = [s["r_CT"] for s in stats_list if s.get("r_CT") is not None]
    per_rho = [s["rho_T"] for s in stats_list if s.get("rho_T") is not None]
    per_ceil = [s["ceiling"] for s in stats_list if s.get("ceiling") is not None]
    r = n_in / n_obs if n_obs else None
    rho = n_hid / n_T if n_T else None
    return {"n_incidents": len(stats_list), "n_T": n_T, "n_T_obs": n_obs, "n_T_hid": n_hid, "n_T_obs_in_C": n_in,
            "r_CT_pooled": r, "rho_T_pooled": rho,
            "ceiling_pooled": (1 - rho) * r if (r is not None and rho is not None) else None,
            "r_CT_mean": float(np.mean(per_r)) if per_r else None,
            "rho_T_mean": float(np.mean(per_rho)) if per_rho else None,
            "ceiling_mean": float(np.mean(per_ceil)) if per_ceil else None,
            "n_incidents_with_T_obs": len(per_r)}
