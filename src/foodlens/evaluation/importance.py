"""Interpretation (P6.T4): grouped permutation importance on held-out pieces + partial dependence.

Permutation importance: for every leave-one-piece-out fold the model is fitted (with its nested
grouped tuning) on the other pieces; on the HELD-OUT piece a column (or a family of columns,
shuffled together with one row permutation) is shuffled and the drop in the headline metric is
recorded (``train.importance_repeats`` shuffles). Positive = the model needs it on unseen
pieces. Families (one light, one feature group) avoid the "correlated twins share the credit"
problem of single columns.
Partial dependence: the final model's mean prediction when one feature is set to each value of a
grid (5-95th percentile of the data), all else kept as observed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from foodlens.config import Config
from foodlens.evaluation import metrics as M
from foodlens.evaluation.splits import logo_splits
from foodlens.features.schema import triplet_specs

PDP_QUANTILES = (0.05, 0.95)  # grid range of the partial-dependence curves (robust to outliers)


def feature_families(features: list[str], cfg: Config) -> dict[str, list[str]]:
    """``light:<RED|GREEN|BLUE|cross>`` and ``group:<name>`` column families of ``features``."""
    spec = {s.name: s for s in triplet_specs(cfg)}
    fam: dict[str, list[str]] = {}
    for f in features:
        s = spec[f]
        light = s.light if "/" not in s.light else "cross-light"
        fam.setdefault(f"light:{light}", []).append(f)
        fam.setdefault(f"group:{s.group}", []).append(f)
    return fam


def _score_fn(task: str) -> tuple[Callable, bool]:
    return (M.mae, False) if task == "regression" else (M.balanced_accuracy, True)


def grouped_permutation_importance(make: Callable, task: str, df: pd.DataFrame,
                                   features: list[str], target: str, groups: np.ndarray,
                                   cfg: Config) -> pd.DataFrame:  # fmt: skip
    """Importance of every single column and every family; one row per (item, fold).
    ``importance`` = metric drop when shuffled (MAE increase for regression)."""
    x = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(str if task == "classification" else float)
    items = {f: [i] for i, f in enumerate(features)}
    col = {f: i for i, f in enumerate(features)}
    items |= {k: [col[c] for c in v] for k, v in feature_families(features, cfg).items()}
    fn, greater = _score_fn(task)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for tr, te in logo_splits(groups):
        model = make(task)
        kw = {"groups": groups[tr]} if getattr(model, "uses_groups", False) else {}
        model.fit(x[tr], y[tr], **kw)
        xt, yt = x[te], y[te]
        base = fn(yt, model.predict(xt))
        reps = cfg.train.importance_repeats
        for name, cols in items.items():
            xp = np.tile(xt, (reps, 1))  # all shuffles in ONE predict call (much faster)
            for r in range(reps):
                xp[r * len(te) : (r + 1) * len(te), cols] = xt[rng.permutation(len(te))][:, cols]
            pred = model.predict(xp).reshape(reps, len(te))
            scores = np.array([fn(yt, p) for p in pred])
            drops = base - scores if greater else scores - base
            rows.append({"item": name, "kind": "family" if ":" in name else "feature",
                         "n_cols": len(cols), "fold": groups[te][0], "base": base,
                         "importance": float(np.mean(drops))})  # fmt: skip
    return pd.DataFrame(rows)


def summarize_importance(imp: pd.DataFrame) -> pd.DataFrame:
    """Mean / std over folds and the number of folds where the item helped (> 0)."""
    g = imp.groupby(["item", "kind", "n_cols"])["importance"]
    out = g.agg(mean="mean", std="std", n_folds="size").reset_index()
    out["folds_positive"] = g.apply(lambda s: int((s > 0).sum())).to_numpy()
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def partial_dependence(model, df: pd.DataFrame, features: list[str], feature: str,
                       n_grid: int) -> pd.DataFrame:  # fmt: skip
    """Mean prediction of ``model`` (fitted on ``features``) over a grid of ``feature``."""
    x = df[features].to_numpy(dtype=float)
    j = features.index(feature)
    v = x[:, j][np.isfinite(x[:, j])]
    grid = np.linspace(*np.quantile(v, PDP_QUANTILES), n_grid)
    mean = []
    for g in grid:
        xg = x.copy()
        xg[:, j] = g
        mean.append(float(np.mean(model.predict(xg))))
    return pd.DataFrame({"feature": feature, "value": grid, "mean_prediction": mean})
