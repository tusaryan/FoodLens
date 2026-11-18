"""Leakage demo (P5.T4): the SAME RandomForest on the SAME data, scored with grouped CV (leave one
piece out) and with a LEAKY random split over triplets. Only the split changes.

Why the random split is wrong: neighbouring triplets of one piece are ~3 min apart and almost
identical, so a random split puts near-copies of each test photo in the training set. The model
then "recognises the piece and its time" instead of learning freshness, and the score is inflated.
Outputs ``reports/tables/leakage_demo.csv`` and figure ``F10b_leakage_demo``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from foodlens.config import Config
from foodlens.evaluation.cv import HEADLINE, evaluate
from foodlens.evaluation.splits import logo_splits, random_frame_splits_LEAKY

CAPTION = "Same model, same data. Only the split changed."


def rf_factory(cfg: Config):
    """Fixed-config RandomForest pipeline factory (median imputer fitted in the fold)."""
    ev = cfg.evaluation

    def make(task: str) -> Pipeline:
        kw = dict(n_estimators=ev.rf_n_estimators, min_samples_leaf=ev.rf_min_samples_leaf,
                  random_state=cfg.seed, n_jobs=cfg.n_jobs)  # fmt: skip
        model = (
            RandomForestRegressor(**kw)
            if task == "regression"
            else RandomForestClassifier(class_weight="balanced", **kw)
        )
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])

    return make


def run_leakage_demo(
    df: pd.DataFrame, features: list[str], groups: np.ndarray, cfg: Config
) -> pd.DataFrame:
    """Rows: split (grouped / LEAKY_random), task, target, metric, value, CI, n_groups."""
    make = rf_factory(cfg)
    tasks = [("regression", cfg.evaluation.targets[0]),
             ("classification", cfg.evaluation.class_target)]  # fmt: skip
    schemes = {"grouped_LOGO": logo_splits(groups),
               "LEAKY_random_KFold": random_frame_splits_LEAKY(len(df), cfg)}  # fmt: skip
    rows = []
    for task, target in tasks:
        name, _ = HEADLINE[task]
        for scheme, splits in schemes.items():
            res = evaluate(make, task, df, features, target, groups, splits, cfg, permute=False)
            lo, hi = res["ci"][name]
            rows.append({"split": scheme, "task": task, "target": target, "metric": name,
                         "value": res["metrics"][name], "ci_lo": lo, "ci_hi": hi,
                         "n_groups": len(np.unique(groups)), "n_samples": len(df),
                         "n_features": len(features)})  # fmt: skip
    return pd.DataFrame(rows)


def leakage_figure(table: pd.DataFrame, figures_dir: Path) -> Path:
    """F10b: bar per split for each task, with 95 % group-bootstrap CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from foodlens.viz import save_fig

    tasks = list(dict.fromkeys(table["task"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6))
    for ax, task in zip(np.atleast_1d(axes), tasks, strict=True):
        t = table[table["task"] == task]
        err = np.vstack([t["value"] - t["ci_lo"], t["ci_hi"] - t["value"]])
        colors = ["#3a6ea5" if s.startswith("grouped") else "#c0504d" for s in t["split"]]
        ax.bar(t["split"], t["value"], yerr=err, color=colors, capsize=6)
        unit = " (h)" if task == "regression" else ""
        ax.set_title(f"{t['target'].iloc[0]}: {t['metric'].iloc[0]}{unit}", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"F10b leakage demo. {CAPTION}", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "F10b_leakage_demo", figures_dir)
