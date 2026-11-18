"""Model selection rule and the saved final model (P6.T5).

Rule (``train.selection``, fixed in S09 before any P6 result was seen): among the model-zoo
classifiers on ``selection.feature_set`` under ``selection.scheme``, keep those whose 95 % CI
lower bound of balanced accuracy beats the BEST single-shot baseline's value; pick the highest
balanced accuracy; candidates within ``tie_tol`` of it are a tie -> the simplest (``models.zoo``
order). If no model meets the rule the best one is still reported, flagged ``meets_rule=False``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from foodlens.config import Config, config_hash
from foodlens.errors import E_MODEL_CAPABILITY, ModelError
from foodlens.evaluation.stats import git_commit
from foodlens.inference.ood import MahalanobisOOD
from foodlens.models.calibration import calibrated_factory
from foodlens.models.classical import tuned_factory


def select_model(table: pd.DataFrame, cfg: Config) -> dict:
    """Apply the selection rule to ``model_comparison.csv`` rows; returns the decision."""
    sel, zoo = cfg.train.selection, cfg.models.zoo
    t = table[(table["task"] == "classification") & (table["scheme"] == sel.scheme)
              & (table["metric"] == sel.metric)]  # fmt: skip
    base = t[t["model"].isin(sel.baseline_models)].sort_values("value", ascending=False)
    if base.empty:
        raise ModelError(
            E_MODEL_CAPABILITY, "no baseline rows in model_comparison.csv: run `foodlens evaluate`"
        )
    best_base = base.iloc[0]
    names = {f"{m}__{sel.feature_set}": i for i, m in enumerate(zoo)}
    cand = t[t["model"].isin(names)].copy()
    if cand.empty:
        raise ModelError(
            E_MODEL_CAPABILITY, "no model-zoo rows: run `foodlens train --experiment model_zoo`"
        )
    cand["beats_baseline"] = cand["ci_lo"] > best_base["value"]
    pool = cand[cand["beats_baseline"]] if cand["beats_baseline"].any() else cand
    top = pool["value"].max()
    ties = pool[pool["value"] >= top - sel.tie_tol].copy()
    ties["order"] = ties["model"].map(names)
    pick = ties.sort_values("order").iloc[0]
    return {
        "model": pick["model"], "family": pick["model"].split("__")[0],
        "feature_set": sel.feature_set, "scheme": sel.scheme, "metric": sel.metric,
        "value": float(pick["value"]), "ci": [float(pick["ci_lo"]), float(pick["ci_hi"])],
        "best_baseline": best_base["model"], "best_baseline_value": float(best_base["value"]),
        "meets_rule": bool(pick["beats_baseline"]),
        "tied_with": [m for m in ties["model"] if m != pick["model"]],
        "candidates": cand[["model", "value", "ci_lo", "ci_hi", "beats_baseline"]]
        .to_dict(orient="records"),
    }  # fmt: skip


def fit_final(df: pd.DataFrame, features: list[str], groups: np.ndarray, family: str,
              cfg: Config, calibrated: bool = True) -> dict:  # fmt: skip
    """Final bundle trained on ALL groups: classifier (probability-calibrated if ``calibrated``),
    equivalent-hour regressor (same family, nested grouped tuning) and the OOD detector (grouped
    threshold)."""
    x = df[features].to_numpy(dtype=float)
    ev = cfg.evaluation
    make = calibrated_factory if calibrated else tuned_factory
    clf = make(family, cfg)("classification")
    clf.fit(x, df[ev.class_target].to_numpy(str), groups=groups)
    reg = tuned_factory(family, cfg)("regression")
    reg.fit(x, df[ev.targets[0]].to_numpy(float), groups=groups)
    ood = MahalanobisOOD(cfg.train.ood_percentile, cfg.models.variance_threshold).fit(x, groups)
    return {"classifier": clf, "regressor": reg, "ood": ood, "features": list(features),
            "classes": list(cfg.labels.classes), "calibrated": calibrated,
            "regression_target": ev.targets[0],
            "equiv_h_thresholds": list(cfg.labels.equiv_h_thresholds),
            "n_train": int(len(df)), "n_groups": int(len(np.unique(groups))),
            "config_hash": config_hash(cfg), "git_commit": git_commit(cfg.root)}  # fmt: skip


def save_final(bundle: dict, card: str, cfg: Config) -> Path:
    """``<models_dir>/<final_model_name>/model.joblib`` + ``model_card.md``."""
    out = cfg.path("models_dir") / cfg.train.final_model_name
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "model.joblib")
    (out / "model_card.md").write_text(card)
    return out
