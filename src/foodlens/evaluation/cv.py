"""Out-of-fold prediction and one evaluated run (P5.T3-T5).

A run = one model x one target x one CV scheme. Metrics are computed on the pooled out-of-fold
predictions; CIs by bootstrapping groups; the permutation test re-runs the whole CV.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from foodlens.config import Config
from foodlens.evaluation import metrics as M
from foodlens.evaluation.splits import Splits
from foodlens.evaluation.stats import bootstrap_group_ci, group_permutation_test

HEADLINE = {"regression": ("mae", False), "classification": ("balanced_accuracy", True)}
CI_METRICS = {
    "regression": {"mae": M.mae, "rmse": M.rmse, "spearman": M.spearman},
    "classification": {"balanced_accuracy": M.balanced_accuracy, "macro_f1": M.macro_f1},
}


def oof_predict(
    make: Callable[[str], Pipeline],
    task: str,
    x: np.ndarray,
    y: np.ndarray,
    splits: Splits,
    classes: list[str],
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Fit a fresh pipeline per split on train rows, predict its test rows. Models with
    ``uses_groups`` (nested grouped tuning, ``models.classical``) also get the TRAINING rows'
    groups, so their inner CV never sees the test fold.

    Returns ``(test_idx, yhat, proba)`` concatenated over splits (``proba`` columns follow
    ``classes``; a class absent from a training fold gets probability 0)."""
    idx, preds, probs = [], [], []
    for tr, te in splits:
        model = make(task)
        grouped = groups is not None and getattr(model, "uses_groups", False)
        kw = {"groups": groups[tr]} if grouped else {}
        model.fit(x[tr], y[tr], **kw)
        idx.append(te)
        preds.append(model.predict(x[te]))
        if task == "classification":
            p = np.zeros((len(te), len(classes)))
            col = {c: i for i, c in enumerate(classes)}
            for j, c in enumerate(model.classes_):
                p[:, col[c]] = model.predict_proba(x[te])[:, j]
            probs.append(p)
    proba = np.concatenate(probs) if probs else None
    return np.concatenate(idx), np.concatenate(preds), proba


def score(task: str, y: np.ndarray, yhat: np.ndarray, proba, groups, cfg: Config) -> dict:
    """All metrics of one run (see ``evaluation.metrics``)."""
    if task == "regression":
        return M.regression_metrics(y, yhat, groups)
    return M.classification_metrics(y, yhat, proba, cfg.labels.classes, cfg.cv.ece_bins)


def _permutation(make, task, x, y, groups, splits: Splits, cfg: Config, n: int) -> dict:
    """Grouped permutation test of the headline metric (the whole CV re-runs per repetition)."""
    name, better = HEADLINE[task]
    fn = CI_METRICS[task][name]

    def run(y_perm: np.ndarray) -> float:
        i, p, _ = oof_predict(make, task, x, y_perm, splits, cfg.labels.classes, groups)
        return fn(y_perm[i], p)

    res = group_permutation_test(run, y, groups, n, cfg.seed, greater_is_better=better)
    return res | {"metric": name}


def evaluate(
    make: Callable[[str], Pipeline],
    task: str,
    df: pd.DataFrame,
    features: list[str],
    target: str,
    groups: np.ndarray,
    splits: Splits,
    cfg: Config,
    permute: bool,
    n_permutations: int | None = None,
) -> dict:
    """Metrics, group-bootstrap CIs and (optionally) the grouped permutation test of one run.

    Returns ``{"metrics", "ci", "permutation", "n_test", "oof"}``; ``oof`` is a DataFrame of
    the test rows (index into ``df``, y, yhat, group and, for classification, the probability
    columns ``p_<class>``). ``n_permutations`` overrides ``cv.n_permutations``."""
    x = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy()
    if task == "classification":
        y = y.astype(str)
    classes = cfg.labels.classes
    te, yhat, proba = oof_predict(make, task, x, y, splits, classes, groups)
    yt, gt = y[te], groups[te]
    out = {"metrics": score(task, yt, yhat, proba, gt, cfg), "n_test": int(len(te))}
    cv = cfg.cv
    out["ci"] = {
        name: bootstrap_group_ci(yt, yhat, gt, fn, cv.n_bootstrap, cfg.seed, cv.ci_level)
        for name, fn in CI_METRICS[task].items()
    }
    out["permutation"] = {}
    if permute:
        n_perm = n_permutations or cv.n_permutations
        out["permutation"] = _permutation(make, task, x, y, groups, splits, cfg, n_perm)
    out["oof"] = pd.DataFrame({"row": df.index[te], "y": yt, "yhat": yhat, "group": gt})
    if proba is not None:
        for j, c in enumerate(classes):
            out["oof"][f"p_{c}"] = proba[:, j]
    return out
