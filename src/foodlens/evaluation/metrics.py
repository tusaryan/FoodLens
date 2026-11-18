"""Regression and classification metrics (P5.T3). Hours in h; probabilities in [0, 1]."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(mean_absolute_error(y, yhat))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    """Spearman rho; NaN when either side is constant (e.g. the mean baseline per fold)."""
    if np.ptp(np.asarray(y, float)) == 0 or np.ptp(np.asarray(yhat, float)) == 0:
        return float("nan")
    return float(spearmanr(y, yhat).statistic)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(r2_score(y, yhat))


def regression_metrics(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray) -> dict:
    """MAE, RMSE, Spearman, R² (pooled out-of-fold) and per-group MAE."""
    per = pd.DataFrame({"g": groups, "e": np.abs(np.asarray(y) - np.asarray(yhat))})
    return {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "spearman": spearman(y, yhat),
        "r2": r2(y, yhat),
        "per_group_mae": {str(k): float(v) for k, v in per.groupby("g")["e"].mean().items()},
    }


def balanced_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean per-class recall over the classes present in ``y`` (bootstrap resamples may lack
    some classes; sklearn's warning about that is expected and silenced)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(balanced_accuracy_score(y, yhat))


def macro_f1(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(f1_score(y, yhat, average="macro", zero_division=0))


def brier_score(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    """Multi-class Brier score: mean over rows of sum_k (p_k - 1[y = k])²."""
    onehot = (np.asarray(y)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(
    y: np.ndarray, proba: np.ndarray, classes: list[str], n_bins: int
) -> float:
    """Top-label ECE: |accuracy - confidence| weighted by bin size over ``n_bins`` equal-width
    confidence bins."""
    conf = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y)).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(
    y: np.ndarray, yhat: np.ndarray, proba: np.ndarray | None, classes: list[str], n_bins: int
) -> dict:
    """Balanced accuracy, macro-F1, per-class precision/recall, confusion matrix (rows = true,
    columns = predicted, order ``classes``) and, with probabilities, Brier score and ECE."""
    p, r, _, n = precision_recall_fscore_support(y, yhat, labels=classes, zero_division=0)
    out = {
        "balanced_accuracy": balanced_accuracy(y, yhat),
        "macro_f1": macro_f1(y, yhat),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "support": int(n[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y, yhat, labels=classes).tolist(),
        "classes": list(classes),
    }
    if proba is not None:
        out["brier"] = brier_score(y, proba, classes)
        out["ece"] = expected_calibration_error(y, proba, classes, n_bins)
    return out
