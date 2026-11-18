"""Calibration and abstention of the selected classifier (P6.T3).

``GroupCalibratedModel``: nested grouped tuning (``GroupTunedModel``) picks the parameters on
the training groups, then ``CalibratedClassifierCV`` (``train.calibration_method``, sigmoid)
refits that config on inner GroupKFold folds of the SAME training groups and learns the
probability mapping on their held-out inner groups. The outer test piece is never used.
``evaluate_calibration`` scores the uncalibrated vs calibrated model under the same outer CV
(ECE / Brier before vs after, reliability diagram F13). The final model uses the calibrated
classifier only if its out-of-fold ECE is lower (S09 rule; calibration can hurt with few groups).
OOD abstention is scored on the chosen kind, with the grouped and the in-sample threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV

from foodlens.config import Config
from foodlens.evaluation import metrics as M
from foodlens.evaluation.cv import evaluate
from foodlens.evaluation.splits import inner_group_kfold, logo_splits
from foodlens.inference.ood import MahalanobisOOD, abstention_summary
from foodlens.models.classical import tuned_factory


class GroupCalibratedModel(BaseEstimator):
    """Tuned pipeline + grouped probability calibration (``fit(X, y, groups)``)."""

    uses_groups = True

    def __init__(self, tuned, method, inner_splits, min_groups):
        self.tuned, self.method = tuned, method
        self.inner_splits, self.min_groups = inner_splits, min_groups

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
        tuned = clone(self.tuned).fit(x, y, groups=groups)
        self.best_params_ = tuned.best_params_
        inner = inner_group_kfold(np.asarray(groups), self.inner_splits, self.min_groups)
        if inner is None:  # too few groups to calibrate honestly: keep the uncalibrated model
            self.model_, self.calibrated_ = tuned.best_estimator_, False
        else:
            base = clone(tuned.base).set_params(**tuned.best_params_)
            self.model_ = CalibratedClassifierCV(base, method=self.method, cv=inner).fit(x, y)
            self.calibrated_ = True
        self.classes_ = self.model_.classes_
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(x)


def calibrated_factory(name: str, cfg: Config):
    tuned = tuned_factory(name, cfg)

    def make(task: str) -> GroupCalibratedModel:
        return GroupCalibratedModel(tuned(task), cfg.train.calibration_method,
                                    cfg.cv.inner_splits, cfg.cv.min_groups_for_tuning)  # fmt: skip

    return make


def reliability_bins(oof: pd.DataFrame, classes: list[str], n_bins: int) -> pd.DataFrame:
    """Top-label reliability table: per confidence bin, mean confidence, accuracy, count."""
    p = oof[[f"p_{c}" for c in classes]].to_numpy()
    conf = p.max(axis=1)
    correct = np.asarray(classes)[p.argmax(axis=1)] == oof["y"].to_numpy(str)
    b = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    t = pd.DataFrame({"bin": b, "conf": conf, "correct": correct})
    return t.groupby("bin").agg(confidence=("conf", "mean"), accuracy=("correct", "mean"),
                                n=("conf", "size")).reset_index()  # fmt: skip


def ood_by_fold(df: pd.DataFrame, features: list[str], groups: np.ndarray, cfg: Config,
                mode: str) -> np.ndarray:  # fmt: skip
    """OOD flag of every row, from a detector fitted on the OTHER groups (LOGO folds); the
    grouped threshold uses only those training groups (see ``inference.ood``)."""
    x = df[features].to_numpy(dtype=float)
    flag = np.zeros(len(df), dtype=bool)
    for tr, te in logo_splits(groups):
        det = MahalanobisOOD(cfg.train.ood_percentile, cfg.models.variance_threshold, mode)
        flag[te] = det.fit(x[tr], groups[tr]).is_ood(x[te])
    return flag


def _abstention(oof: pd.DataFrame, flag: np.ndarray, mode: str) -> tuple[dict, pd.DataFrame]:
    y, yhat, grp = oof["y"].to_numpy(str), oof["yhat"].to_numpy(str), oof["group"].to_numpy()
    summ = {"threshold_mode": mode, **abstention_summary(y, yhat, flag)}
    kept = ~flag
    summ["balanced_accuracy_all"] = M.balanced_accuracy(y, yhat)
    bal_kept = M.balanced_accuracy(y[kept], yhat[kept]) if kept.any() else np.nan
    summ["balanced_accuracy_kept"] = bal_kept
    per = [{"threshold_mode": mode, "group": g, **abstention_summary(y[m], yhat[m], flag[m])}
           for g in np.unique(grp) for m in [grp == g]]  # fmt: skip
    return summ, pd.DataFrame(per)


def swap_lights(df: pd.DataFrame, features: list[str], a: str = "RED", b: str = "BLUE"):
    """Copy of ``df`` with the ``__<a>`` and ``__<b>`` columns exchanged (cross-light ratios
    keep their values): a simulated wrong-LED capture, used to check the OOD detector reacts."""
    out = df.copy()
    for f in features:
        if f.endswith(f"__{a}") and f[: -len(a)] + b in df.columns:
            g = f[: -len(a)] + b
            out[f], out[g] = df[g].to_numpy(), df[f].to_numpy()
    return out


def ood_shift_rate(df: pd.DataFrame, features: list[str], groups: np.ndarray, cfg: Config) -> float:
    """Share of held-out triplets flagged (grouped threshold) after ``swap_lights``."""
    x = df[features].to_numpy(dtype=float)
    xs = swap_lights(df, features)[features].to_numpy(dtype=float)
    flag = np.zeros(len(df), dtype=bool)
    for tr, te in logo_splits(groups):
        det = MahalanobisOOD(cfg.train.ood_percentile, cfg.models.variance_threshold)
        flag[te] = det.fit(x[tr], groups[tr]).is_ood(xs[te])
    return float(flag.mean())


def evaluate_calibration(name: str, df: pd.DataFrame, features: list[str], groups: np.ndarray,
                         cfg: Config) -> dict:  # fmt: skip
    """Uncalibrated vs calibrated classifier ``name`` under LOGO on ``groups``, plus OOD
    abstention of the calibrated one. Returns metrics, reliability tables and the oof frames."""
    target, classes = cfg.evaluation.class_target, cfg.labels.classes
    splits = logo_splits(groups)
    out: dict = {"metrics": {}, "reliability": {}, "oof": {}}
    for kind, make in (("uncalibrated", tuned_factory(name, cfg)),
                       ("calibrated", calibrated_factory(name, cfg))):  # fmt: skip
        res = evaluate(make, "classification", df, features, target, groups, splits, cfg, False)
        out["metrics"][kind] = {k: res["metrics"][k] for k in
                                ("balanced_accuracy", "macro_f1", "brier", "ece")}  # fmt: skip
        out["metrics"][kind]["ci"] = {k: list(v) for k, v in res["ci"].items()}
        out["reliability"][kind] = reliability_bins(res["oof"], classes, cfg.cv.ece_bins)
        out["oof"][kind] = res["oof"]
    best = min(out["metrics"], key=lambda k: out["metrics"][k]["ece"])
    out["use_calibrated"] = best == "calibrated"  # final model: the kind with the lower OOF ECE
    oof = out["oof"][best]
    parts = [_abstention(oof, ood_by_fold(df, features, groups, cfg, mode)[oof["row"].to_numpy()],
                         mode) for mode in ("grouped", "in_sample")]  # fmt: skip
    out["ood"] = {"overall": [p[0] for p in parts],
                  "per_group": pd.concat([p[1] for p in parts], ignore_index=True)}  # fmt: skip
    out["ood"]["overall"][0]["shift_check_flag_rate_RB_swapped"] = ood_shift_rate(
        df, features, groups, cfg)  # fmt: skip
    return out
