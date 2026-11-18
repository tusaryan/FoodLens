"""Out-of-distribution (OOD) check (P6.T3): Mahalanobis distance on the scaled features.

Fitted on the TRAINING fold only: median imputation, scaling, a Ledoit-Wolf shrunk covariance
(the ~150 features are strongly correlated, so the plain covariance is singular) and a threshold
at the ``train.ood_percentile`` (99th) percentile of reference distances:
- ``mode="grouped"`` (default, S09): distances of each training group from a detector fitted on
  the OTHER training groups (leave-one-group-out inside the training fold). This answers "how far
  does an unseen piece from the same setup usually sit?". The phase's plain rule (``"in_sample"``,
  distances of the rows the covariance was fitted on) under-estimates that distance in ~150
  dimensions and flagged 100 % of held-out pieces in the S09 smoke run; it is still reported.
A capture whose distance exceeds the threshold is "unlike anything seen in training": the model
abstains instead of guessing. Abstention rate and accuracy on the kept rows are ALWAYS
reported together (abstaining on hard rows inflates the kept-row accuracy).
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foodlens.errors import ConfigError, SplitError


class MahalanobisOOD:
    """``fit(x_train)`` -> ``distance(x)``, ``is_ood(x)`` (distance > ``threshold_``)."""

    def __init__(self, percentile: float, variance_threshold: float = 0.0, mode: str = "grouped"):
        self.percentile = percentile
        self.variance_threshold = variance_threshold
        self.mode = mode

    def _fit_core(self, x: np.ndarray) -> MahalanobisOOD:
        self.prep_ = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("drop_const", VarianceThreshold(self.variance_threshold)),
            ("scale", StandardScaler()),
        ]).fit(x)  # fmt: skip
        self.cov_ = LedoitWolf().fit(self.prep_.transform(x))
        return self

    def fit(self, x: np.ndarray, groups: np.ndarray | None = None) -> MahalanobisOOD:
        """Fit on ``x``; ``groups`` (>= 2 distinct) are needed for ``mode="grouped"``."""
        self._fit_core(x)
        ref = self.distance(x)
        if self.mode == "grouped":
            if groups is None or len(np.unique(groups)) < 2:
                raise SplitError("MahalanobisOOD(mode='grouped') needs >= 2 training groups")
            groups = np.asarray(groups)
            ref = np.empty(len(x))
            for g in np.unique(groups):
                m = groups == g
                ref[m] = MahalanobisOOD(self.percentile, self.variance_threshold)._fit_core(
                    x[~m]).distance(x[m])  # fmt: skip
        elif self.mode != "in_sample":
            raise ConfigError(f"unknown OOD mode {self.mode!r}")
        self.threshold_ = float(np.percentile(ref, self.percentile))
        return self

    def distance(self, x: np.ndarray) -> np.ndarray:
        """Mahalanobis distance (in units of the shrunk training covariance), >= 0."""
        z = self.prep_.transform(x)
        return np.sqrt(np.maximum(self.cov_.mahalanobis(z), 0.0))

    def is_ood(self, x: np.ndarray) -> np.ndarray:
        return self.distance(x) > self.threshold_


def abstention_summary(y: np.ndarray, yhat: np.ndarray, ood: np.ndarray) -> dict:
    """Abstention rate and accuracy on kept / abstained rows (NaN if a part is empty)."""
    y, yhat, ood = np.asarray(y), np.asarray(yhat), np.asarray(ood, dtype=bool)
    kept = ~ood

    def acc(m: np.ndarray) -> float:
        return float(np.mean(y[m] == yhat[m])) if m.any() else float("nan")

    return {"n": int(len(y)), "abstention_rate": float(ood.mean()), "n_kept": int(kept.sum()),
            "accuracy_all": acc(np.ones_like(kept)), "accuracy_kept": acc(kept),
            "accuracy_abstained": acc(ood)}  # fmt: skip
