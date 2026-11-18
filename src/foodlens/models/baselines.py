"""Baselines (P5.T4). Every model is a fixed-config sklearn ``Pipeline`` (imputer and scaler are
fitted inside each training fold), so nothing is tuned and nothing leaks.

- ``baseline_mean``: predict the training mean / majority class (features ignored).
- ``baseline_red_median``: linear / logistic model on the ROI median of RED only.
- ``baseline_glint_count``: the old method, spot (glint) count density with the absolute floor.
- ``baseline_first_frame``: the mentor's first-frame-normalised intensity (needs the session's
  first photo, so NOT usable for a single photo of unknown history; science reference only).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foodlens.config import Config

LOGREG_MAX_ITER = 2000  # iterations for the lbfgs solver; convergence, not a tuned value


@dataclass(frozen=True)
class ModelSpec:
    """A named model: its feature columns and a factory ``task -> unfitted Pipeline``."""

    name: str
    features: list[str]
    make: Callable[[str], Pipeline]
    single_shot_ok: bool
    description: str


def _linear(task: str) -> Pipeline:
    model = (
        LinearRegression()
        if task == "regression"
        else LogisticRegression(max_iter=LOGREG_MAX_ITER, class_weight="balanced")
    )
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return Pipeline([*steps, ("model", model)])


def _dummy(task: str) -> Pipeline:
    model = (
        DummyRegressor(strategy="mean")
        if task == "regression"
        else DummyClassifier(strategy="most_frequent")
    )
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


def baseline_specs(cfg: Config) -> dict[str, ModelSpec]:
    """All baselines by name (feature names from ``cfg.evaluation``)."""
    ev = cfg.evaluation
    specs = [
        ModelSpec("baseline_mean", [ev.baseline_single], _dummy, True,
                  "training mean / majority class"),
        ModelSpec("baseline_red_median", [ev.baseline_single], _linear, True,
                  f"linear on {ev.baseline_single} only"),
        ModelSpec("baseline_glint_count", [ev.baseline_glint], _linear, True,
                  f"old method: linear on {ev.baseline_glint} only"),
        ModelSpec("baseline_first_frame", [ev.baseline_first_frame], _linear, False,
                  f"mentor: linear on {ev.baseline_first_frame} (needs the first photo)"),
    ]  # fmt: skip
    return {s.name: s for s in specs}
