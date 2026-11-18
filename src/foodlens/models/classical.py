"""Classical model zoo (P6.T1): sklearn Pipelines with nested grouped tuning.

Every model = ``Pipeline(impute median -> drop near-constant -> scale -> top-k select -> model)``
(all fitted inside the training fold). ``GroupTunedModel`` wraps it: on ``fit(X, y, groups)`` it
tunes the small config grid with an inner GroupKFold over the TRAINING groups only
(``evaluation.splits.inner_group_kfold``); with fewer than ``cv.min_groups_for_tuning`` training
groups it fits the fixed default config instead. The test fold is never seen while tuning.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectorMixin, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from foodlens.config import Config
from foodlens.errors import ConfigError
from foodlens.evaluation import metrics as M
from foodlens.evaluation.splits import inner_group_kfold
from foodlens.models.baselines import ModelSpec

# inner-CV scoring = the outer headline metrics (project balanced accuracy silences the expected
# "class missing from an inner fold" warning; values are identical to sklearn's scorer)
SCORING = {
    "regression": "neg_mean_absolute_error",
    "classification": make_scorer(M.balanced_accuracy),
}


class TopKSelector(SelectorMixin, BaseEstimator):
    """Keep the ``k`` columns with the highest univariate F score (``"all"`` = keep every column;
    ``k`` larger than the column count keeps all). Fitted on the training fold only."""

    def __init__(self, k: int | str = "all", task: str = "regression"):
        self.k = k
        self.task = task

    def fit(self, x: np.ndarray, y: np.ndarray) -> TopKSelector:
        n = x.shape[1]
        k = n if self.k in ("all", None) else min(int(self.k), n)
        fn = f_regression if self.task == "regression" else f_classif
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.nan_to_num(fn(x, y)[0], nan=0.0, posinf=0.0)
        keep = np.argsort(-scores, kind="stable")[:k]
        self.mask_ = np.zeros(n, dtype=bool)
        self.mask_[keep] = True
        return self

    def _get_support_mask(self) -> np.ndarray:
        return self.mask_


def _estimator(name: str, task: str, cfg: Config):
    m, seed = cfg.models, cfg.seed
    reg = task == "regression"
    if name == "linear":
        if reg:
            return Ridge()
        return LogisticRegression(max_iter=m.linear_max_iter, class_weight="balanced")
    if name == "rf":
        cls = RandomForestRegressor if reg else RandomForestClassifier
        kw = {} if reg else {"class_weight": "balanced"}
        return cls(n_estimators=m.rf_n_estimators, random_state=seed, n_jobs=1, **kw)
    if name == "hgb":
        if reg:
            return HistGradientBoostingRegressor(random_state=seed)
        return HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    if name == "svm":
        if reg:
            return SVR(kernel="rbf")
        return SVC(kernel="rbf", class_weight="balanced", probability=m.svc_probability,
                   random_state=seed)  # fmt: skip
    raise ConfigError(f"unknown model {name!r} (models.zoo)")


def build_pipeline(name: str, task: str, cfg: Config) -> Pipeline:
    """Untuned pipeline of model ``name`` with its fixed default parameters from config."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("drop_const", VarianceThreshold(cfg.models.variance_threshold)),
        ("scale", StandardScaler()),
        ("select", TopKSelector(task=task)),
        ("model", _estimator(name, task, cfg)),
    ])  # fmt: skip
    return pipe.set_params(**model_defaults(name, task, cfg))


def model_defaults(name: str, task: str, cfg: Config) -> dict:
    g = cfg.models.grids.get(name)
    if g is None:
        raise ConfigError(f"models.grids has no entry for {name!r}")
    return dict(g.defaults_regression if task == "regression" else g.defaults_classification)


def model_grid(name: str, task: str, cfg: Config) -> dict[str, list]:
    """The tuning grid; raises ``ConfigError`` above ``models.max_grid_combos`` combinations."""
    grid = dict(getattr(cfg.models.grids[name], task))
    n = int(np.prod([len(v) for v in grid.values()])) if grid else 1
    if n > cfg.models.max_grid_combos:
        raise ConfigError(f"grid {name}/{task} has {n} combos > {cfg.models.max_grid_combos}")
    return grid


class GroupTunedModel(BaseEstimator):
    """Pipeline + nested grouped grid search. ``fit`` needs ``groups`` (the training rows' CV
    groups); ``evaluation.cv.oof_predict`` passes them because ``uses_groups`` is True."""

    uses_groups = True

    def __init__(self, base, grid, scoring, inner_splits, min_groups, n_jobs):
        self.base, self.grid, self.scoring = base, grid, scoring
        self.inner_splits, self.min_groups, self.n_jobs = inner_splits, min_groups, n_jobs

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        inner = None
        if groups is not None and self.grid:
            inner = inner_group_kfold(np.asarray(groups), self.inner_splits, self.min_groups)
        if inner is None:  # < min_groups training groups (or no grid): fixed default config
            self.best_estimator_, self.best_params_, self.tuned_ = clone(self.base), {}, False
            self.best_estimator_.fit(x, y)
        else:
            gs = GridSearchCV(clone(self.base), self.grid, scoring=self.scoring, cv=inner,
                              n_jobs=self.n_jobs, refit=True, error_score="raise")  # fmt: skip
            gs.fit(x, y)
            self.best_estimator_, self.best_params_ = gs.best_estimator_, gs.best_params_
            self.tuned_ = True
        est = self.best_estimator_.steps[-1][1]
        if "n_jobs" in est.get_params():  # fitted single-threaded inside the grid; predict on all
            est.set_params(n_jobs=self.n_jobs)
        if hasattr(self.best_estimator_, "classes_"):
            self.classes_ = self.best_estimator_.classes_
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.best_estimator_.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.best_estimator_.predict_proba(x)


def tuned_factory(name: str, cfg: Config):
    """``task -> GroupTunedModel`` for model ``name`` (the ``ModelSpec.make`` of the zoo)."""

    def make(task: str) -> GroupTunedModel:
        return GroupTunedModel(
            build_pipeline(name, task, cfg), model_grid(name, task, cfg), SCORING[task],
            cfg.cv.inner_splits, cfg.cv.min_groups_for_tuning, cfg.n_jobs,
        )  # fmt: skip

    return make


def classical_specs(cfg: Config, feature_sets: dict[str, list[str]],
                    models: list[str] | None = None) -> list[ModelSpec]:  # fmt: skip
    """One ModelSpec per (model, feature set), named ``<model>__<feature set>``."""
    out = []
    for m, (fs, cols) in itertools.product(models or cfg.models.zoo, feature_sets.items()):
        desc = f"{m} (nested grouped tuning) on feature set {fs} ({len(cols)} columns)"
        single = not any(c.endswith("__rel") for c in cols)  # __rel needs the session start
        out.append(ModelSpec(f"{m}__{fs}", cols, tuned_factory(m, cfg), single, desc))
    return out
