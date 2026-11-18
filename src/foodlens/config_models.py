"""Config sections of the P6 model zoo and training (`models`, `train` in ``default.yaml``).

Kept apart from ``config.py`` to respect the module size limit; ``config.Config`` includes them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

Grid = dict[str, list[int | float | str | None]]  # Pipeline parameter -> candidate values


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelGrid(_Section):
    """Fixed default parameters and the (small) tuning grid of one model, per task."""

    regression: Grid = Field(default_factory=dict)
    classification: Grid = Field(default_factory=dict)
    defaults_regression: dict[str, int | float | str | None] = Field(default_factory=dict)
    defaults_classification: dict[str, int | float | str | None] = Field(default_factory=dict)


class ModelsConfig(_Section):
    zoo: list[str] = Field(min_length=1)  # model names, simplest first (tie-break order)
    variance_threshold: float = Field(ge=0)  # drop near-constant columns (variance <= this)
    max_grid_combos: int = Field(ge=1)  # guard: phase rule "<= 12 combos"
    linear_max_iter: int = Field(ge=100)
    rf_n_estimators: int = Field(ge=10)
    svc_probability: bool  # SVC needs Platt probabilities for Brier / ECE
    grids: dict[str, ModelGrid]


class SelectionConfig(_Section):
    """Final-model selection rule, fixed BEFORE looking at the P6 results (P6.T5)."""

    metric: str  # headline classification metric
    scheme: str  # CV scheme the rule is applied on
    feature_set: str  # candidate feature set(s) are the model-zoo runs on this set
    baseline_models: list[str]  # the CI lower bound must beat the best of these
    tie_tol: float = Field(ge=0)  # within this of the best = tie -> simpler model (zoo order)


class TrainConfig(_Section):
    experiments_dir: str  # experiment YAMLs, relative to the repo root
    n_permutations: int = Field(ge=1)  # grouped permutation reps for TUNED models (full nested CV)
    calibration_method: str  # CalibratedClassifierCV method
    ood_percentile: float = Field(gt=0, lt=100)  # OOD threshold = this percentile of train dist.
    importance_repeats: int = Field(ge=1)  # permutation-importance shuffles per fold
    importance_top_n: int = Field(ge=1)  # features shown in F14
    pdp_top_k: int = Field(ge=1)  # partial-dependence plots for the top-k features
    pdp_grid: int = Field(ge=5)  # grid points per partial-dependence curve
    final_model_name: str
    selection: SelectionConfig
