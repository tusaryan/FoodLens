"""Configuration loader: ``configs/default.yaml`` -> validated pydantic ``Config``.

All paths, thresholds and parameters come from here (no magic numbers in code). Relative paths in
the YAML are relative to the repository root (the parent of the ``configs/`` folder).
Any invalid or unknown value raises ``ConfigError`` (code ``E_CONFIG``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from foodlens.errors import ConfigError
from foodlens.samples_config import SamplesConfig, SessionSpec, TempSpec

__all__ = ["Config", "SamplesConfig", "SessionSpec", "TempSpec", "load_config", "load_samples"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
HASH_LEN = 10


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Section):
    raw_dir: str
    data_dir: str
    reports_dir: str
    models_dir: str
    samples_file: str


class LocationConfig(_Section):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    tz: str


class QCConfig(_Section):
    dark_p95: float = Field(ge=0, le=255)
    min_signal: float = Field(ge=0, le=255)
    dominance_ratio: float = Field(gt=0, le=1)
    scene_z: float = Field(gt=0)
    blur_div: float = Field(gt=0)
    scene_window: int = Field(ge=3)
    scene_floor_log_area: float = Field(gt=0)
    scene_floor_px: float = Field(gt=0)
    scene_floor_log_mean: float = Field(gt=0)
    step_k: int = Field(ge=1)
    step_min_log: float = Field(gt=0)
    step_merge_s: float = Field(ge=0)


class ROIConfig(_Section):
    blur_sigma: float = Field(gt=0)
    otsu_classes: int = Field(ge=2, le=5)
    min_level: float = Field(ge=0, le=255)
    ref_lights: list[Literal["RED", "GREEN", "BLUE"]] = Field(min_length=1)
    close_r: int = Field(ge=0)
    open_r: int = Field(ge=0)
    min_frac: float = Field(gt=0, lt=1)
    max_frac: float = Field(gt=0, le=1)
    erode_px: int = Field(ge=0)
    min_iou: float = Field(ge=0, le=1)
    foreign_max_blue_red: float = Field(gt=0)
    foreign_min_area_px: int = Field(ge=1)
    foreign_max_frames: int = Field(ge=1)

    @model_validator(mode="after")
    def _frac_order(self) -> ROIConfig:
        if self.min_frac >= self.max_frac:
            raise ValueError("roi.min_frac must be < roi.max_frac")
        return self


class GlintConfig(_Section):
    tophat_r: int = Field(gt=0)
    k: float = Field(gt=0)
    abs_min: float = Field(ge=0, le=255)
    min_area_px: int = Field(ge=1)
    match_px: float = Field(gt=0)
    merge_area_ratio: float = Field(gt=1)
    sat_level: int = Field(ge=1, le=255)
    rel_k: float = Field(gt=0)
    rel_floor: float = Field(ge=0, le=255)


class VisionConfig(_Section):
    work_long_side: int = Field(gt=0)
    glint_long_side: int = Field(gt=0)
    glcm_distances: list[int] = Field(min_length=1)
    glcm_levels: int = Field(ge=2, le=256)
    glcm_clip_pct: list[float] = Field(min_length=2, max_length=2)
    lbp_p: int = Field(gt=0)
    lbp_r: int = Field(gt=0)


class CaptureConfig(_Section):
    light_order: list[Literal["RED", "GREEN", "BLUE"]] = Field(min_length=1)
    image_exts: list[str] = Field(min_length=1)
    ignore_dir_substr: str
    experiment_pattern: str
    session_pattern: str
    filename_time_pattern: str
    filename_time_format: str


class TemperatureConfig(_Section):
    archive_url: str
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)
    cache_subdir: str


class QAConfig(_Section):
    positions: list[float] = Field(min_length=1)
    crop_margin: float = Field(ge=0)
    marker_px: float = Field(gt=0)
    dpi: int = Field(ge=50)
    crop_pad_px: int = Field(ge=0)  # meat crops: margin around the full-meat bbox (work res px)
    crop_sample: int = Field(ge=1)  # meat crops: frames in the visual-review contact sheet


class EDAConfig(_Section):
    shadow_min_level: float = Field(ge=0, le=255)
    shadow_open_r: int = Field(ge=0)
    positions: list[float] = Field(min_length=2)
    stretch_pct: list[float] = Field(min_length=2, max_length=2)
    diff_pct: float = Field(gt=0, le=100)
    rel_base_n: int = Field(ge=1)
    rolling_window: int = Field(ge=1)
    min_frames: int = Field(ge=3)
    alpha: float = Field(gt=0, lt=1)
    window_frac: float = Field(gt=0, le=0.5)
    promising_min_sessions: int = Field(ge=1)
    sample_outliers: int = Field(ge=0)
    outlier_floor: float = Field(gt=0)
    ref_gap_px: int = Field(ge=0)
    ref_min_level: float = Field(ge=0, le=255)


class FeaturesConfig(_Section):
    roi_neck_r: int = Field(ge=0)
    glint_edge_px: int = Field(ge=0)
    blur_nan_groups: list[str]
    ratio_sources: list[str] = Field(min_length=1)
    log_ratio_sources: list[str]
    corr_max: float = Field(gt=0, le=1)


class TripletConfig(_Section):
    max_span_s: float = Field(gt=0)


class CVConfig(_Section):
    group_col_preference: list[str] = Field(min_length=1)
    inner_splits: int = Field(ge=2)
    min_groups_for_tuning: int = Field(ge=2)
    temporal_train_frac: float = Field(gt=0, lt=1)
    n_bootstrap: int = Field(ge=1)
    n_permutations: int = Field(ge=1)
    ci_level: float = Field(gt=0, lt=1)
    ece_bins: int = Field(ge=1)


class LabelsConfig(_Section):
    class_definition: Literal["A", "B"]
    classes: list[str] = Field(min_length=2)
    equiv_h_thresholds: list[float]

    @model_validator(mode="after")
    def _thresholds(self) -> LabelsConfig:
        t = self.equiv_h_thresholds
        if len(t) != len(self.classes) - 1 or any(a >= b for a, b in zip(t, t[1:], strict=False)):
            raise ValueError(
                "labels.equiv_h_thresholds must be increasing, len(classes) - 1 values"
            )
        return self


class ThermalConfig(_Section):
    T_min: float
    T_min_sensitivity: float
    T_ref: float
    indoor_offset_c: float

    @model_validator(mode="after")
    def _order(self) -> ThermalConfig:
        if self.T_min >= self.T_ref:
            raise ValueError("thermal.T_min must be < thermal.T_ref")
        return self


class EvaluationConfig(_Section):
    targets: list[str] = Field(min_length=1)  # regression label columns (labels.labels)
    class_target: str  # classification label column
    group_cols: list[str] = Field(min_length=1)  # CV groupings to report (first = headline)
    baseline_single: str  # one-feature linear baseline (ROI median of RED)
    baseline_glint: str  # the old method: spot (glint) count density, absolute floor
    baseline_first_frame: str  # mentor's first-frame-normalised intensity (needs session start)
    leaky_folds: int = Field(ge=2)
    rf_n_estimators: int = Field(ge=10)
    rf_min_samples_leaf: int = Field(ge=1)
    complete_only: bool  # evaluate complete R/G/B triplets only
    drop_feature_prefixes: list[str]  # model-excluded feature prefixes (setup proxies)
    step_sensitivity_sessions: list[str]  # post-step triplets of these = sensitivity case


class Config(_Section):
    """Full validated configuration. ``root`` is where relative paths are resolved from."""

    paths: PathsConfig
    seed: int
    n_jobs: int
    location: LocationConfig
    capture: CaptureConfig
    temperature: TemperatureConfig
    qc: QCConfig
    roi: ROIConfig
    glint: GlintConfig
    vision: VisionConfig
    qa: QAConfig
    eda: EDAConfig
    features: FeaturesConfig
    triplet: TripletConfig
    cv: CVConfig
    labels: LabelsConfig
    thermal: ThermalConfig
    evaluation: EvaluationConfig
    root: Path = Field(default=REPO_ROOT, exclude=True)

    def path(self, key: str) -> Path:
        """Absolute path for a ``paths`` entry, e.g. ``cfg.path("data_dir")``."""
        if key not in PathsConfig.model_fields:
            raise ConfigError(f"unknown paths key {key!r}")
        p = Path(getattr(self.paths, key))
        return p if p.is_absolute() else self.root / p


# ---- loading ----
def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _validate[M: BaseModel](model: type[M], data: dict, source: Path) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{source}: {exc}") from exc


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the main config (default: ``configs/default.yaml``)."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data = _read_yaml(p)
    data["root"] = p.resolve().parent.parent
    return _validate(Config, data, p)


def load_samples(cfg: Config, path: str | Path | None = None) -> SamplesConfig:
    """Load and validate ``samples.yaml`` (default: ``cfg.paths.samples_file``)."""
    p = Path(path) if path is not None else cfg.path("samples_file")
    return _validate(SamplesConfig, _read_yaml(p), p)


def config_hash(cfg: Config, sections: list[str] | None = None) -> str:
    """SHA-1 (first 10 hex chars) of the canonical JSON of the chosen sections (default: all).

    Used in cache/output paths so a stage re-runs when its parameters change.
    """
    dumped = cfg.model_dump(mode="json")
    if sections is not None:
        unknown = [s for s in sections if s not in dumped]
        if unknown:
            raise ConfigError(f"unknown config sections for hashing: {unknown}")
        dumped = {s: dumped[s] for s in sections}
    canonical = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:HASH_LEN]
