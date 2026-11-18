"""Feature schema: every column of the frame and triplet feature tables, with units and meaning.

- Frame level (``features_frame.parquet``): the ``specs.base_rows`` features, light = ``EACH``
  (computed for whichever light the frame was taken under).
- Triplet level (``features_triplet.parquet``): ``<feature>__<LIGHT>`` for each light, the
  cross-light ratios ``<src>__ratio_<A><B>`` / ``<src>__logratio_<A><B>``
  (``features.ratio_sources``),
  and the session-relative variants ``<col>__rel`` (group ``relative``, ``single_shot_ok`` False:
  they divide by the session's first complete triplet, so they need the start of the session and
  are for EDA / the "session-normalised" experiment only).
Meta columns (ids, times, QC, flags) are listed separately and are NEVER model inputs.
Banned as features (image-pipeline skill): timestamp, file index, session id, orientation, image
size, ROI centroid / bbox / area.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from foodlens.config import Config
from foodlens.features.specs import GREY, UNITLESS, base_rows

EACH = "EACH"
GROUPS = ("intensity", "glint", "texture", "ratio", "relative")
RATIO_PAIRS = [("RED", "GREEN"), ("GREEN", "BLUE"), ("RED", "BLUE")]
SEP = "__"

FRAME_META = [
    "frame_id", "session_id", "experiment", "piece_id", "light", "ts", "elapsed_s", "elapsed_h",
    "temp_c", "scene_segment", "qc_status", "qc_reasons", "ref_mean", "roi_gap_s", "step_gain",
    "blur_nan",
]  # fmt: skip
TRIPLET_META_BASE = [
    "session_id", "experiment", "piece_id", "triplet", "ts", "elapsed_s", "elapsed_h", "temp_c",
    "scene_segment", "n_lights", "is_complete", "blur_nan_any",
]  # fmt: skip
SPECTRAL = (
    "spectral shape of the reflectance (the R, G, B LEDs probe different absorption bands of "
    "myoglobin forms); a brightness change common to all lights cancels"
)


@dataclass(frozen=True)
class FeatureSpec:
    """One feature column. ``light``: RED / GREEN / BLUE, a pair ``RED/GREEN``, or ``EACH``."""

    name: str
    light: str
    group: str
    unit: str
    description: str
    physical_meaning: str
    single_shot_ok: bool
    level: str = "frame"


def frame_specs(cfg: Config) -> list[FeatureSpec]:
    """Frame-level features (light = EACH), in output column order."""
    return [
        FeatureSpec(n, EACH, g, u, d, m, single_shot_ok=True)
        for n, g, u, d, m in base_rows(cfg.vision.lbp_p)
    ]


def multiplicative_features(cfg: Config) -> list[str]:
    """Frame features that scale with the LED / camera gain (intensity group, grey levels):
    the ones divided by the scene-step gain."""
    return [s.name for s in frame_specs(cfg) if s.group == "intensity" and s.unit == GREY]


def group_features(cfg: Config, groups: list[str]) -> list[str]:
    """Frame feature names in ``groups``."""
    return [s.name for s in frame_specs(cfg) if s.group in groups]


def ratio_names(src: str, kind: str = "ratio") -> list[str]:
    """``<src>__<kind>_<A><B>`` for RATIO_PAIRS (``kind``: ratio or logratio)."""
    return [f"{src}{SEP}{kind}_{a[0]}{b[0]}" for a, b in RATIO_PAIRS]


def _per_light(cfg: Config) -> list[FeatureSpec]:
    out = []
    for s in frame_specs(cfg):
        for light in cfg.capture.light_order:
            out.append(
                FeatureSpec(
                    name=f"{s.name}{SEP}{light}",
                    light=light,
                    group=s.group,
                    unit=s.unit,
                    description=f"{s.description} ({light} frame of the triplet)",
                    physical_meaning=s.physical_meaning,
                    single_shot_ok=s.single_shot_ok,
                    level="triplet",
                )
            )
    return out


def _cross_light(cfg: Config) -> list[FeatureSpec]:
    out = []
    kinds = [("ratio", cfg.features.ratio_sources), ("logratio", cfg.features.log_ratio_sources)]
    for kind, srcs in kinds:
        for src in srcs:
            for name, (a, b) in zip(ratio_names(src, kind), RATIO_PAIRS, strict=True):
                op = f"{src}({a}) / {src}({b})"
                desc = (op if kind == "ratio" else f"ln[{op}]") + " within one triplet"
                light = f"{a}/{b}"
                out.append(
                    FeatureSpec(name, light, "ratio", UNITLESS, desc, SPECTRAL, True, "triplet")
                )
    return out


def relative_sources(cfg: Config) -> list[FeatureSpec]:
    """Triplet columns that get a session-relative ``__rel`` variant: every per-light feature
    except the LBP histogram bins, and the cross-light ratios (not the log-ratios)."""
    bins = {f"lbp_{i}" for i in range(cfg.vision.lbp_p + 2)}
    per = [s for s in _per_light(cfg) if s.name.split(SEP)[0] not in bins]
    return per + [s for s in _cross_light(cfg) if f"{SEP}ratio_" in s.name]


def triplet_specs(cfg: Config) -> list[FeatureSpec]:
    """Triplet-level features in output column order."""
    rel = [
        FeatureSpec(
            name=f"{s.name}{SEP}rel",
            light=s.light,
            group="relative",
            unit=f"ratio to the session's first complete triplet ({s.unit} / same)",
            description=f"{s.description}, divided by its first-complete-triplet value",
            physical_meaning=s.physical_meaning + "; change since the start of the session",
            single_shot_ok=False,
            level="triplet",
        )
        for s in relative_sources(cfg)
    ]
    return _per_light(cfg) + _cross_light(cfg) + rel


def triplet_meta(cfg: Config) -> list[str]:
    """Meta columns of the triplet table (flags and per-light frame ids included)."""
    lights = cfg.capture.light_order
    return (
        TRIPLET_META_BASE + [f"has_{li}" for li in lights] + [f"frame_id{SEP}{li}" for li in lights]
    )


def export_feature_dictionary(cfg: Config, path: Path) -> pd.DataFrame:
    """Write the feature dictionary (report table T5) as CSV; returns it."""
    df = pd.DataFrame([asdict(s) for s in frame_specs(cfg) + triplet_specs(cfg)])
    cols = ["level", "name", "light", "group", "unit", "single_shot_ok", "description"]
    df = df[cols + ["physical_meaning"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
