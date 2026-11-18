"""Detect glints inside the ROI (Phase P2).

Glints are specular reflections of the surface moisture film ("bright spots"). They are NOT
bacteria or colonies: they are used only as moisture / surface-texture features
(``docs/01_RESEARCH_AND_RATIONALE.md`` §3).

Algorithm (``image-pipeline`` skill, parameters in ``configs/default.yaml`` ``glint``), on the
verified light's channel at glint resolution (``vision.glint_long_side``):

1. Pixels outside the ROI are set to the ROI median (so the meat/background border is not a
   peak), then a white top-hat with a disk of radius ``glint.tophat_r`` px removes the smooth
   meat shading.
2. Threshold = median + ``glint.k`` x MAD of the top-hat values inside the ROI, and at least
   ``glint.abs_min`` (0-255).
3. 8-connected components inside the ROI with area >= ``glint.min_area_px``.

``detect_glints_relative`` (S06, open question 7) keeps steps 1 and 3 but replaces the absolute
floor by a rule relative to the local surface brightness (the opening): a pixel is a glint if its
top-hat is above the noise threshold (median + k x MAD, at least ``glint.rel_floor``) AND at least
``glint.rel_k`` x the local surface level. Specular and diffuse light both scale with the LED
power, so this ratio does not depend on how bright the light is; the absolute floor does.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from skimage import measure

from foodlens.config import GlintConfig

PROP_COLS = [
    "label",
    "area",
    "centroid_y",
    "centroid_x",
    "eccentricity",
    "mean_intensity",
    "max_intensity",
    "saturated_frac",
]
DENSITY_PER_PX = 1e4  # glint density is reported as count per 10^4 ROI pixels


@dataclass(frozen=True)
class GlintResult:
    """``labels``: int32 (H, W), 0 = background; ``props_df``: one row per glint (PROP_COLS).

    Areas in px, centroids in px (row, col), intensities 0-255 of the input channel.
    ``threshold`` is the top-hat threshold actually used (0-255).
    """

    labels: np.ndarray
    props_df: pd.DataFrame
    threshold: float


def white_tophat(channel: np.ndarray, radius: int) -> np.ndarray:
    """White top-hat (image minus its opening) with an elliptical disk of ``radius`` px."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, k)


def glint_threshold(tophat_in_roi: np.ndarray, cfg: GlintConfig) -> float:
    """max(median + k * MAD, abs_min) of the top-hat values inside the ROI (0-255)."""
    med = float(np.median(tophat_in_roi))
    mad = float(np.median(np.abs(tophat_in_roi - med)))
    return max(med + cfg.k * mad, float(cfg.abs_min))


def _props(labels: np.ndarray, channel: np.ndarray, cfg: GlintConfig) -> pd.DataFrame:
    rows = []
    for p in measure.regionprops(labels, intensity_image=channel):
        vals = channel[p.coords[:, 0], p.coords[:, 1]]
        rows.append(
            {
                "label": p.label,
                "area": int(p.area),
                "centroid_y": float(p.centroid[0]),
                "centroid_x": float(p.centroid[1]),
                "eccentricity": float(p.eccentricity),
                "mean_intensity": float(vals.mean()),
                "max_intensity": float(vals.max()),
                "saturated_frac": float((vals >= cfg.sat_level).mean()),
            }
        )
    return pd.DataFrame(rows, columns=PROP_COLS)


def _prepare(
    channel: np.ndarray, roi_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(uint8 channel, bool ROI, channel with outside-ROI pixels set to the ROI median)."""
    if channel.shape != roi_mask.shape:
        raise ValueError(f"channel {channel.shape} and roi_mask {roi_mask.shape} differ")
    roi = roi_mask.astype(bool)
    if not roi.any():
        raise ValueError("roi_mask is empty")
    ch = channel.astype(np.uint8)
    # Outside the ROI use the ROI median, so the meat/background border is not a top-hat peak.
    return ch, roi, np.where(roi, ch, np.uint8(np.median(ch[roi])))


def detect_glints(channel: np.ndarray, roi_mask: np.ndarray, cfg: GlintConfig) -> GlintResult:
    """Glints of one uint8 channel (H, W) inside the bool ``roi_mask`` (same shape)."""
    ch, roi, filled = _prepare(channel, roi_mask)
    th = white_tophat(filled, cfg.tophat_r)
    thr = glint_threshold(th[roi].astype(np.float32), cfg)
    return _components((th > thr) & roi, ch, thr, cfg)


def detect_glints_relative(
    channel: np.ndarray, roi_mask: np.ndarray, cfg: GlintConfig
) -> GlintResult:
    """Illumination-relative glints: top-hat > max(median + k x MAD, ``rel_floor``) and
    top-hat >= ``rel_k`` x the local surface level (the opening, 0-255).

    ``threshold`` in the result is the scalar noise threshold (0-255).
    """
    ch, roi, filled = _prepare(channel, roi_mask)
    th = white_tophat(filled, cfg.tophat_r)
    surface = filled - th  # the opening: local surface brightness without the peaks
    vals = th[roi].astype(np.float32)
    med = float(np.median(vals))
    noise = max(med + cfg.k * float(np.median(np.abs(vals - med))), float(cfg.rel_floor))
    binary = (th > noise) & (th.astype(np.float32) >= cfg.rel_k * surface) & roi
    return _components(binary, ch, noise, cfg)


def _components(binary: np.ndarray, ch: np.ndarray, thr: float, cfg: GlintConfig) -> GlintResult:
    """8-connected components of ``binary`` with area >= ``min_area_px`` and their properties."""
    labels = measure.label(binary, connectivity=2)
    if labels.max():
        sizes = np.bincount(labels.ravel())
        small = sizes < cfg.min_area_px
        small[0] = False
        labels[small[labels]] = 0
        labels = measure.label(labels > 0, connectivity=2)
    labels = labels.astype(np.int32)
    return GlintResult(labels, _props(labels, ch, cfg), thr)


def glint_features(result: GlintResult, roi_area_px: int) -> dict[str, float]:
    """Single-frame glint features (``image-pipeline`` skill list).

    Keys: ``glint_count``, ``glint_density`` (count per 10^4 ROI px), ``glint_area_frac``,
    ``glint_size_p50``, ``glint_size_p90``, ``glint_size_max`` (px), ``glint_ecc_mean``,
    ``glint_sat_frac`` (saturated share of all glint pixels), ``glint_brightness_median`` (0-255).
    With zero glints, size / shape / brightness values are undefined and returned as NaN.
    """
    if roi_area_px <= 0:
        raise ValueError("roi_area_px must be > 0")
    p = result.props_df
    n = len(p)
    total = float(p["area"].sum()) if n else 0.0
    nan = float("nan")
    return {
        "glint_count": float(n),
        "glint_density": n / roi_area_px * DENSITY_PER_PX,
        "glint_area_frac": total / roi_area_px,
        "glint_size_p50": float(p["area"].quantile(0.5)) if n else nan,
        "glint_size_p90": float(p["area"].quantile(0.9)) if n else nan,
        "glint_size_max": float(p["area"].max()) if n else nan,
        "glint_ecc_mean": float(p["eccentricity"].mean()) if n else nan,
        "glint_sat_frac": float((p["saturated_frac"] * p["area"]).sum() / total) if n else nan,
        "glint_brightness_median": float(p["mean_intensity"].median()) if n else nan,
    }
