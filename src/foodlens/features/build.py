"""Frame features (P4.T2). ``extract_frame_features`` is THE function the Predictor calls at
inference: no training-only shortcuts, it sees one photo, its light and its masks.

It reuses the P3 measurement code (``eda/frame_stats.intensity_stats`` / ``shadow_stats``,
``vision.glints``, ``vision.texture``) and adds the P4 decisions (S06 open questions 5-8):
- main glint features = relative rule (``glint_*``), with glints whose centroid lies within
  ``features.glint_edge_px`` of the ROI edge dropped; the fixed-floor rule is kept as the
  ``glintabs_*`` "old method" baseline;
- gain-free ratios (``roi_cv``, ``roi_mean_rel_ref``, ...); LBP next to GLCM.
Table-level post-processing (training data only, needs QC / session context):
``blur_to_nan`` (W_BLUR frames: glint + texture -> NaN) and ``step_correct_frames`` (brightness
divided by the scene-step gain, the D2A camera colour-balance jump).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd

from foodlens.config import Config
from foodlens.eda.frame_stats import intensity_stats, shadow_stats
from foodlens.eda.trends import step_correct
from foodlens.features.schema import frame_specs, group_features, multiplicative_features
from foodlens.vision.glints import (
    GlintResult,
    detect_glints,
    detect_glints_relative,
    glint_features,
)
from foodlens.vision.roi import CHANNEL_INDEX, resize_mask
from foodlens.vision.texture import glcm_features, lbp_features

NAN = float("nan")


def _div(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(b) and np.isfinite(a) else NAN


def edge_inner(mask: np.ndarray, band_px: int) -> np.ndarray:
    """``mask`` eroded by an elliptical disk of radius ``band_px`` (px); ``mask`` if 0."""
    if band_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def drop_edge_glints(res: GlintResult, inner: np.ndarray) -> GlintResult:
    """Keep only glints whose (rounded) centroid lies inside ``inner``."""
    p = res.props_df
    if p.empty:
        return res
    ys = p["centroid_y"].round().astype(int).clip(0, inner.shape[0] - 1)
    xs = p["centroid_x"].round().astype(int).clip(0, inner.shape[1] - 1)
    keep = inner[ys.to_numpy(), xs.to_numpy()]
    return GlintResult(res.labels, p[keep].reset_index(drop=True), res.threshold)


def _glints(ch: np.ndarray, lit: np.ndarray, cfg: Config) -> tuple[dict[str, float], np.ndarray]:
    """(glint features of both rules, bool mask of all glint pixels to exclude from intensity)."""
    g_abs = detect_glints(ch, lit, cfg.glint)
    g_rel = detect_glints_relative(ch, lit, cfg.glint)
    inner = edge_inner(lit, cfg.features.glint_edge_px)
    area_inner = int(inner.sum()) if inner.any() else int(lit.sum())
    rel = glint_features(drop_edge_glints(g_rel, inner), area_inner)
    out = {k: v for k, v in rel.items() if k != "glint_count"}
    absf = glint_features(g_abs, int(lit.sum()))
    out |= {k.replace("glint_", "glintabs_"): v for k, v in absf.items() if k != "glint_count"}
    return out, (g_abs.labels > 0) | (g_rel.labels > 0)


def _derived(f: dict[str, float]) -> dict[str, float]:
    """Gain-free ratios computed from the frame's own values."""
    return {
        "roi_cv": _div(f["roi_std"], f["roi_mean"]),
        "roi_iqr_rel": _div(f["roi_iqr"], f["roi_median"]),
        "roi_mean_rel_ref": _div(f["roi_mean"], f["ref_mean"]),
        "shd_rel_roi": _div(f["shd_mean"], f["roi_mean"]),
        "glint_brightness_rel": _div(f["glint_brightness_median"], f["roi_median"]),
    }


def extract_frame_features(
    img_rgb: np.ndarray,
    light: str,
    roi_mask: np.ndarray,
    cfg: Config,
    shadow_mask: np.ndarray | None = None,
    ref_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """All frame features (``schema.frame_specs`` names) of one RGB uint8 photo.

    ``img_rgb`` should be at ``vision.glint_long_side``; ``light`` is the verified LED (RED /
    GREEN / BLUE); masks are bool and are resized to the image (nearest). Without a shadow /
    reference mask the shadow / reference features are NaN. Also returns ``ref_mean`` (meta).
    Raises ``ValueError`` if the ROI is empty.
    """
    ch = img_rgb[..., CHANNEL_INDEX[light]]
    lit = resize_mask(roi_mask, ch.shape)
    if not lit.any():
        raise ValueError("ROI mask is empty")
    glints, glint_px = _glints(ch, lit, cfg)
    clean = lit & ~glint_px
    f: dict[str, Any] = intensity_stats(ch[clean], "roi_")
    f["roi_sat_frac"] = float((ch[lit] >= cfg.glint.sat_level).mean())
    shadow = np.zeros_like(lit)
    if shadow_mask is not None:
        shadow = resize_mask(shadow_mask, ch.shape) & ~lit
    f |= shadow_stats(ch, shadow, cfg)
    full = clean | shadow
    f["full_median"] = float(np.median(ch[full]))
    f["full_mean"] = float(ch[full].astype(np.float32).mean())
    f["ref_mean"] = NAN
    if ref_mask is not None:
        ref = resize_mask(ref_mask, ch.shape)
        f["ref_mean"] = float(ch[ref].astype(np.float32).mean()) if ref.any() else NAN
    f |= glints | glcm_features(ch, lit, cfg.vision) | lbp_features(ch, lit, cfg.vision)
    f |= _derived(f)
    return {s.name: f[s.name] for s in frame_specs(cfg)} | {"ref_mean": f["ref_mean"]}


def blur_to_nan(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """W_BLUR frames (``qc_reasons``): features of ``features.blur_nan_groups`` -> NaN, flag
    ``blur_nan``. Defocus smears glints and texture; brightness stays valid (question 5)."""
    out = df.copy()
    out["blur_nan"] = out["qc_reasons"].fillna("").str.contains("W_BLUR")
    cols = group_features(cfg, cfg.features.blur_nan_groups)
    out.loc[out["blur_nan"], cols] = np.nan
    return out


def step_correct_frames(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Grey-level intensity features divided by the scene-step gain per session x light
    (``eda.trends.step_correct``); ``step_gain`` = raw / corrected ``roi_mean`` (1 = no step).
    Ratios, glints and texture are not changed (gain-free or not multiplicative)."""
    cols = multiplicative_features(cfg)
    sc = step_correct(df, cols, cfg.qc.step_k)
    sc["step_gain"] = (df["roi_mean"] / sc["roi_mean"]).where(sc["roi_mean"] > 0, 1.0)
    return sc
