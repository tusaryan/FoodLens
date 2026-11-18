"""Texture features on the masked ROI (P3.T1 GLCM; P4 adds LBP).

GLCM (grey-level co-occurrence matrix, skimage ``graycomatrix``) on the ROI bounding box with the
ROI mask: ROI pixels are quantised to levels 1..L (``vision.glcm_levels``) over the ROI's own
robust range (``vision.glcm_clip_pct`` percentiles), pixels outside the ROI get level 0, and row /
column 0 of the co-occurrence matrix is dropped before normalising, so only ROI-ROI pixel pairs
count. Quantising over the ROI's own range makes the texture values insensitive to the overall
brightness of the light (a dim BLUE frame and a bright RED frame use all L levels).
Distances ``vision.glcm_distances`` (px at the input resolution), angles 0, 45, 90, 135 degrees;
each property is averaged over distances and angles.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from foodlens.config import VisionConfig

GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")


def quantise_roi(channel: np.ndarray, mask: np.ndarray, levels: int, clip_pct) -> np.ndarray:
    """uint8 image: ROI pixels mapped to 1..``levels`` over the ROI [p_lo, p_hi] range, others 0."""
    vals = channel[mask].astype(np.float32)
    lo, hi = np.percentile(vals, clip_pct)
    scale = (levels - 1) / max(float(hi - lo), 1.0)
    q = np.clip(np.floor((channel.astype(np.float32) - lo) * scale), 0, levels - 1) + 1
    return np.where(mask, q, 0).astype(np.uint8)


def glcm_features(channel: np.ndarray, mask: np.ndarray, cfg: VisionConfig) -> dict[str, float]:
    """``glcm_<prop>`` for GLCM_PROPS of the uint8 ``channel`` inside the bool ``mask``.

    Contrast is in squared grey levels of the L-level quantisation; the others are unitless.
    Raises ``ValueError`` if the mask is empty or shapes differ.
    """
    if channel.shape != mask.shape:
        raise ValueError(f"channel {channel.shape} and mask {mask.shape} differ")
    if not mask.any():
        raise ValueError("mask is empty")
    ys, xs = np.nonzero(mask)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    q = quantise_roi(channel[box], mask[box], cfg.glcm_levels, cfg.glcm_clip_pct)
    m = graycomatrix(
        q, cfg.glcm_distances, GLCM_ANGLES, levels=cfg.glcm_levels + 1, symmetric=True
    ).astype(np.float64)[1:, 1:]
    if (m.sum((0, 1)) == 0).any():
        raise ValueError("ROI too small for the GLCM distances (no pixel pairs)")
    return {f"glcm_{k}": float(graycoprops(m, k).mean()) for k in GLCM_PROPS}
