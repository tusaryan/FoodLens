"""Segment the meat region of interest (multi-Otsu + morphology + largest component) (Phase P2).

Algorithm (``image-pipeline`` skill, parameters in ``configs/default.yaml`` ``roi``), applied at
work resolution (``vision.work_long_side``) to the verified light's channel:

1. Gaussian blur (``roi.blur_sigma`` px).
2. Multi-Otsu with ``roi.otsu_classes`` classes; the lowest threshold (floored at
   ``roi.min_level``, 0-255) separates the meat from the dark box. Plain 2-class Otsu kept only
   the brightly lit half of the meat on the real frames (S04).
3. Close (disk ``roi.close_r``), open (disk ``roi.open_r``), fill holes, keep the largest
   connected component; reject if its area fraction is outside [``min_frac``, ``max_frac``].
4. Erode by ``roi.erode_px`` to drop the edge / shadow rim.

The session ROI (``session_reference_mask``) is the per-pixel median of the masks of the
``roi.ref_lights`` frames; ``exclude_foreign`` then removes non-meat material (the holder / tray)
that touches the ROI edge and is much bluer than meat. It is the ROI used for every frame of the
session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from foodlens.config import ROIConfig
from foodlens.errors import E_NO_MEAT_FOUND

CHANNEL_INDEX = {"RED": 0, "GREEN": 1, "BLUE": 2}


@dataclass(frozen=True)
class RoiResult:
    """Segmentation of one frame.

    ``mask``: bool (H, W); ``bbox``: (y0, x0, y1, x1) px, end-exclusive, or None;
    ``area_frac``: mask pixels / frame pixels; ``reason``: "" or an error code.
    """

    mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    area_frac: float
    ok: bool
    reason: str


def _failed(shape: tuple[int, ...], area_frac: float = 0.0) -> RoiResult:
    return RoiResult(np.zeros(shape[:2], bool), None, area_frac, False, E_NO_MEAT_FOUND)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, x0, y1, x1) end-exclusive bounding box of a bool mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Bool mask of the largest 8-connected component (all False if ``mask`` is empty)."""
    lab = measure.label(mask, connectivity=2)
    if lab.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def roi_threshold(blurred: np.ndarray, cfg: ROIConfig) -> float | None:
    """Lowest multi-Otsu threshold (0-255) floored at ``min_level``; None if not computable."""
    if float(blurred.max()) <= cfg.min_level:
        return None
    try:
        t = filters.threshold_multiotsu(blurred, classes=cfg.otsu_classes)[0]
    except ValueError:  # too few distinct grey levels for the requested classes
        return None
    return max(float(t), cfg.min_level)


def segment_roi(img_rgb: np.ndarray, light: str, cfg: ROIConfig) -> RoiResult:
    """Meat mask of one RGB uint8 frame lit by ``light`` (RED/GREEN/BLUE)."""
    if light not in CHANNEL_INDEX:
        raise ValueError(f"light must be one of {list(CHANNEL_INDEX)}, got {light!r}")
    plane = img_rgb[..., CHANNEL_INDEX[light]].astype(np.float32)
    blurred = filters.gaussian(plane, sigma=cfg.blur_sigma, preserve_range=True)
    t = roi_threshold(blurred, cfg)
    if t is None:
        return _failed(img_rgb.shape)
    mask = blurred > t
    if cfg.close_r:
        mask = morphology.closing(mask, morphology.disk(cfg.close_r))
    if cfg.open_r:
        mask = morphology.opening(mask, morphology.disk(cfg.open_r))
    mask = largest_component(ndi.binary_fill_holes(mask))
    frac = float(mask.mean())
    if not cfg.min_frac <= frac <= cfg.max_frac:
        return _failed(img_rgb.shape, frac)
    if cfg.erode_px:
        mask = morphology.erosion(mask, morphology.disk(cfg.erode_px))
    bbox = mask_bbox(mask)
    if bbox is None:
        return _failed(img_rgb.shape)
    return RoiResult(mask, bbox, float(mask.mean()), True, "")


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    out = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)


def session_reference_mask(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel median (majority vote) of ``masks``, largest component, holes filled.

    Masks of another shape are resized (nearest) to the shape of the first one.
    """
    if not masks:
        raise ValueError("session_reference_mask needs at least one mask")
    shape = masks[0].shape
    stack = np.stack([resize_mask(m, shape) for m in masks])
    med = np.median(stack, axis=0) >= 0.5
    return largest_component(ndi.binary_fill_holes(med))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two bool masks of the same shape (0.0 if both are empty)."""
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """(cx, cy) of a bool mask as fractions (0-1) of width and height; NaN if empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return float("nan"), float("nan")
    h, w = mask.shape
    return float(xs.mean() / w), float(ys.mean() / h)


def exclude_foreign(
    mask: np.ndarray, red_med: np.ndarray, blue_med: np.ndarray, cfg: ROIConfig
) -> tuple[np.ndarray, int]:
    """Remove holder / tray material from a session ROI; return (new mask, px removed).

    ``red_med``/``blue_med`` are session-median RED and BLUE channel images (0-255, same shape as
    ``mask``). Candidates: ROI pixels with blurred (BLUE + 1) / (RED + 1) above
    ``roi.foreign_max_blue_red`` (meat is ~0.3-0.45). Only candidate regions of at least
    ``roi.foreign_min_area_px`` px that touch the ROI edge are removed (interior bluish specks are
    meat / glints), grown by ``roi.erode_px`` to drop their mixed rim.
    """
    if not mask.shape == red_med.shape == blue_med.shape:
        raise ValueError("mask, red_med and blue_med must have the same shape")
    r = filters.gaussian(red_med.astype(np.float32), sigma=cfg.blur_sigma, preserve_range=True)
    b = filters.gaussian(blue_med.astype(np.float32), sigma=cfg.blur_sigma, preserve_range=True)
    cand = mask & ((b + 1.0) / (r + 1.0) > cfg.foreign_max_blue_red)
    edge = mask & ~ndi.binary_erosion(mask, border_value=0)
    lab = measure.label(cand, connectivity=2)
    sizes = np.bincount(lab.ravel())
    hit = np.unique(lab[edge & cand])
    keep_ids = [i for i in hit if i and sizes[i] >= cfg.foreign_min_area_px]
    removed = np.isin(lab, keep_ids)
    if cfg.erode_px and removed.any():
        removed = morphology.dilation(removed, morphology.disk(cfg.erode_px))
    out = largest_component(mask & ~removed)
    return out, int(mask.sum() - out.sum())
