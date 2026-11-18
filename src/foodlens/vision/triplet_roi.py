"""Per-triplet meat masks (S07): the ROI used by the P4 feature table and by inference.

Why not one fixed session ROI (S04)? It was valid for the 2-6 h sessions, but in the 16-22 h
overnight sessions (EXP-3/4/5) the meat slides and shrinks, so a session-median mask covers tray
and misses meat. Each triplet therefore gets its own masks from ITS OWN RED frame (RED masks are
the stable ones, S04) and BLUE frame (holder / tray removal by BLUE/RED, S04), and all three
frames of the triplet use the same pixels (they are <= ``triplet.max_span_s`` apart). This is
also exactly what a single-shot predictor can compute from one R/G/B capture.

Steps (work resolution, ``vision.work_long_side``):
1. lit = ``segment_roi`` of the RED frame (multi-Otsu, close/open, largest component, erode).
2. Holder / tray touching the edge removed with ``exclude_foreign`` (needs the BLUE frame).
3. Neck cut: opening with a disk of ``features.roi_neck_r`` px + largest component. As the meat
   darkens with age, the dim lit tray strip next to it falls in the same multi-Otsu class and
   joins the mask through a thin neck (S07 QA: exp3/4/5 late frames); the opening cuts it.
4. Full meat (lit + shadowed side) = ``eda.meat_zones.full_meat_mask`` on the same RED / BLUE
   frames, neck-cut the same way; shadow = full minus lit.
Reference region per frame: lit non-meat pixels (tray / holder / walls) further than
``eda.ref_gap_px`` from the full meat and brighter than ``eda.ref_min_level`` in that frame's own
light: the LED / camera-gain control behind ``roi_mean_rel_ref``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

from foodlens.config import Config
from foodlens.eda.meat_zones import full_meat_mask
from foodlens.vision.roi import exclude_foreign, largest_component, segment_roi


@dataclass(frozen=True)
class TripletMasks:
    """Bool masks at work resolution. ``ok`` False -> ``reason`` holds the error code."""

    lit: np.ndarray
    shadow: np.ndarray
    full: np.ndarray
    ok: bool
    reason: str = ""


def neck_cut(mask: np.ndarray, radius: int) -> np.ndarray:
    """Largest component of ``mask`` opened with a disk of ``radius`` px (thin attachments cut);
    the input if the opening removes everything or ``radius`` is 0."""
    if radius <= 0 or not mask.any():
        return mask
    opened = largest_component(ndi.binary_opening(mask, structure=morphology.disk(radius)))
    return opened if opened.any() else mask


def triplet_masks(red_rgb: np.ndarray, blue_ch: np.ndarray | None, cfg: Config) -> TripletMasks:
    """Masks of one triplet from its RED frame (RGB uint8) and BLUE channel (uint8, same size;
    None if the triplet has no usable BLUE frame: no holder removal, shadow empty)."""
    seg = segment_roi(red_rgb, "RED", cfg.roi)
    empty = np.zeros(red_rgb.shape[:2], bool)
    if not seg.ok:
        return TripletMasks(empty, empty, empty, False, seg.reason or "E_NO_MEAT_FOUND")
    red = red_rgb[..., 0]
    lit = seg.mask
    if blue_ch is not None:
        lit, _ = exclude_foreign(lit, red, blue_ch, cfg.roi)
    lit = neck_cut(lit, cfg.features.roi_neck_r)
    if not lit.any():
        return TripletMasks(empty, empty, empty, False, "E_NO_MEAT_FOUND")
    full = lit
    if blue_ch is not None:
        full = neck_cut(full_meat_mask(red, blue_ch, lit, cfg), cfg.features.roi_neck_r) | lit
    return TripletMasks(lit, full & ~lit, full, True)


def frame_reference(ch_work: np.ndarray, full: np.ndarray, cfg: Config) -> np.ndarray | None:
    """Reference mask (see module doc) for one frame's lit channel at work resolution; None if
    smaller than ``roi.foreign_min_area_px``."""
    if ch_work.shape != full.shape:
        raise ValueError(f"channel {ch_work.shape} and mask {full.shape} differ")
    sigma = cfg.roi.blur_sigma
    smooth = filters.gaussian(ch_work.astype(np.float32), sigma=sigma, preserve_range=True)
    far = ~ndi.binary_dilation(full, iterations=cfg.eda.ref_gap_px)
    ref = far & (smooth > cfg.eda.ref_min_level)
    return ref if ref.sum() >= cfg.roi.foreign_min_area_px else None
