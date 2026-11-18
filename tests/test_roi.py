"""Tests for vision/roi.py on tiny synthetic images (never the real dataset)."""

from __future__ import annotations

import numpy as np
import pytest

from foodlens.config import load_config
from foodlens.errors import E_NO_MEAT_FOUND
from foodlens.vision.roi import (
    exclude_foreign,
    mask_bbox,
    mask_centroid,
    mask_iou,
    segment_roi,
    session_reference_mask,
)


@pytest.fixture
def roi_cfg():
    return load_config().roi


def _ellipse(h: int, w: int, cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _truth(h: int = 300, w: int = 400) -> np.ndarray:
    """Ground truth of conftest._make_meat_image (centre, radii 0.3 x size)."""
    return _ellipse(h, w, h / 2, w / 2, h * 0.3, w * 0.3)


@pytest.mark.parametrize("light", ["RED", "GREEN", "BLUE"])
def test_ellipse_on_noise_matches_truth(make_meat_image, roi_cfg, light):
    cfg = roi_cfg.model_copy(update={"erode_px": 0})
    res = segment_roi(make_meat_image(light=light, seed=3), light, cfg)
    assert res.ok and res.reason == ""
    assert mask_iou(res.mask, _truth()) > 0.9
    assert res.area_frac == pytest.approx(_truth().mean(), rel=0.1)


def test_erosion_shrinks_mask_inside_truth(make_meat_image, roi_cfg):
    res = segment_roi(make_meat_image(light="RED"), "RED", roi_cfg)
    assert res.ok
    assert res.mask.sum() < _truth().sum()
    assert np.logical_and(res.mask, ~_truth()).sum() == 0  # eroded mask stays inside the meat


def test_dark_image_has_no_meat(make_dark_image, roi_cfg):
    res = segment_roi(make_dark_image(), "RED", roi_cfg)
    assert not res.ok
    assert res.reason == E_NO_MEAT_FOUND
    assert res.bbox is None and not res.mask.any()


def test_two_blobs_keeps_larger(roi_cfg):
    h, w = 300, 400
    rng = np.random.default_rng(1)
    img = rng.normal(4, 2, size=(h, w, 3))
    big = _ellipse(h, w, 150, 120, 80, 90)
    small = _ellipse(h, w, 150, 330, 40, 40)
    img[..., 0][big | small] = 120
    img = np.clip(img, 0, 255).astype(np.uint8)
    res = segment_roi(img, "RED", roi_cfg.model_copy(update={"erode_px": 0}))
    assert res.ok
    assert mask_iou(res.mask, big) > 0.9
    assert not np.logical_and(res.mask, small).any()


def test_too_large_region_is_rejected(roi_cfg):
    img = np.zeros((100, 100, 3), np.uint8)
    img[5:95, 5:95, 0] = 150  # 81 % of the frame > max_frac 0.8
    img[0, 0, 0] = 1
    res = segment_roi(img, "RED", roi_cfg.model_copy(update={"erode_px": 0}))
    assert not res.ok and res.reason == E_NO_MEAT_FOUND
    assert res.area_frac > roi_cfg.max_frac


def test_unknown_light_raises(make_meat_image, roi_cfg):
    with pytest.raises(ValueError):
        segment_roi(make_meat_image(), "WHITE", roi_cfg)


def test_session_reference_is_majority_vote():
    base = _ellipse(60, 80, 30, 40, 15, 20)
    shifted = np.roll(base, 3, axis=1)
    outlier = _ellipse(60, 80, 10, 10, 5, 5)
    ref = session_reference_mask([base, base, shifted, outlier, base])
    assert mask_iou(ref, base) > 0.95


def test_session_reference_resizes_masks():
    a = _ellipse(60, 80, 30, 40, 15, 20)
    b = _ellipse(120, 160, 60, 80, 30, 40)
    ref = session_reference_mask([a, b, a])
    assert ref.shape == a.shape
    assert mask_iou(ref, a) > 0.9


def test_session_reference_needs_masks():
    with pytest.raises(ValueError):
        session_reference_mask([])


def test_mask_iou_values():
    a = np.zeros((10, 10), bool)
    a[:, :5] = True
    b = np.zeros((10, 10), bool)
    b[:, 5:] = True
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, b) == 0.0
    assert mask_iou(a, a | b) == pytest.approx(0.5)
    assert mask_iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 0.0
    with pytest.raises(ValueError):
        mask_iou(a, np.zeros((5, 5), bool))


def test_bbox_and_centroid():
    m = np.zeros((10, 20), bool)
    m[2:4, 5:9] = True
    assert mask_bbox(m) == (2, 5, 4, 9)
    cx, cy = mask_centroid(m)
    assert cx == pytest.approx(6.5 / 20) and cy == pytest.approx(2.5 / 10)
    assert mask_bbox(np.zeros((3, 3), bool)) is None
    assert np.isnan(mask_centroid(np.zeros((3, 3), bool))[0])


def _foreign_scene():
    """Meat rectangle (RED 100, BLUE 40) with a bluish holder strip on its bottom edge and a
    bluish interior speck. Returns mask, red, blue, holder, speck."""
    h, w = 120, 160
    mask = np.zeros((h, w), bool)
    mask[20:100, 30:130] = True
    red = np.full((h, w), 2.0)
    blue = np.full((h, w), 2.0)
    red[mask], blue[mask] = 100, 40
    holder = np.zeros_like(mask)
    holder[85:100, 60:100] = True  # touches the ROI's bottom edge
    red[holder], blue[holder] = 20, 50
    speck = np.zeros_like(mask)
    speck[50:60, 75:85] = True  # interior, bluish but surrounded by meat
    blue[speck] = 150
    return mask, red, blue, holder, speck


def test_exclude_foreign_removes_edge_holder_keeps_interior(roi_cfg):
    mask, red, blue, holder, speck = _foreign_scene()
    out, removed = exclude_foreign(mask, red, blue, roi_cfg)
    assert not np.logical_and(out, holder).any()
    assert out[speck].all()
    assert removed == mask.sum() - out.sum() > holder.sum()  # holder plus its grown rim
    assert removed < 3 * holder.sum()


def test_exclude_foreign_noop_on_plain_meat(roi_cfg):
    mask, red, blue, _, _ = _foreign_scene()
    red[mask], blue[mask] = 100, 40
    out, removed = exclude_foreign(mask, red, blue, roi_cfg)
    assert removed == 0 and np.array_equal(out, mask)
    with pytest.raises(ValueError):
        exclude_foreign(mask, red[:10], blue, roi_cfg)
