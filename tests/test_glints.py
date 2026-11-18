"""Tests for vision/glints.py and vision/glint_dynamics.py on synthetic images / tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foodlens.config import load_config
from foodlens.vision.glint_dynamics import match_glints
from foodlens.vision.glints import detect_glints, glint_features

DOT_R = 2
DOT_AREA = 13  # pixels of a radius-2 digital disk
H, W = 300, 400


@pytest.fixture
def gcfg():
    return load_config().glint


def _dot(img: np.ndarray, y: int, x: int, value: float, r: int = DOT_R) -> None:
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    img[(yy - y) ** 2 + (xx - x) ** 2 <= r**2] = value


def _scene(inside: list[tuple[int, int]], outside: list[tuple[int, int]], seed: int = 0):
    """Channel: background ~4, meat rectangle ~120 (noise sd 6), saturated dots. ROI = rectangle."""
    rng = np.random.default_rng(seed)
    ch = rng.normal(4, 2, size=(H, W))
    roi = np.zeros((H, W), bool)
    roi[60:240, 80:320] = True
    ch[roi] = rng.normal(120, 6, size=int(roi.sum()))
    for y, x in inside + outside:
        _dot(ch, y, x, 255)
    return np.clip(ch, 0, 255).astype(np.uint8), roi


GRID = [(90 + 30 * i, 110 + 45 * j) for i in range(2) for j in range(5)]  # 10 dots in the ROI


def test_detects_ten_known_dots(gcfg):
    ch, roi = _scene(GRID, [])
    res = detect_glints(ch, roi, gcfg)
    p = res.props_df
    assert len(p) == 10
    assert np.allclose(p["area"], DOT_AREA, rtol=0.2)
    got = sorted(zip(p["centroid_y"].round(), p["centroid_x"].round(), strict=True))
    assert got == sorted((float(y), float(x)) for y, x in GRID)
    assert (p["saturated_frac"] == 1.0).all()
    assert res.threshold >= gcfg.abs_min


def test_dots_outside_roi_are_ignored(gcfg):
    ch, roi = _scene(GRID, [(20, 20), (280, 380), (30, 200)])
    res = detect_glints(ch, roi, gcfg)
    assert len(res.props_df) == 10
    assert not (res.labels[~roi] > 0).any()


def test_no_glints_on_plain_meat(gcfg):
    ch, roi = _scene([], [])
    res = detect_glints(ch, roi, gcfg)
    assert len(res.props_df) == 0
    f = glint_features(res, int(roi.sum()))
    assert f["glint_count"] == 0 and f["glint_area_frac"] == 0
    assert np.isnan(f["glint_size_p50"])


def test_dim_spots_below_abs_min_ignored(gcfg):
    ch, roi = _scene([], [])
    _dot(ch, 150, 200, 120 + gcfg.abs_min / 2)  # top-hat ~abs_min / 2 above the meat
    assert len(detect_glints(ch, roi, gcfg).props_df) == 0


def test_glint_features_values(gcfg):
    ch, roi = _scene(GRID, [])
    area = int(roi.sum())
    f = glint_features(detect_glints(ch, roi, gcfg), area)
    assert f["glint_count"] == 10
    assert f["glint_density"] == pytest.approx(10 / area * 1e4)
    assert f["glint_area_frac"] == pytest.approx(10 * DOT_AREA / area, rel=0.2)
    assert f["glint_size_p50"] == pytest.approx(DOT_AREA, rel=0.2)
    assert f["glint_sat_frac"] == 1.0
    with pytest.raises(ValueError):
        glint_features(detect_glints(ch, roi, gcfg), 0)


def test_detect_glints_input_checks(gcfg):
    ch, roi = _scene(GRID, [])
    with pytest.raises(ValueError):
        detect_glints(ch, roi[:10], gcfg)
    with pytest.raises(ValueError):
        detect_glints(ch, np.zeros_like(roi), gcfg)


def _props(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["centroid_y", "centroid_x", "area"])


def test_match_persist_birth_death():
    prev = _props([(10, 10, 10), (50, 50, 10), (90, 90, 10)])
    curr = _props([(11, 10, 10), (50, 52, 10), (150, 150, 10)])
    out = match_glints(prev, curr, max_dist_px=6, area_ratio=1.5)
    assert out == {"births": 1, "deaths": 1, "merges": 0, "splits": 0, "persist": 2}


def test_match_merge():
    prev = _props([(50, 50, 10), (50, 58, 10), (200, 200, 10)])
    curr = _props([(50, 54, 25), (200, 201, 10)])
    out = match_glints(prev, curr, max_dist_px=6, area_ratio=1.5)
    assert out == {"births": 0, "deaths": 0, "merges": 1, "splits": 0, "persist": 1}


def test_match_split():
    prev = _props([(50, 54, 25)])
    curr = _props([(50, 50, 10), (50, 58, 10)])
    out = match_glints(prev, curr, max_dist_px=6, area_ratio=1.5)
    assert out == {"births": 0, "deaths": 0, "merges": 0, "splits": 1, "persist": 0}


def test_close_pair_without_growth_is_not_a_merge():
    prev = _props([(50, 50, 10), (50, 58, 10)])
    curr = _props([(50, 54, 12)])  # area grew < 1.5x: one persists, one dies
    out = match_glints(prev, curr, max_dist_px=6, area_ratio=1.5)
    assert out == {"births": 0, "deaths": 1, "merges": 0, "splits": 0, "persist": 1}


def test_match_empty_frames():
    empty = _props([])
    two = _props([(1, 1, 5), (30, 30, 5)])
    assert match_glints(empty, two, 6, 1.5)["births"] == 2
    assert match_glints(two, empty, 6, 1.5)["deaths"] == 2
    assert match_glints(empty, empty, 6, 1.5) == dict.fromkeys(
        ["births", "deaths", "merges", "splits", "persist"], 0
    )
