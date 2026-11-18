"""Tests for the P3 EDA functions (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foodlens.config import load_config
from foodlens.eda.dynamics import pattern_label
from foodlens.eda.frame_stats import intensity_stats
from foodlens.eda.meat_zones import full_meat_mask
from foodlens.eda.signal import signal_ranking
from foodlens.eda.trends import (
    mixed_slope,
    relative_to_first,
    series_trend,
    step_correct,
)
from foodlens.eda.visual_sample import trend_outlier_score
from foodlens.features.triplets import add_ratios, assign_triplets, triplet_table
from foodlens.vision.glints import detect_glints, detect_glints_relative
from foodlens.vision.texture import glcm_features

CFG = load_config()
T0 = pd.Timestamp("2026-09-01 10:00", tz="Asia/Kolkata")


def _speckled(level: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Flat meat at ``level`` with 20 glints at 3x the level; returns (channel, roi)."""
    rng = np.random.default_rng(seed)
    ch = np.clip(rng.normal(level, 1.0, (200, 200)), 0, 255)
    for y, x in rng.integers(30, 170, size=(20, 2)):
        ch[y - 1 : y + 2, x - 1 : x + 2] = min(3 * level, 255)
    roi = np.zeros((200, 200), bool)
    roi[20:180, 20:180] = True
    return ch.astype(np.uint8), roi


def test_intensity_stats_values_and_empty():
    s = intensity_stats(np.arange(101, dtype=np.uint8), "roi_")
    assert s["roi_median"] == 50 and s["roi_p10"] == 10 and s["roi_iqr"] == 50
    assert np.isnan(intensity_stats(np.array([]), "x_")["x_mean"])


def test_relative_glints_do_not_depend_on_light_level():
    bright, roi = _speckled(60)
    dim, _ = _speckled(15)
    n_rel = [len(detect_glints_relative(c, roi, CFG.glint).props_df) for c in (bright, dim)]
    n_abs = [len(detect_glints(c, roi, CFG.glint).props_df) for c in (bright, dim)]
    assert n_rel[0] == n_rel[1] > 10  # same specks found at both light levels
    assert n_abs[1] < n_abs[0]  # the absolute floor loses them when the light is dim


def test_glcm_is_brightness_invariant_and_checks_mask():
    rng = np.random.default_rng(1)
    img = rng.integers(40, 200, (120, 120)).astype(np.uint8)
    mask = np.zeros_like(img, bool)
    mask[10:110, 10:110] = True
    a = glcm_features(img, mask, CFG.vision)
    b = glcm_features((img // 4).astype(np.uint8), mask, CFG.vision)
    assert a["glcm_contrast"] == pytest.approx(b["glcm_contrast"], rel=0.1)
    with pytest.raises(ValueError):
        glcm_features(img, np.zeros_like(mask), CFG.vision)


def _series(n: int = 20, step_at: int | None = None) -> pd.DataFrame:
    h = np.arange(n) * 0.05
    y = 100 - 10 * h
    seg = np.zeros(n)
    if step_at is not None:
        y[step_at:] *= 1.5
        seg[step_at:] = 1
    return pd.DataFrame(
        {
            "session_id": "s1",
            "light": "RED",
            "ts": [T0 + pd.Timedelta(hours=x) for x in h],
            "elapsed_h": h,
            "scene_segment": seg,
            "roi_mean": y,
            "glint_count": np.full(n, 7.0),
        }
    )


def test_step_correct_removes_the_gain_and_keeps_other_columns():
    df = _series(step_at=10)
    out = step_correct(df, ["roi_mean"], k=3)
    jump = out["roi_mean"].iloc[10] / out["roi_mean"].iloc[9]
    assert jump == pytest.approx(1.0, abs=0.02)
    assert (out["roi_mean"].iloc[:10] == df["roi_mean"].iloc[:10]).all()
    assert (out["glint_count"] == 7.0).all()


def test_relative_to_first_and_series_trend():
    df = _series()
    rel = relative_to_first(df, ["roi_mean"], n=1)
    assert rel["roi_mean"].iloc[0] == 1.0
    t = series_trend(df["elapsed_h"].to_numpy(), df["roi_mean"].to_numpy(), min_n=5)
    assert t["rho"] == pytest.approx(-1.0) and t["slope_h"] == pytest.approx(-10.0)
    assert np.isnan(series_trend(np.arange(3.0), np.ones(3), min_n=5)["rho"])


def test_mixed_slope_recovers_common_slope():
    rng = np.random.default_rng(0)
    rows = [
        {"session_id": s, "elapsed_h": h, "y": off - 2 * h + rng.normal(0, 0.1)}
        for s, off in (("a", 10), ("b", 30), ("c", 50))
        for h in np.linspace(0, 3, 15)
    ]
    r = mixed_slope(pd.DataFrame(rows), "y")
    assert r["slope_h"] == pytest.approx(-2.0, abs=0.1) and r["n_sessions"] == 3


def _frames(lights: list[str], gaps_s: list[float]) -> pd.DataFrame:
    ts = T0 + pd.to_timedelta(np.cumsum([0, *gaps_s]), unit="s")
    return pd.DataFrame({"session_id": "s1", "light": lights, "ts": ts})


def test_triplets_missing_frames_do_not_shift_cycles():
    order = CFG.capture.light_order
    df = _frames(["RED", "GREEN", "RED", "GREEN", "BLUE", "GREEN", "BLUE"], [60] * 6)
    assert assign_triplets(df, order, 200).tolist() == [0, 0, 1, 1, 1, 2, 2]
    split = _frames(["RED", "GREEN", "BLUE"], [60, 300])
    assert assign_triplets(split, order, 200).tolist() == [0, 0, 1]


def test_triplet_table_and_ratios():
    df = _frames(["RED", "GREEN", "BLUE", "RED", "GREEN"], [60] * 4)
    df["roi_mean"] = [40.0, 20.0, 10.0, 44.0, 22.0]
    df["elapsed_h"] = np.arange(5) / 60
    t = triplet_table(df, ["roi_mean"], CFG.capture.light_order, 200)
    t, cols = add_ratios(t, ["roi_mean"])
    assert t["roi_mean_ratio_RG"].tolist() == [2.0, 2.0]
    assert t["has_BLUE"].tolist() == [True, False] and np.isnan(t["roi_mean_ratio_RB"].iloc[1])
    assert "roi_mean_ratio_GB" in cols


def test_signal_ranking_consistency():
    trends = pd.DataFrame(
        {
            "feature": ["f"] * 5 + ["g"] * 5,
            "light": "RED",
            "rho": [-0.9, -0.8, -0.7, -0.9, 0.5, 0.5, -0.5, 0.4, -0.3, 0.1],
            "p": 0.01,
            "slope_pct_h": -1.0,
        }
    )
    r = signal_ranking(trends, alpha=0.05, min_same=4).set_index("feature")
    assert r.at["f", "consistency_label"] == "4/5" and bool(r.at["f", "promising"])
    assert not bool(r.at["g", "promising"]) and r.index[0] == "f"


def test_pattern_labels():
    assert pattern_label(-0.8, 0.001, -0.7, 0.001, 0.05).startswith("evaporation")
    assert pattern_label(0.8, 0.001, -0.7, 0.001, 0.05).startswith("film breaking")
    assert pattern_label(0.8, 0.5, 0.7, 0.5, 0.05) == "no clear trend"


def test_outlier_score_flags_a_spike_but_not_tiny_wiggles():
    n = 30
    df = pd.DataFrame(
        {
            "session_id": "s1",
            "light": "RED",
            "ts": [T0 + pd.Timedelta(minutes=3 * i) for i in range(n)],
            "roi_median": 50 + 0.001 * np.arange(n) % 2,
            "glint_density": np.full(n, 2.0),
            "rglint_density": np.full(n, 2.0),
            "glcm_contrast": 3.0 + 0.001 * (np.arange(n) % 2),
        }
    )
    df.loc[15, "glint_density"] = 6.0
    z = trend_outlier_score(df, window=5, floor=0.05)
    assert z.idxmax() == 15 and z.drop(15).max() < 1.0


def test_full_meat_mask_adds_dim_meat_but_not_thin_strips():
    red = np.full((200, 200), 1.0)
    red[50:150, 40:100] = 80  # lit half
    red[50:150, 100:160] = 15  # shadowed half, same piece
    red[20:23, 10:190] = 20  # thin lit tray edge strip
    red[20:50, 60:63] = 20  # joined to the meat by a thin bridge
    lit = np.zeros_like(red, bool)
    lit[56:144, 46:94] = True
    full = full_meat_mask(red, red * 0.3, lit, CFG)
    assert full[100, 130] and full[100, 60]
    assert not full[21, 150]
