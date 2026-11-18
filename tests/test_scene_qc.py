"""Tests for qc/scene_qc.py: local robust z, scene steps, scene change, ROI QC table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from foodlens.config import load_config
from foodlens.qc.scene_qc import (
    ROI_QC_COLS,
    detect_steps,
    local_robust_z,
    roi_qc_table,
    scene_qc,
    session_steps,
)

CFG = load_config()
T0 = pd.Timestamp("2026-08-28 12:00:00", tz="Asia/Kolkata")


def _session(n: int = 30, sid: str = "s", light: str = "RED", offset_s: int = 0) -> pd.DataFrame:
    """n frames of one light, 180 s apart, with a slow trend and tiny jitter (real-like)."""
    rng = np.random.default_rng(0)
    i = np.arange(n)
    return pd.DataFrame(
        {
            "frame_id": [f"{sid}_{light}_{k:02d}" for k in i],
            "session_id": sid,
            "light": light,
            "ts": T0 + pd.to_timedelta(i * 180 + offset_s, unit="s"),
            "roi_area_frac": 0.2 + rng.normal(0, 0.0005, n),
            "roi_cx": 0.4 + rng.normal(0, 0.0005, n),
            "roi_cy": 0.5 + i * 0.0005,  # slow drift (meat settling)
            "work_w": 1024,
            "work_h": 768,
            "lit_mean": 10.0 - i * 0.03 + rng.normal(0, 0.02, n),  # slow darkening
            "sroi_mean": 60.0 - i * 0.1,
        }
    )


def test_local_robust_z_spike_and_trend():
    x = pd.Series(np.linspace(0, 5, 40))
    x[20] += 3.0
    z = local_robust_z(x, window=9, floor=0.1)
    assert abs(z[20]) > 6
    rest = z.drop(20).abs()
    assert (rest < 3).all()  # the trend alone is not an outlier (small bias at the 4 edge frames)
    assert (rest.iloc[4:-4] < 1.5).all()  # neighbours of the spike shift by one trend step


def test_local_robust_z_floor_suppresses_jitter():
    x = pd.Series(np.r_[np.zeros(20), [0.002], np.zeros(19)])  # MAD = 0 without a floor
    assert local_robust_z(x, window=9, floor=0.03).abs().max() < 1


def test_detect_steps_finds_persistent_shift_not_spike():
    x = np.r_[np.zeros(15), np.full(10, 0.3)]
    assert [i for i, _ in detect_steps(x, k=3, min_log=0.1)] == [15]
    spike = np.zeros(25)
    spike[12] = 1.0
    assert detect_steps(spike, k=3, min_log=0.1) == []


def test_session_steps_merge_lights():
    frames = []
    for li, off in [("RED", 0), ("GREEN", 60), ("BLUE", 120)]:
        g = _session(20, light=li, offset_s=off)
        g.loc[10:, "lit_mean"] *= 1.3  # +26 % log shift, all lights, same moment
        frames.append(g)
    steps = session_steps(pd.concat(frames, ignore_index=True), CFG.qc)
    assert len(steps) == 1 and steps.loc[0, "lights"] == "BLUE,GREEN,RED"
    assert steps.loc[0, "step_ts"] == T0 + pd.Timedelta(seconds=10 * 180)


def test_scene_qc_flags_outliers_only():
    red, blue = _session(30), _session(30, light="BLUE", offset_s=120)
    red.loc[12, "roi_cy"] += 0.08  # RED mask jumps 61 px for one frame -> scene change
    blue.loc[5, "roi_area_frac"] = 0.35  # BLUE mask leaks into the tray -> geometry ignored
    blue.loc[20, "lit_mean"] *= 1.6  # BLUE frame much brighter (lights on) -> scene change
    df = pd.concat([red, blue], ignore_index=True)
    out, steps = scene_qc(df, CFG.qc, ref_lights=["RED"])
    flagged = set(df.loc[out["scene_change"], "frame_id"])
    assert flagged == {"s_RED_12", "s_BLUE_20"}
    assert out.loc[df["frame_id"] == "s_RED_12", "scene_feature"].item() == "cy_px"
    assert steps.empty and (out["scene_segment"] == 0).all()
    assert out["drift_px"].between(0, 70).all()


def test_scene_qc_segments_after_step():
    df = _session(30)
    df.loc[18:, "lit_mean"] *= 1.4
    out, steps = scene_qc(df, CFG.qc, ref_lights=["RED"])
    assert len(steps) == 1
    assert out["scene_segment"].tolist() == [0] * 18 + [1] * 12
    assert not out["scene_change"].any()  # a persistent step is not a transient scene change


def test_roi_qc_table_joins_and_skips_failed_rois():
    df = _session(12)
    df.loc[4, "lit_mean"] *= 2.0
    man = df[["frame_id", "session_id", "ts"]]
    stats = pd.DataFrame(
        {
            "frame_id": df["frame_id"],
            "pixel_light": "RED",
            "global_mean_r": df["lit_mean"],
            "global_mean_g": 0.1,
            "global_mean_b": 0.4,
        }
    )
    roi = df.drop(columns=["session_id", "ts", "light", "lit_mean"]).assign(
        roi_ok=True, lap_var_roi=5.0
    )
    roi.loc[7, "roi_ok"] = False
    out, _ = roi_qc_table(man, stats, roi, CFG)
    assert list(out.columns) == ["frame_id", *ROI_QC_COLS]
    out = out.set_index("frame_id")
    assert out.loc["s_RED_04", "scene_change"]
    assert not out.loc["s_RED_07", "roi_ok"] and np.isnan(out.loc["s_RED_07", "scene_z"])
    assert out["scene_change"].sum() == 1
