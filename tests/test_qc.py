"""Tests for qc/light_check.py and qc/frame_qc.py with synthetic images (no real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from foodlens.config import load_config
from foodlens.errors import (
    E_DARK_FRAME,
    E_DECODE,
    E_MIXED_LIGHT,
    E_NO_MEAT_FOUND,
    E_NO_TIME,
    E_SCENE_CHANGE,
    W_BLUR,
    W_LIGHT_MISMATCH,
)
from foodlens.io.images import load_rgb
from foodlens.qc.frame_qc import (
    apply_qc,
    compute_frame_stats,
    laplacian_variance,
    qc_rejects,
    qc_summary,
)
from foodlens.qc.light_check import is_mismatch, light_stats

CFG = load_config().model_copy(update={"n_jobs": 1})


@pytest.mark.parametrize("light", ["RED", "GREEN", "BLUE"])
def test_lit_frame_classified(make_meat_image, light):
    s = light_stats(make_meat_image(light=light), CFG.qc)
    assert s["pixel_light"] == light and s["dom_ratio"] > 0.9


def test_black_frame_is_dark(make_dark_image):
    s = light_stats(make_dark_image(), CFG.qc)
    assert s["pixel_light"] == "DARK" and s["p95_max"] < CFG.qc.dark_p95


def test_equal_channels_is_mixed():
    img = np.full((60, 80, 3), 100, np.uint8)  # white room light: all channels equal
    s = light_stats(img, CFG.qc)
    assert s["pixel_light"] == "MIXED" and s["dom_ratio"] == pytest.approx(1 / 3, abs=1e-3)


def test_mismatch_rules():
    assert is_mismatch("RED", "BLUE")
    assert not is_mismatch("RED", "RED")
    assert not is_mismatch("DARK", "BLUE")  # dark frames are rejects, not mismatches


def test_load_rgb_downscales_and_rejects_garbage(tmp_path, make_meat_image):
    p = tmp_path / "a.jpg"
    Image.fromarray(make_meat_image(size=(400, 300))).save(p)
    assert load_rgb(p, 100).shape == (75, 100, 3)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    with pytest.raises(Exception) as exc:
        load_rgb(bad, 100)
    assert getattr(exc.value, "code", None) == E_DECODE


def _manifest(tmp_path, images: dict[str, np.ndarray], schedule: list[str]) -> pd.DataFrame:
    rows = []
    for i, (name, img) in enumerate(images.items()):
        p = tmp_path / f"{name}.png"
        if img is None:
            p.write_bytes(b"broken")
        else:
            Image.fromarray(img).save(p)
        rows.append(
            {
                "frame_id": f"s__{name}",
                "path": str(p),
                "sha1": name,
                "session_id": "s",
                "ingest_status": "ok",
                "ingest_reason": "",
                "schedule_light": schedule[i],
            }
        )
    rows.append(
        {
            "frame_id": "s__notime",
            "path": "x",
            "sha1": "x",
            "session_id": "s",
            "ingest_status": "reject",
            "ingest_reason": E_NO_TIME,
            "schedule_light": None,
        }
    )
    return pd.DataFrame(rows)


def test_apply_qc_statuses(tmp_path, make_meat_image, make_dark_image):
    imgs = {f"r{i}": make_meat_image("RED", seed=i) for i in range(6)}
    imgs |= {"dark": make_dark_image(), "white": np.full((300, 400, 3), 90, np.uint8)}
    imgs |= {"swap": make_meat_image("GREEN"), "broken": None}
    sched = ["RED"] * 6 + ["BLUE", "RED", "RED", "RED"]
    man = _manifest(tmp_path, imgs, sched)
    stats = compute_frame_stats(man, CFG)
    df = apply_qc(man, stats, CFG).set_index("frame_id")
    assert df.loc["s__dark", "qc_reasons"] == E_DARK_FRAME
    assert df.loc["s__white", "qc_reasons"] == E_MIXED_LIGHT
    assert df.loc["s__broken", "qc_reasons"] == E_DECODE
    assert df.loc["s__notime", "qc_reasons"] == E_NO_TIME
    assert W_LIGHT_MISMATCH in df.loc["s__swap", "qc_reasons"]
    assert df.loc["s__swap", "light"] == "GREEN"  # pixel label wins
    assert pd.isna(df.loc["s__dark", "light"])
    assert (df.loc[[f"s__r{i}" for i in range(6)], "qc_status"] == "ok").all()
    summ = qc_summary(df.reset_index()).set_index("session_id")
    assert summ.loc["s", "reject"] == 4 and summ.loc["TOTAL", "n_frames"] == 11
    assert len(qc_rejects(df.reset_index())) == 4


def test_blur_warning(tmp_path, make_meat_image):
    imgs = {f"r{i}": make_meat_image("RED", seed=i) for i in range(20)}
    blurred = np.full((300, 400, 3), 4, np.uint8)
    blurred[..., 0] = 60  # flat red frame: lit, but almost zero Laplacian variance
    imgs["blurry"] = blurred
    man = _manifest(tmp_path, imgs, ["RED"] * 21)
    df = apply_qc(man, compute_frame_stats(man, CFG), CFG).set_index("frame_id")
    assert df.loc["s__blurry", "qc_status"] == "warn"
    assert W_BLUR in df.loc["s__blurry", "qc_reasons"]


def test_stats_cache_reused(tmp_path, make_meat_image):
    man = _manifest(tmp_path, {"a": make_meat_image("RED")}, ["RED"])
    stats = compute_frame_stats(man, CFG)
    stats.loc[0, "p95_max"] = -1.0  # marker
    again = compute_frame_stats(man, CFG, cached=stats)
    assert again.loc[0, "p95_max"] == -1.0


def _roi_qc(frame_ids: list[str]) -> pd.DataFrame:
    """A hand-made ``roi_qc_table`` result: all ROI ok, no scene change, sharp ROI."""
    n = len(frame_ids)
    return pd.DataFrame(
        {
            "frame_id": frame_ids,
            "roi_ok": True,
            "sroi_mean": 100.0,
            "lap_var_roi": 50.0,
            "scene_z": 1.0,
            "scene_feature": "log_mean",
            "scene_change": False,
            "drift_px": 1.0,
            "scene_segment": 0,
        },
        index=range(n),
    )


def test_scene_change_and_no_roi_rejected(tmp_path, make_meat_image):
    imgs = {f"r{i}": make_meat_image("RED", seed=i) for i in range(21)}
    man = _manifest(tmp_path, imgs, ["RED"] * 21)
    stats = compute_frame_stats(man, CFG)
    roi_qc = _roi_qc([f"s__r{i}" for i in range(21)])
    roi_qc.loc[3, "scene_change"] = True
    roi_qc.loc[5, "roi_ok"] = False
    roi_qc.loc[8, "lap_var_roi"] = 0.5  # ROI blur, although the whole frame is sharp
    df = apply_qc(man, stats, CFG, roi_qc).set_index("frame_id")
    assert df.loc["s__r3", "qc_reasons"] == E_SCENE_CHANGE
    assert df.loc["s__r5", "qc_reasons"] == E_NO_MEAT_FOUND
    assert df.loc["s__r8", "qc_reasons"] == W_BLUR and df.loc["s__r8", "qc_status"] == "warn"
    assert (df.loc[["s__r3", "s__r5"], "qc_pre_roi"] == "ok").all()  # ROI stage keeps them
    assert df.loc[["s__r3", "s__r5"], "light"].isna().all()
    assert df.loc["s__notime", "qc_pre_roi"] == "reject"
    again = apply_qc(df.reset_index(), stats, CFG, roi_qc).set_index("frame_id")
    pd.testing.assert_frame_equal(df, again[df.columns])  # idempotent


def test_blur_falls_back_to_whole_frame_without_full_roi(tmp_path, make_meat_image):
    imgs = {f"r{i}": make_meat_image("RED", seed=i) for i in range(4)}
    man = _manifest(tmp_path, imgs, ["RED"] * 4)
    stats = compute_frame_stats(man, CFG)
    roi_qc = _roi_qc(["s__r0", "s__r1"])  # partial ROI index (a --limit run)
    roi_qc.loc[0, "lap_var_roi"] = 0.01
    df = apply_qc(man, stats, CFG, roi_qc).set_index("frame_id")
    assert W_BLUR not in df.loc["s__r0", "qc_reasons"]  # whole-frame lap_var used
    assert pd.isna(df.loc["s__r2", "roi_ok"]) and df.loc["s__r2", "qc_status"] == "ok"


def test_laplacian_variance_inside_mask():
    rng = np.random.default_rng(0)
    img = np.zeros((60, 80, 3), np.uint8)
    img[:, :40, 0] = rng.integers(0, 255, (60, 40))  # sharp texture on the left
    img[:, 40:, 0] = 100  # flat on the right
    right = np.zeros((60, 80), bool)
    right[5:-5, 45:-5] = True
    assert laplacian_variance(img, 0, right) == 0.0
    assert laplacian_variance(img, 0, ~right) > 1000
