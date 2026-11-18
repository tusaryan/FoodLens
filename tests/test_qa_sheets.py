"""Tests for qc/qa_sheets.py frame selection and crop box (rendering is covered end to end in
tests/test_roi_stage.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from foodlens.qc.qa_sheets import crop_box, sheet_frames


def test_sheet_frames_positions_and_usable_only():
    n = 9
    man = pd.DataFrame(
        {
            "frame_id": [f"f{i}" for i in range(n)] + ["bad"],
            "path": "x",
            "session_id": "s",
            "light": ["RED"] * n + ["RED"],
            "qc_status": ["ok"] * n + ["reject"],
            "ts": pd.date_range("2026-08-28", periods=n + 1, freq="3min"),
            "elapsed_s": np.arange(n + 1) * 180.0,
        }
    )
    out = sheet_frames(man, "s", [0.0, 0.5, 1.0])
    assert out["frame_id"].tolist() == ["f0", "f4", "f8"]  # the reject is never shown
    assert sheet_frames(man, "other", [0.0]).empty


def test_crop_box_margin_and_clip():
    mask = np.zeros((100, 200), bool)
    mask[40:60, 50:150] = True
    assert crop_box(mask, 0.25) == (35, 25, 65, 175)
    assert crop_box(mask, 5.0) == (0, 0, 100, 200)
