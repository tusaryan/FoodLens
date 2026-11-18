"""`foodlens roi` end to end on a tiny synthetic session, plus the roi_status rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from typer.testing import CliRunner

from foodlens.cli import app
from foodlens.config import load_config
from foodlens.errors import E_NO_MEAT_FOUND, W_ROI_LOW_IOU
from foodlens.vision.roi_batch import read_mask, roi_status

BLUE_MEAT_LEVEL = 45  # real meat is ~0.3-0.45 x as bright under BLUE as under RED

SESSION = {
    "exp1_D1A": {
        "experiment": "EXP-1",
        "folder": "EXP-1_x/exp1_D1A",
        "temp": {"mode": "fixed", "value_c": 30.0},
    }
}


def _save(img: np.ndarray, path, when: str) -> None:
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = when
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, exif=exif)


def test_roi_stage(tmp_path, write_tmp_config, make_meat_image, make_dark_image):
    s = tmp_path / "experiment" / "EXP-1_x" / "exp1_D1A"
    for i, light in enumerate(["RED", "GREEN", "BLUE"] * 2):
        level = BLUE_MEAT_LEVEL if light == "BLUE" else 120
        img = make_dark_image() if i == 2 else make_meat_image(light=light, seed=i, level=level)
        _save(img, s / f"IMG_20260828_12{i:02d}00.jpg", f"2026:08:28 12:{i:02d}:00")
    cfg = str(write_tmp_config(tmp_path, sessions=SESSION))
    runner = CliRunner()
    assert runner.invoke(app, ["roi", "--config", cfg]).exit_code != 0  # needs ingest + qc first
    for cmd in ["ingest", "qc"]:
        r = runner.invoke(app, [cmd, "--config", cfg])
        assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["roi", "--config", cfg])
    assert r.exit_code == 0, r.output
    assert "STAGE DONE roi rows=5" in r.output  # the dark frame is not processed
    idx = pd.read_parquet(tmp_path / "data" / "processed" / "roi_index.parquet")
    assert idx["roi_ok"].all()
    assert (idx.loc[idx["light"] == "RED", "roi_iou_session"] > 0.9).all()
    assert idx["sroi_area_frac"].between(0.1, 0.3).all()
    assert (idx["sroi_foreign_px"] == 0).all()  # no holder in the synthetic scene
    for p in idx["crop_path"]:
        crop = np.asarray(Image.open(tmp_path / p))
        assert crop.ndim == 2 and crop.dtype == np.uint8 and crop.mean() > 25
    ref = read_mask(tmp_path / idx["sroi_mask_path"].iloc[0])
    assert ref.sum() == idx["sroi_area_px"].iloc[0]
    for f in ["roi_sessions.csv", "roi_summary.csv"]:
        assert (tmp_path / "reports" / "tables" / f).is_file()
    assert idx["sroi_mean"].between(25, 255).all() and (idx["lap_var_roi"] > 0).all()
    man = pd.read_parquet(tmp_path / "data" / "processed" / "manifest.parquet")
    usable = man[man["qc_pre_roi"] != "reject"]
    assert len(usable) == 5 and usable["roi_ok"].all()  # roi refreshed the manifest QC
    assert {"scene_z", "drift_px", "scene_segment", "lap_var_roi"} <= set(man.columns)
    assert (tmp_path / "reports" / "tables" / "scene_steps.csv").is_file()
    r = runner.invoke(app, ["roi", "--config", cfg])  # idempotent: same frames, same QC
    assert "STAGE DONE roi rows=5" in r.output
    man2 = pd.read_parquet(tmp_path / "data" / "processed" / "manifest.parquet")
    assert man2["qc_reasons"].tolist() == man["qc_reasons"].tolist()
    r = runner.invoke(app, ["eda-sheets", "--config", cfg])
    assert r.exit_code == 0, r.output
    assert "STAGE DONE eda-sheets rows=15" in r.output  # 3 lights x 5 positions
    assert (tmp_path / "reports" / "figures" / "qa_roi_exp1_D1A.png").is_file()
    sheet = pd.read_csv(tmp_path / "reports" / "tables" / "qa_sheet_frames.csv")
    assert (sheet["glint_count"] > 0).all() and (sheet["glint_thr"] >= 40).all()
    r2 = runner.invoke(app, ["roi", "--config", cfg, "--limit", "2"])  # cached masks reused
    assert "STAGE DONE roi rows=2" in r2.output


def test_roi_status_rules():
    cfg = load_config()
    df = pd.DataFrame(
        {
            "light": ["RED", "RED", "BLUE", "RED", "RED"],
            "seg_ok": [True, True, True, False, True],
            "roi_iou_session": [0.95, 0.3, 0.3, np.nan, np.nan],
            "sroi_mask_path": ["a.png", "a.png", "a.png", "a.png", None],
        }
    )
    out = roi_status(df, cfg)
    assert out["roi_ok"].tolist() == [True, True, True, False, False]
    assert out["roi_reason"].tolist() == ["", W_ROI_LOW_IOU, "", E_NO_MEAT_FOUND, E_NO_MEAT_FOUND]
