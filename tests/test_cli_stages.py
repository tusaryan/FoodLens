"""End-to-end CLI test of `foodlens ingest` + `foodlens qc` on a tiny synthetic session."""

from __future__ import annotations

import pandas as pd
from PIL import Image
from typer.testing import CliRunner

from foodlens.cli import app

SESSION = {
    "exp1_D1A": {
        "experiment": "EXP-1",
        "folder": "EXP-1_x/exp1_D1A",
        "temp": {"mode": "fixed", "value_c": 30.0},
    }
}


def _save(img, path, when: str) -> None:
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = when
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, exif=exif)


def test_ingest_then_qc(tmp_path, write_tmp_config, make_meat_image, make_dark_image):
    s = tmp_path / "experiment" / "EXP-1_x" / "exp1_D1A"
    lights = ["RED", "GREEN", "BLUE"] * 2
    for i, light in enumerate(lights):
        img = make_dark_image() if i == 2 else make_meat_image(light=light, seed=i)
        _save(img, s / f"IMG_20260828_12{i:02d}00.jpg", f"2026:08:28 12:{i:02d}:00")
    cfg = str(write_tmp_config(tmp_path, sessions=SESSION))
    runner = CliRunner()
    r1 = runner.invoke(app, ["ingest", "--config", cfg])
    assert r1.exit_code == 0, r1.output
    assert "STAGE DONE ingest rows=6" in r1.output
    r2 = runner.invoke(app, ["qc", "--config", cfg])
    assert r2.exit_code == 0, r2.output
    assert "STAGE DONE qc rows=6" in r2.output
    man = pd.read_parquet(tmp_path / "data" / "processed" / "manifest.parquet")
    assert (man["temp_c"] == 30.0).all()
    assert man["qc_status"].value_counts().to_dict() == {"ok": 5, "reject": 1}
    assert man.loc[man["qc_status"] == "reject", "schedule_light"].item() == "BLUE"
    for f in ["tables/inventory.csv", "tables/qc_summary.csv", "qc_rejects.csv"]:
        assert (tmp_path / "reports" / f).is_file()
    assert (tmp_path / "reports" / "figures" / "temperature_sessions.png").is_file()
