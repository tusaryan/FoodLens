"""Tests for the P0 skeleton: fixtures, logging, viz, CLI, size checker."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from foodlens.cli import app  # noqa: E402
from foodlens.logging_utils import get_logger  # noqa: E402
from foodlens.viz import LIGHT_COLORS, save_fig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTED = {"ingest", "qc", "roi", "eda-sheets"}
STAGES = ["ingest", "qc", "roi", "features", "labels", "train", "evaluate", "predict", "eda-sheets"]


@pytest.mark.parametrize(("light", "ch"), [("RED", 0), ("GREEN", 1), ("BLUE", 2)])
def test_meat_image_lit_channel(make_meat_image, light, ch):
    img = make_meat_image(light=light, size=(80, 60), glints=3, seed=1)
    assert img.shape == (60, 80, 3) and img.dtype == np.uint8
    means = img.reshape(-1, 3).mean(axis=0)
    assert means.argmax() == ch
    assert (img == 255).all(axis=2).sum() > 0  # saturated glints present


def test_meat_image_deterministic(make_meat_image):
    assert np.array_equal(make_meat_image(seed=3), make_meat_image(seed=3))


def test_dark_image(make_dark_image):
    img = make_dark_image(size=(50, 40))
    assert img.shape == (40, 50, 3)
    assert np.percentile(img.max(axis=2), 95) < 3


def test_get_logger_namespace(monkeypatch):
    monkeypatch.setenv("FOODLENS_LOG", "DEBUG")
    log = get_logger("qc.frame_qc")
    assert log.name == "foodlens.qc.frame_qc"
    assert logging.getLogger("foodlens").level == logging.DEBUG


def test_get_logger_bad_level(monkeypatch):
    monkeypatch.setenv("FOODLENS_LOG", "LOUD")
    with pytest.raises(ValueError):
        get_logger("x")


def test_save_fig_writes_png_and_svg(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color=LIGHT_COLORS["RED"])
    png = save_fig(fig, "unit", figures_dir=tmp_path)
    plt.close(fig)
    assert png.exists() and (tmp_path / "unit.svg").exists()


def test_cli_lists_all_stages():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for stage in STAGES:
        assert stage in result.output


@pytest.mark.parametrize("stage", [s for s in STAGES if s not in IMPLEMENTED])
def test_cli_stub_accepts_common_options(stage, tmp_path, write_tmp_config):
    # tmp config: if a stage gets implemented but stays listed here, it cannot touch real data/
    cfg = str(write_tmp_config(tmp_path))
    result = CliRunner().invoke(app, [stage, "--config", cfg, "--limit", "20", "--force"])
    assert result.exit_code == 0
    assert "not implemented yet" in result.output


def test_size_checker_flags_long_module(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        check_sizes = importlib.import_module("check_sizes")
    finally:
        sys.path.pop(0)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "big.py").write_text("x = 1\n" * 301)
    body = "".join("    y = 1\n" for _ in range(55))
    (tmp_path / "src" / "ok.py").write_text(f"def f():\n{body}")
    errors, warnings = check_sizes.check(tmp_path)
    assert len(errors) == 1 and "big.py" in errors[0]
    assert len(warnings) == 1 and "f()" in warnings[0]
