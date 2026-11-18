"""Shared pytest fixtures: tiny synthetic images (never the real dataset)."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

CHANNEL = {"RED": 0, "GREEN": 1, "BLUE": 2}
BG_LEVEL = 4  # dark-box background (0-255), similar to real frames
MEAT_LEVEL = 120  # meat brightness in the lit channel (0-255)
NOISE_SD = 2.0
GLINT_RADIUS = 2


def _make_meat_image(
    light: str = "RED",
    size: tuple[int, int] = (400, 300),
    glints: int = 10,
    seed: int = 0,
    level: float = MEAT_LEVEL,
) -> np.ndarray:
    """Synthetic frame: dark background, bright ellipse in ``light``'s channel, saturated dots.

    ``size`` is (width, height) in px; ``level`` is the meat brightness (0-255).
    Returns an RGB uint8 array of shape (height, width, 3).
    """
    if light not in CHANNEL:
        raise ValueError(f"light must be one of {list(CHANNEL)}")
    w, h = size
    rng = np.random.default_rng(seed)
    img = rng.normal(BG_LEVEL, NOISE_SD, size=(h, w, 3))
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx, ry, rx = h / 2, w / 2, h * 0.3, w * 0.3
    ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    c = CHANNEL[light]
    img[..., c][ellipse] = rng.normal(level, NOISE_SD * 3, size=int(ellipse.sum()))
    inside = np.argwhere(ellipse)
    for y, x in inside[rng.choice(len(inside), size=glints, replace=False)]:
        dot = (yy - y) ** 2 + (xx - x) ** 2 <= GLINT_RADIUS**2
        img[dot] = 255  # specular glint saturates all channels
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_dark_image(size: tuple[int, int] = (400, 300), seed: int = 0) -> np.ndarray:
    """Synthetic black frame (LED off): near-zero noise. Shape (height, width, 3), uint8."""
    w, h = size
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0.5, 0.5, size=(h, w, 3)), 0, 255).astype(np.uint8)


@pytest.fixture
def make_meat_image() -> Callable[..., np.ndarray]:
    return _make_meat_image


@pytest.fixture
def make_dark_image() -> Callable[..., np.ndarray]:
    return _make_dark_image


def _write_tmp_config(root: Path, sessions: dict | None = None, **overrides: dict) -> Path:
    """Copy configs/default.yaml into ``root/configs`` (so relative paths resolve under ``root``).

    ``sessions`` replaces samples.yaml sessions; ``overrides`` replace whole top-level sections.
    """
    cfg_dir = root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    data["n_jobs"] = 1
    for key, value in overrides.items():
        data[key] = {**data[key], **value} if isinstance(value, dict) else value
    (cfg_dir / "default.yaml").write_text(yaml.safe_dump(data))
    if sessions is None:
        shutil.copy(REPO / "configs" / "samples.yaml", cfg_dir / "samples.yaml")
    else:
        (cfg_dir / "samples.yaml").write_text(yaml.safe_dump({"sessions": sessions}))
    return cfg_dir / "default.yaml"


@pytest.fixture
def write_tmp_config() -> Callable[..., Path]:
    return _write_tmp_config
