"""Tests for foodlens.errors."""

from __future__ import annotations

import pytest

from foodlens import errors
from foodlens.errors import (
    E_CONFIG,
    E_DARK_FRAME,
    ConfigError,
    DataError,
    FoodLensError,
    ModelError,
    QCError,
)


def test_error_has_code_and_detail():
    e = QCError(E_DARK_FRAME, "p95=0.0 < 3")
    assert e.code == E_DARK_FRAME
    assert e.detail == "p95=0.0 < 3"
    assert "E_DARK_FRAME" in str(e)
    assert isinstance(e, FoodLensError)


@pytest.mark.parametrize("cls", [DataError, QCError, ModelError])
def test_subclasses_are_foodlens_errors(cls):
    assert issubclass(cls, FoodLensError)


def test_config_error_uses_e_config():
    e = ConfigError("bad value")
    assert e.code == E_CONFIG
    assert isinstance(e, FoodLensError)


def test_unknown_code_rejected():
    with pytest.raises(ValueError):
        FoodLensError("E_MADE_UP")


def test_all_codes_listed():
    expected = {
        "E_DECODE",
        "E_NO_TIME",
        "E_DARK_FRAME",
        "E_SCENE_CHANGE",
        "E_NO_MEAT_FOUND",
        "E_OOD",
        "E_TEMP_UNAVAILABLE",
        "E_MODEL_CAPABILITY",
        "E_CONFIG",
        "E_NO_SESSION",
        "E_MIXED_LIGHT",
        "E_DUPLICATE",
    }
    assert expected == errors.ALL_CODES
    for code in expected:
        assert getattr(errors, code) == code
    assert {"W_LIGHT_MISMATCH", "W_BLUR", "W_ROI_LOW_IOU"} == errors.WARNING_CODES
    assert not errors.WARNING_CODES & errors.ALL_CODES
