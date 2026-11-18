"""Typed errors for FoodLens. Every error carries a stable string code (e.g. ``E_DARK_FRAME``).

Codes live here and nowhere else, so logs, API responses and tests can match on them.
"""

from __future__ import annotations

# ---- Error codes (single source of truth) ----
E_DECODE = "E_DECODE"  # image file could not be decoded
E_NO_TIME = "E_NO_TIME"  # no capture time in EXIF or file name
E_DARK_FRAME = "E_DARK_FRAME"  # frame is black (LED off)
E_SCENE_CHANGE = "E_SCENE_CHANGE"  # frame differs strongly from its session (lights on, moved)
E_NO_MEAT_FOUND = "E_NO_MEAT_FOUND"  # ROI segmentation found no plausible meat region
E_OOD = "E_OOD"  # input is out of the training distribution
E_TEMP_UNAVAILABLE = "E_TEMP_UNAVAILABLE"  # ambient temperature could not be obtained
E_MODEL_CAPABILITY = "E_MODEL_CAPABILITY"  # model cannot do what was asked
E_CONFIG = "E_CONFIG"  # invalid or missing configuration

ALL_CODES: frozenset[str] = frozenset(
    {
        E_DECODE,
        E_NO_TIME,
        E_DARK_FRAME,
        E_SCENE_CHANGE,
        E_NO_MEAT_FOUND,
        E_OOD,
        E_TEMP_UNAVAILABLE,
        E_MODEL_CAPABILITY,
        E_CONFIG,
    }
)


class FoodLensError(Exception):
    """Base error. ``code`` is an ``E_*`` constant; ``detail`` is a human-readable reason."""

    def __init__(self, code: str, detail: str = "") -> None:
        if code not in ALL_CODES:
            raise ValueError(f"unknown FoodLens error code: {code!r}")
        self.code = code
        self.detail = detail
        super().__init__(f"[{code}] {detail}" if detail else f"[{code}]")


class DataError(FoodLensError):
    """Problems with raw inputs (decode failure, missing timestamp, missing temperature)."""


class QCError(FoodLensError):
    """A frame failed quality control (dark, scene change, no meat found)."""


class ModelError(FoodLensError):
    """Model-level problems (out-of-distribution input, unsupported capability)."""


class ConfigError(FoodLensError):
    """Invalid configuration. Always uses code ``E_CONFIG``."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(E_CONFIG, detail)
