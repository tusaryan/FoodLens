"""Verify which LED (RED/GREEN/BLUE) lit a frame from its pixels, vs the schedule (Phase P1).

Rules (``image-pipeline`` skill, thresholds in ``configs/default.yaml`` ``qc``):
1. p95 of the per-pixel max channel < ``qc.dark_p95`` -> ``DARK`` (LED off).
2. Otherwise take pixels whose max channel > ``qc.min_signal`` ("lit" pixels), compute the mean of
   each channel over them; the dominant channel's share of the sum of the three means is the
   dominance ratio. Ratio >= ``qc.dominance_ratio`` -> that light, else ``MIXED``.
A schedule mismatch is only a warning: the pixel label wins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from foodlens.config import QCConfig

CHANNELS = ("RED", "GREEN", "BLUE")
DARK, MIXED = "DARK", "MIXED"
VALID_LIGHTS = frozenset(CHANNELS)
DARK_PERCENTILE = 95  # percentile of the max channel used for the dark test


def light_stats(img: np.ndarray, qc: QCConfig) -> dict[str, Any]:
    """Per-frame light statistics (pixel values 0-255) and the pixel light label.

    Keys: ``p95_max``, ``global_mean_r|g|b``, ``lit_frac``, ``dom_channel``, ``dom_ratio``,
    ``pixel_light`` (RED/GREEN/BLUE/DARK/MIXED).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {img.shape}")
    px = img.reshape(-1, 3).astype(np.float32)
    mx = px.max(axis=1)
    means = px.mean(axis=0)
    out: dict[str, Any] = {
        "p95_max": float(np.percentile(mx, DARK_PERCENTILE)),
        "global_mean_r": float(means[0]),
        "global_mean_g": float(means[1]),
        "global_mean_b": float(means[2]),
    }
    lit = mx > qc.min_signal
    out["lit_frac"] = float(lit.mean())
    if out["p95_max"] < qc.dark_p95 or not lit.any():
        out.update(dom_channel=None, dom_ratio=float("nan"), pixel_light=DARK)
        return out
    lit_means = px[lit].mean(axis=0)
    dom = int(lit_means.argmax())
    ratio = float(lit_means[dom] / lit_means.sum())
    out["dom_channel"] = CHANNELS[dom]
    out["dom_ratio"] = ratio
    out["pixel_light"] = CHANNELS[dom] if ratio >= qc.dominance_ratio else MIXED
    return out


def is_mismatch(pixel_light: str, schedule_light: str | None) -> bool:
    """True when a valid pixel light differs from the schedule light (DARK/MIXED never count)."""
    return (
        pixel_light in VALID_LIGHTS and schedule_light is not None and pixel_light != schedule_light
    )
