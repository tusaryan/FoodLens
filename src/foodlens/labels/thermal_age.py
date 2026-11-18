"""Temperature-equivalent hours at T_ref (Ratkowsky square-root model), P5.T1.

Bacterial growth rate r follows sqrt(r) = b (T - T_min) (Ratkowsky et al. 1982; tested on 28
poultry-spoilage datasets, -2..25 °C, 1984). Relative to T_ref the rate is
``((T - T_min) / (T_ref - T_min))**2``, and T <= T_min contributes 0. One hour at 33 °C with
T_min = -8 °C counts as (41/33)**2 ≈ 1.544 h at 25 °C; one hour at 4 °C as ≈ 0.13 h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def relative_rate(temp_c: np.ndarray | float, t_min: float, t_ref: float) -> np.ndarray:
    """Growth rate at ``temp_c`` (°C) relative to ``t_ref`` (°C); 0 at or below ``t_min``."""
    if t_min >= t_ref:
        raise ValueError("t_min must be < t_ref")
    t = np.asarray(temp_c, dtype=float)
    return (np.clip(t - t_min, 0.0, None) / (t_ref - t_min)) ** 2


def equiv_hours(
    ts: pd.Series, temp_c: pd.Series, t_min: float, t_ref: float, offset_c: float = 0.0
) -> np.ndarray:
    """Cumulative equivalent hours at ``t_ref`` for ONE time-sorted series.

    ``ts``: timestamps; ``temp_c``: °C at each timestamp (``offset_c`` added, e.g. indoor offset).
    Trapezoid rule on the relative rate between consecutive samples; the first value is 0.
    Raises ``ValueError`` on NaN temperature or unsorted time (never silently filled).
    """
    t = pd.to_datetime(ts).to_numpy()
    if len(t) and (np.diff(t).astype("timedelta64[ns]").astype(np.int64) < 0).any():
        raise ValueError("timestamps must be sorted")
    temp = np.asarray(temp_c, dtype=float) + offset_c
    if np.isnan(temp).any():
        raise ValueError("temperature has NaN values")
    if len(t) == 0:
        return np.zeros(0)
    dt_h = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 3.6e12
    rate = relative_rate(temp, t_min, t_ref)
    steps = 0.5 * (rate[1:] + rate[:-1]) * dt_h
    return np.concatenate([[0.0], np.cumsum(steps)])


def clock_hours_for(equiv_h: float, temp_c: float, t_min: float, t_ref: float) -> float:
    """Clock hours at constant ``temp_c`` that accumulate ``equiv_h`` equivalent hours
    (inf when the temperature is at or below ``t_min``: no growth)."""
    rate = float(relative_rate(temp_c, t_min, t_ref))
    return float("inf") if rate == 0 else equiv_h / rate
