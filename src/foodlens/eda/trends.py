"""Time-series helpers and trend statistics (P3.T3, used by T5).

- ``step_correct``: removes a persistent scene step (``scene_segment``, e.g. D2A 12:59:32, a
  camera/white-balance change during the 92 s gap) from MULTIPLICATIVE intensity features: each
  later segment is divided by the gain median(first ``k`` frames after) / median(last ``k``
  frames before), per session x light x feature. Glint / texture features are not rescaled.
- ``relative_to_first``: value / median of the first ``eda.rel_base_n`` frames (the mentor's
  "first = 1.0" normalisation).
- ``trend_table``: per session x light x feature, Spearman rho (+ p) and the Theil-Sen slope per
  hour with its 95 % CI, also as % of the start level per hour.
- ``mixed_table``: pooled ``value ~ hours`` with a random intercept per session (statsmodels
  MixedLM): does the feature change consistently across pieces?
Frames within a session are autocorrelated, so per-session p-values are optimistic; the pooled
model and the sign consistency across sessions (T5) are the honest summaries.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

NON_FEATURES = {
    "frame_id", "session_id", "light", "ts", "elapsed_s", "elapsed_h", "scene_segment",
    "qc_status", "qc_reasons", "temp_c", "glint_thr", "variant",
}  # fmt: skip
MULTIPLICATIVE_PREFIXES = ("roi_", "shd_", "full_")
NOT_MULTIPLICATIVE = {"roi_sat_frac", "shd_rglint_density"}
KEYS = ["session_id", "light"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns of a frame_stats-like table (meta columns excluded)."""
    num = df.select_dtypes(include="number").columns
    return [c for c in num if c not in NON_FEATURES]


def multiplicative_columns(cols: list[str]) -> list[str]:
    """Intensity columns that scale with the light / camera gain (step-corrected)."""
    return [
        c for c in cols if c.startswith(MULTIPLICATIVE_PREFIXES) and c not in NOT_MULTIPLICATIVE
    ]


def step_correct(df: pd.DataFrame, cols: list[str], k: int) -> pd.DataFrame:
    """Copy of ``df`` with ``cols`` divided by the scene-step gain(s) (see module doc)."""
    out = df.copy()
    for idx in df.groupby(KEYS).groups.values():
        order = df.loc[idx].sort_values("ts").index
        seg = df.loc[order, "scene_segment"]
        levels = sorted(seg.dropna().unique())
        for prev, nxt in zip(levels, levels[1:], strict=False):
            before = out.loc[order[seg == prev], cols].tail(k).median()
            after = out.loc[order[seg == nxt], cols].head(k).median()
            ratio = (after / before).replace([np.inf, -np.inf, 0], np.nan).fillna(1.0)
            later = order[seg >= nxt]
            out.loc[later, cols] = out.loc[later, cols] / ratio
    return out


def relative_to_first(df: pd.DataFrame, cols: list[str], n: int) -> pd.DataFrame:
    """Copy with ``cols`` divided by the median of the first ``n`` frames per session x light
    (NaN where that base is 0 or NaN)."""
    out = df.sort_values("ts").copy()
    base = out.groupby(KEYS)[cols].transform(lambda s: s.head(n).median())
    out[cols] = out[cols] / base.where(base != 0)
    return out


def rolling_median(df: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    """Copy with ``cols`` replaced by a centred rolling median over ``window`` frames."""
    out = df.sort_values("ts").copy()
    roll = out.groupby(KEYS)[cols].transform(
        lambda s: s.rolling(window, center=True, min_periods=1).median()
    )
    out[cols] = roll
    return out


def series_trend(hours: np.ndarray, y: np.ndarray, min_n: int) -> dict[str, float]:
    """Spearman rho / p and Theil-Sen slope (per hour, 95 % CI) of one series; NaN if too short
    or constant."""
    ok = np.isfinite(y)
    h, v = hours[ok], y[ok]
    nan = dict.fromkeys(["rho", "p", "slope_h", "slope_lo", "slope_hi", "base"], np.nan)
    if len(v) < min_n or np.unique(v).size < 2:
        return nan | {"n": float(len(v))}
    rho, p = sps.spearmanr(h, v)
    slope, _, lo, hi = sps.theilslopes(v, h, alpha=0.95)
    base = float(np.median(v[: min(3, len(v))]))
    return {"n": float(len(v)), "rho": float(rho), "p": float(p), "slope_h": float(slope),
            "slope_lo": float(lo), "slope_hi": float(hi), "base": base}  # fmt: skip


def trend_table(df: pd.DataFrame, cols: list[str], min_n: int) -> pd.DataFrame:
    """One row per session x light x feature (see ``series_trend``) + ``slope_pct_h``
    (slope as % of the start level per hour)."""
    rows = []
    for (sid, light), g in df.sort_values("ts").groupby(KEYS):
        h = g["elapsed_h"].to_numpy(float)
        for c in cols:
            r = series_trend(h, g[c].to_numpy(float), min_n)
            rows.append({"session_id": sid, "light": light, "feature": c} | r)
    t = pd.DataFrame(rows)
    t["slope_pct_h"] = 100 * t["slope_h"] / t["base"].where(t["base"] != 0).abs()
    return t


def mixed_slope(df: pd.DataFrame, col: str) -> dict[str, object]:
    """MixedLM ``col ~ elapsed_h`` with a random intercept per session: slope per hour, SE, 95 %
    CI, p, n, convergence and any warning text."""
    import statsmodels.formula.api as smf

    d = df[["session_id", "elapsed_h", col]].dropna().rename(columns={col: "y"})
    out: dict[str, object] = {"n_obs": len(d), "n_sessions": d["session_id"].nunique()}
    if out["n_sessions"] < 2 or d["y"].nunique() < 2:
        return out | {"converged": False, "warning": "too few sessions or constant"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = smf.mixedlm("y ~ elapsed_h", d, groups=d["session_id"]).fit()
    ci = res.conf_int().loc["elapsed_h"]
    msgs = sorted({str(w.message).splitlines()[0][:80] for w in caught})
    return out | {
        "slope_h": float(res.params["elapsed_h"]), "se": float(res.bse["elapsed_h"]),
        "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "p": float(res.pvalues["elapsed_h"]),
        "converged": bool(res.converged), "warning": "; ".join(msgs),
    }  # fmt: skip


def mixed_table(df: pd.DataFrame, cols: list[str], scale: str) -> pd.DataFrame:
    """``mixed_slope`` per light x feature; ``scale`` labels the input (raw / relative)."""
    rows = [
        {"light": light, "feature": c, "scale": scale} | mixed_slope(g, c)
        for light, g in df.groupby("light")
        for c in cols
    ]
    return pd.DataFrame(rows)
