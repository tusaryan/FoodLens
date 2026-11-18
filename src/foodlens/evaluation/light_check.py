"""Light-ablation sanity check (P6.T2, S09): is a light's signal the meat or the lamp/setup?

Per session and light: log-linear slope (% per hour) of the meat ROI mean and of the lit
non-meat reference (``ref_mean``: tray / holder / walls). A light whose meat darkens while its
reference stays flat carries a meat signal, not LED drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foodlens.config import Config

USABLE = ("ok", "warn")


def _slope_pct_per_h(t: np.ndarray, v: np.ndarray) -> float:
    return float(100 * np.polyfit(t, np.log(v), 1)[0]) if len(v) > 2 else np.nan


def meat_vs_reference_trends(cfg: Config) -> pd.DataFrame:
    """One row per session x light: n frames, hours, meat and reference slope (%/h)."""
    from foodlens.features.run import features_frame_path

    cols = ["session_id", "light", "elapsed_h", "roi_mean", "ref_mean", "qc_status"]
    f = pd.read_parquet(features_frame_path(cfg), columns=cols)
    f = f[f["qc_status"].isin(USABLE)].dropna()
    f = f[(f["roi_mean"] > 0) & (f["ref_mean"] > 0)]
    rows = []
    for (s, light), d in f.groupby(["session_id", "light"], sort=True):
        t = d["elapsed_h"].to_numpy()
        rows.append({"session_id": s, "light": light, "n": len(d), "hours": float(t.max()),
                     "meat_pct_per_h": _slope_pct_per_h(t, d["roi_mean"].to_numpy()),
                     "ref_pct_per_h": _slope_pct_per_h(t, d["ref_mean"].to_numpy())})  # fmt: skip
    return pd.DataFrame(rows)
