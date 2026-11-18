"""EDA figures: time series (F5) and the signal heat-map (F8). Computation lives elsewhere."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from foodlens.qc.light_check import CHANNELS
from foodlens.viz import LIGHT_COLORS, save_fig

SESSION_STYLES = ["-", "--", "-.", ":", (0, (5, 1))]


def timeseries_figure(
    raw: pd.DataFrame,
    smooth: pd.DataFrame,
    sid: str,
    metrics: list[str],
    step_h: list[float],
    figures_dir: Path,
) -> Path:
    """F5 for one session: rows = lights, columns = metrics. Dots = raw values; line = rolling
    median of the step-corrected values (``smooth``); dashed grey = scene step(s)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(metrics), figsize=(3.3 * len(metrics), 7.5), squeeze=False)
    for i, light in enumerate(CHANNELS):
        r = raw[(raw["session_id"] == sid) & (raw["light"] == light)].sort_values("ts")
        s = smooth[(smooth["session_id"] == sid) & (smooth["light"] == light)].sort_values("ts")
        for j, m in enumerate(metrics):
            ax = axes[i, j]
            ax.plot(r["elapsed_h"], r[m], ".", ms=3, alpha=0.35, color=LIGHT_COLORS[light])
            ax.plot(s["elapsed_h"], s[m], "-", lw=1.4, color=LIGHT_COLORS[light])
            for h in step_h:
                ax.axvline(h, color="grey", ls="--", lw=0.8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(m, fontsize=8)
            if j == 0:
                ax.set_ylabel(light, fontsize=8)
            if i == 2:
                ax.set_xlabel("elapsed (h)", fontsize=8)
    fig.suptitle(
        f"F5 {sid}: dots = raw frames, line = rolling median (step-corrected intensities); "
        "dashed = scene step",
        fontsize=9,
    )
    png = save_fig(fig, f"F5_timeseries_{sid}", figures_dir)
    plt.close(fig)
    return png


def relative_figure(rel: pd.DataFrame, metrics: list[str], figures_dir: Path) -> Path:
    """F5 (pooled): rows = lights, columns = metrics, one line per session of the rolling median
    of value / start value (start = 1.0)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(metrics), figsize=(3.3 * len(metrics), 7.5), squeeze=False)
    sessions = sorted(rel["session_id"].unique())
    for i, light in enumerate(CHANNELS):
        for j, m in enumerate(metrics):
            ax = axes[i, j]
            for k, sid in enumerate(sessions):
                g = rel[(rel["session_id"] == sid) & (rel["light"] == light)].sort_values("ts")
                ls = SESSION_STYLES[k % len(SESSION_STYLES)]
                ax.plot(g["elapsed_h"], g[m], ls=ls, lw=1.2, color=LIGHT_COLORS[light], label=sid)
            ax.axhline(1.0, color="grey", lw=0.6)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"{m} / start", fontsize=8)
            if j == 0:
                ax.set_ylabel(light, fontsize=8)
            if i == 2:
                ax.set_xlabel("elapsed (h)", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(
        "F5 all sessions: relative to the start (first frames = 1.0), rolling median", fontsize=9
    )
    png = save_fig(fig, "F5_relative_all_sessions", figures_dir)
    plt.close(fig)
    return png


def signal_heatmap(rank: pd.DataFrame, figures_dir: Path, top: int = 30) -> Path:
    """F8: mean signed Spearman rho per feature x light, annotated with the sign consistency
    (n same sign / n sessions); the ``top`` features by best |mean rho| are shown."""
    import matplotlib.pyplot as plt

    cols = [c for c in [*CHANNELS, "TRIPLET"] if c in set(rank["light"])]
    best = rank.groupby("feature")["mean_abs_rho"].max().sort_values(ascending=False)
    feats = list(best.index[:top])
    rho = rank.pivot(index="feature", columns="light", values="mean_rho").reindex(feats)[cols]
    cons = rank.pivot(index="feature", columns="light", values="consistency_label")
    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 3, 0.28 * len(feats) + 1.5))
    im = ax.imshow(rho.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(feats)), feats, fontsize=7)
    for y, f in enumerate(feats):
        for x, c in enumerate(cols):
            lab = cons.at[f, c] if f in cons.index and c in cons.columns else ""
            if isinstance(lab, str) and lab:
                ax.text(x, y, lab, ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="mean Spearman rho with hours (over sessions)")
    ax.set_title(
        "F8: which light carries the signal? text = sessions with the majority sign", fontsize=9
    )
    png = save_fig(fig, "F8_signal_heatmap", figures_dir)
    plt.close(fig)
    return png


def finite(x: pd.Series) -> np.ndarray:
    """Finite values of a series as a float array."""
    v = x.to_numpy(float)
    return v[np.isfinite(v)]
