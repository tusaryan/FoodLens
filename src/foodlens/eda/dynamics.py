"""Glint dynamics over time per session x light (P3.T4, the team's droplet hypothesis; F7).

Consecutive usable frames of the same light (~3 min apart) are matched with
``vision.glint_dynamics.match_glints`` (``glint.match_px``, ``glint.merge_area_ratio``):
births, deaths, merges, splits, persisting glints, as rates per hour. Pattern labels compare the
trend of the glint count with the trend of the median glint size (Spearman, ``eda.alpha``):
- fewer + smaller -> "evaporation (fewer, smaller)"
- more + smaller  -> "film breaking (more, smaller)"
- fewer + bigger  -> "merging (fewer, bigger)"
- more + bigger   -> "more and bigger"
These are descriptive patterns of the surface moisture film (glints are specular reflections),
not bacterial claims. Part of the births/deaths are detection noise near the threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from foodlens.config import Config
from foodlens.qc.light_check import CHANNELS
from foodlens.vision.glint_dynamics import match_glints

EVENTS = ["births", "deaths", "merges", "splits", "persist"]
EMPTY = pd.DataFrame(columns=["centroid_y", "centroid_x", "area"])


def pair_events(
    stats: pd.DataFrame, props: pd.DataFrame, variant: str, cfg: Config
) -> pd.DataFrame:
    """One row per consecutive same-light frame pair: EVENTS, n_prev, n_curr, dt_h, t_h."""
    p = props[props["variant"] == variant]
    by_frame = {f: g for f, g in p.groupby("frame_id")}
    rows = []
    for (sid, light), g in stats.sort_values("ts").groupby(["session_id", "light"]):
        ids, hours = g["frame_id"].tolist(), g["elapsed_h"].to_numpy(float)
        for a, b, ha, hb in zip(ids, ids[1:], hours, hours[1:], strict=False):
            pa, pb = by_frame.get(a, EMPTY), by_frame.get(b, EMPTY)
            ev = match_glints(pa, pb, cfg.glint.match_px, cfg.glint.merge_area_ratio)
            rows.append({"session_id": sid, "light": light, "t_h": (ha + hb) / 2,
                         "dt_h": hb - ha, "n_prev": len(pa), "n_curr": len(pb)} | ev)  # fmt: skip
    return pd.DataFrame(rows)


def _trend(h: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y)
    if ok.sum() < 3 or np.unique(y[ok]).size < 2:
        return np.nan, np.nan
    r, p = sps.spearmanr(h[ok], y[ok])
    return float(r), float(p)


def pattern_label(rho_n: float, p_n: float, rho_s: float, p_s: float, alpha: float) -> str:
    """Droplet pattern from the count and size trends (see module doc)."""
    n = 0 if not p_n < alpha else int(np.sign(rho_n))
    s = 0 if not p_s < alpha else int(np.sign(rho_s))
    labels = {(-1, -1): "evaporation (fewer, smaller)", (1, -1): "film breaking (more, smaller)",
              (-1, 1): "merging (fewer, bigger)", (1, 1): "more and bigger",
              (-1, 0): "fewer (size flat)", (1, 0): "more (size flat)",
              (0, -1): "smaller (count flat)", (0, 1): "bigger (count flat)"}  # fmt: skip
    return labels.get((n, s), "no clear trend")


def dynamics_summary(
    events: pd.DataFrame, stats: pd.DataFrame, variant: str, alpha: float
) -> pd.DataFrame:
    """Per session x light: event rates per hour, persistence, count / size trends, pattern."""
    pre = "glint_" if variant == "abs" else "rglint_"
    rows = []
    for (sid, light), e in events.groupby(["session_id", "light"]):
        s = stats[(stats["session_id"] == sid) & (stats["light"] == light)]
        hrs = float(e["dt_h"].sum())
        h = s["elapsed_h"].to_numpy(float)
        rn, pn = _trend(h, s[f"{pre}count"].to_numpy(float))
        rs, ps = _trend(h, s[f"{pre}size_p50"].to_numpy(float))
        row = {"variant": variant, "session_id": sid, "light": light, "n_pairs": len(e)}
        row |= {f"{k}_per_h": float(e[k].sum()) / hrs for k in EVENTS}
        row["persist_frac"] = float(e["persist"].sum() / max(e["n_prev"].sum(), 1))
        row["mean_count"] = float(s[f"{pre}count"].mean())
        row |= {"count_rho": rn, "count_p": pn, "size_rho": rs, "size_p": ps}
        row["merges_minus_splits_per_h"] = row["merges_per_h"] - row["splits_per_h"]
        row["pattern"] = pattern_label(rn, pn, rs, ps, alpha)
        rows.append(row)
    return pd.DataFrame(rows)


def window_sizes(
    stats: pd.DataFrame, props: pd.DataFrame, variant: str, frac: float
) -> pd.DataFrame:
    """Glint areas (px) of the frames in the first / middle / last ``frac`` of each session's
    time, with a ``window`` column (start / mid / end)."""
    s = stats[["frame_id", "session_id", "light", "elapsed_h"]].copy()
    s["t"] = s["elapsed_h"] / s.groupby("session_id")["elapsed_h"].transform("max")
    s["window"] = np.select(
        [s["t"] <= frac, (s["t"] - 0.5).abs() <= frac / 2, s["t"] >= 1 - frac],
        ["start", "mid", "end"],
        default="",
    )
    p = props[props["variant"] == variant].merge(s[s["window"] != ""], on="frame_id")
    return p[["session_id", "light", "window", "area"]]


def dynamics_figure(
    sizes: pd.DataFrame, summary: pd.DataFrame, variant: str, figures_dir: Path
) -> Path:
    """F7: rows = sessions; columns = size histograms (log x) per light at start / mid / end, and
    event rates per hour per light."""
    import matplotlib.pyplot as plt

    from foodlens.viz import LIGHT_COLORS, save_fig

    sessions = sorted(summary["session_id"].unique())
    fig, axes = plt.subplots(len(sessions), 4, figsize=(14, 2.6 * len(sessions)), squeeze=False)
    bins = np.logspace(np.log10(3), np.log10(max(sizes["area"].max(), 10)), 25)
    styles = {"start": ":", "mid": "--", "end": "-"}
    for i, sid in enumerate(sessions):
        for j, light in enumerate(CHANNELS):
            ax = axes[i, j]
            for w, ls in styles.items():
                a = sizes[
                    (sizes["session_id"] == sid)
                    & (sizes["light"] == light)
                    & (sizes["window"] == w)
                ]
                if len(a):
                    ax.hist(a["area"], bins=bins, histtype="step", ls=ls, density=True,
                            color=LIGHT_COLORS[light], label=f"{w} (n={len(a)})")  # fmt: skip
            ax.set_xscale("log")
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)
            ax.set_title(f"{sid} {light}: glint size (px)", fontsize=8)
        sm = summary[summary["session_id"] == sid].set_index("light").reindex(CHANNELS)
        x = np.arange(len(EVENTS) - 1)
        for k, light in enumerate(CHANNELS):
            vals = [sm.at[light, f"{e}_per_h"] for e in EVENTS[:-1]]
            axes[i, 3].bar(x + (k - 1) * 0.27, vals, 0.27, color=LIGHT_COLORS[light])
        axes[i, 3].set_xticks(x, EVENTS[:-1], fontsize=7)
        axes[i, 3].set_title(f"{sid}: events per hour", fontsize=8)
    fig.suptitle(
        f"F7 glint dynamics ({variant} glint rule); glints = moisture reflections", fontsize=9
    )
    fig.tight_layout()
    png = save_fig(fig, f"F7_glint_dynamics_{variant}", figures_dir)
    plt.close(fig)
    return png
