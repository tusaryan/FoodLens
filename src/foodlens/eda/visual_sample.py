"""A NEW varied frame sample for visual (by-eye) review each session (P3, user request S06).

Scripts measure every frame; only a small sample is looked at by eye. To see the real variety
of the data instead of the same fixed frames every session, the sample is:
- random: per session, one random frame from each time third (random light), and
- outliers: the frames that deviate most from their own session x light trend (robust z of the
  residual from a centred rolling median, max over ``OUTLIER_FEATURES``), and
- special: frames in a W_BLUR warning and the first frame of each light after a scene step.
Frames already reviewed (``tables/qa_sheet_frames.csv`` and ``tables/visual_review_log.csv``)
are excluded, so every session reviews new frames. The sample is appended to the review log.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from foodlens.config import Config

OUTLIER_FEATURES = ["roi_median", "glint_density", "rglint_density", "glcm_contrast"]
MAD_SCALE = 1.4826  # MAD -> standard deviation for normal data
LOG_COLS = ["frame_id", "session_id", "light", "elapsed_h", "reason", "tag"]


def trend_outlier_score(stats: pd.DataFrame, window: int, floor: float) -> pd.Series:
    """Max over OUTLIER_FEATURES of the robust z of the RELATIVE residual from the centred rolling
    median (residual / |rolling median|), per frame. The MAD is floored at ``floor`` (fraction):
    smooth series (GLCM contrast MAD ~0.2 %) would otherwise flag sub-percent wiggles (S06)."""
    scores = []
    for _, g in stats.sort_values("ts").groupby(["session_id", "light"]):
        z = []
        for f in OUTLIER_FEATURES:
            x = g[f].astype(float)
            med = x.rolling(window, center=True, min_periods=1).median()
            # divisor >= 10 % of the series' typical level (zero-glint stretches have median 0)
            r = (x - med) / med.abs().clip(lower=max(0.1 * float(np.nanmedian(x.abs())), 1e-6))
            mad = MAD_SCALE * float(np.nanmedian(np.abs(r - np.nanmedian(r))))
            z.append(r.abs() / max(mad, floor))
        scores.append(pd.concat(z, axis=1).max(axis=1))
    return pd.concat(scores).reindex(stats.index).fillna(0.0)


def reviewed_ids(tables: Path) -> set[str]:
    """frame_ids already looked at: QA sheet panels and earlier visual samples."""
    ids: set[str] = set()
    for name in ("qa_sheet_frames.csv", "visual_review_log.csv"):
        f = tables / name
        if f.is_file():
            ids |= set(pd.read_csv(f)["frame_id"])
    return ids


def pick_sample(
    stats: pd.DataFrame, exclude: set[str], cfg: Config, n_outliers: int
) -> pd.DataFrame:
    """Rows of ``stats`` chosen for review with a ``reason`` column (random / outlier / special)."""
    rng = np.random.default_rng(cfg.seed)
    pool = stats[~stats["frame_id"].isin(exclude)].copy()
    pool["score"] = trend_outlier_score(pool, cfg.eda.rolling_window, cfg.eda.outlier_floor)
    picks: list[pd.DataFrame] = []
    for _, g in pool.groupby("session_id", sort=True):
        third = np.minimum((3 * g["elapsed_h"] / g["elapsed_h"].max()).astype(int), 2)
        for t in range(3):
            cand = g[third == t]
            if len(cand):
                picks.append(cand.iloc[[rng.integers(len(cand))]].assign(reason="random"))
    special = pool["qc_reasons"].fillna("").str.contains("W_BLUR")
    after = pool[pool["scene_segment"] > 0].sort_values("ts").groupby(["session_id", "light"])
    picks.append(pool[special].assign(reason="special: W_BLUR"))
    picks.append(after.head(1).assign(reason="special: first after scene step"))
    chosen = pd.concat(picks)
    rest = pool[~pool["frame_id"].isin(chosen["frame_id"])].sort_values("score", ascending=False)
    per_session = max(1, int(np.ceil(n_outliers / max(pool["session_id"].nunique(), 1))))
    rest = rest.groupby("session_id").head(per_session).head(n_outliers)
    out = pd.concat([chosen, rest.assign(reason=[f"outlier z={s:.1f}" for s in rest["score"]])])
    return out.drop_duplicates("frame_id").reset_index(drop=True)


def render_sample(
    sample: pd.DataFrame,
    masks: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    name: str,
    ncols: int = 4,
) -> Path:
    """Contact sheet of ``sample`` (rows with path, light, ...): lit ROI yellow, full meat cyan,
    glint centroids of the absolute rule (lime +) and the relative rule (magenta o).

    ``masks``: session -> (lit ROI, full meat), work resolution. Returns the PNG path.
    """
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi
    from skimage.segmentation import find_boundaries

    from foodlens.io.images import load_rgb
    from foodlens.qc.qa_sheets import crop_box
    from foodlens.vision.glints import detect_glints, detect_glints_relative
    from foodlens.vision.roi import CHANNEL_INDEX, resize_mask
    from foodlens.viz import contrast_stretch, save_fig

    nrows = int(np.ceil(len(sample) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), sample.to_dict("records"), strict=False):
        ch = load_rgb(cfg.root / r["path"], cfg.vision.glint_long_side)[
            ..., CHANNEL_INDEX[r["light"]]
        ]
        lit, full = (resize_mask(m, ch.shape) for m in masks[r["session_id"]])
        y0, x0, y1, x1 = crop_box(full, cfg.qa.crop_margin)
        vis = np.repeat(contrast_stretch(ch[y0:y1, x0:x1])[..., None], 3, axis=2)
        for m, rgb in ((full, (0, 255, 255)), (lit, (255, 255, 0))):
            vis[ndi.binary_dilation(find_boundaries(m[y0:y1, x0:x1]), iterations=1)] = rgb
        ax.imshow(vis)
        ga, gr = detect_glints(ch, lit, cfg.glint), detect_glints_relative(ch, lit, cfg.glint)
        for p, mk, col in ((gr.props_df, "o", "magenta"), (ga.props_df, "+", "lime")):
            ax.scatter(
                p["centroid_x"] - x0,
                p["centroid_y"] - y0,
                s=6,
                marker=mk,
                **({"facecolors": "none", "edgecolors": col} if mk == "o" else {"c": col}),
                linewidths=0.4,
            )
        ax.set_title(
            f"{r['session_id']} {r['light']} +{r['elapsed_h']:.2f} h\n{r['reason']} | "
            f"abs {len(ga.props_df)} / rel {len(gr.props_df)} glints",
            fontsize=7,
        )
    fig.suptitle(
        "Visual sample: lit ROI yellow, full meat cyan; glints abs (lime +) / rel (magenta o)",
        fontsize=9,
    )
    png = save_fig(fig, name, cfg.path("reports_dir") / "figures", svg=False, dpi=cfg.qa.dpi)
    plt.close(fig)
    return png
