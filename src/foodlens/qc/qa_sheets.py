"""Visual QA contact sheets: session ROI outline + glint centroids on real frames (P2.T5).

One sheet per session: rows = lights (RED, GREEN, BLUE), columns = the usable frames at
``qa.positions`` (fractions of each light's time-sorted usable frames, start .. end). Each panel
shows the contrast-stretched verified channel at glint resolution (``vision.glint_long_side``),
cropped to the session-ROI bbox plus ``qa.crop_margin`` so the holder / tray stay visible; the
session ROI (the feature ROI) is outlined in yellow, glint centroids are cyan circles.
Glints are specular reflections (moisture), NOT bacteria.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries

from foodlens.config import Config
from foodlens.io.images import load_rgb
from foodlens.logging_utils import get_logger
from foodlens.qc.light_check import CHANNELS
from foodlens.vision.glints import detect_glints
from foodlens.vision.roi import CHANNEL_INDEX, mask_bbox, resize_mask
from foodlens.viz import contrast_stretch, save_fig

log = get_logger("qc.qa_sheets")
OUTLINE_RGB = (255, 255, 0)  # yellow session-ROI outline
GLINT_COLOR = "cyan"
OUTLINE_PX = 2  # outline thickness (px at glint resolution)
PANEL_IN = 3.2  # panel size (inches)
SHEET_COLS = [
    "session_id",
    "light",
    "position",
    "frame_id",
    "elapsed_h",
    "glint_count",
    "glint_thr",
]


def sheet_frames(man: pd.DataFrame, session_id: str, positions: list[float]) -> pd.DataFrame:
    """Usable frames (qc_status ok/warn) of one session at ``positions`` per light.

    Returns light, position, frame_id, path, elapsed_s (sorted by light order, position).
    """
    usable = man[(man["session_id"] == session_id) & man["qc_status"].isin(["ok", "warn"])]
    rows = []
    for light in CHANNELS:
        g = usable[usable["light"] == light].sort_values("ts")
        if g.empty:
            log.warning("session %s: no usable %s frame for the QA sheet", session_id, light)
            continue
        for pos in positions:
            r = g.iloc[int(round(pos * (len(g) - 1)))]
            rows.append([light, pos, r["frame_id"], r["path"], r["elapsed_s"]])
    return pd.DataFrame(rows, columns=["light", "position", "frame_id", "path", "elapsed_s"])


def crop_box(mask: np.ndarray, margin: float) -> tuple[int, int, int, int]:
    """(y0, x0, y1, x1) of the mask bbox grown by ``margin`` x its size per side, clipped."""
    y0, x0, y1, x1 = mask_bbox(mask)  # type: ignore[misc]  # session ROI is never empty
    dy, dx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    h, w = mask.shape
    return max(y0 - dy, 0), max(x0 - dx, 0), min(y1 + dy, h), min(x1 + dx, w)


def render_panel(ax, frame: pd.Series, sroi: np.ndarray, cfg: Config) -> dict[str, float]:
    """Draw one frame on ``ax``; return its glint count and threshold (0-255)."""
    img = load_rgb(cfg.root / frame["path"], cfg.vision.glint_long_side)
    ch = img[..., CHANNEL_INDEX[frame["light"]]]
    mask = resize_mask(sroi, ch.shape)
    res = detect_glints(ch, mask, cfg.glint)
    y0, x0, y1, x1 = crop_box(mask, cfg.qa.crop_margin)
    vis = np.repeat(contrast_stretch(ch)[..., None], 3, axis=2)
    edge = dilation(find_boundaries(mask, mode="inner"), disk(OUTLINE_PX))
    vis[edge] = OUTLINE_RGB
    ax.imshow(vis[y0:y1, x0:x1])
    p = res.props_df
    ax.scatter(
        p["centroid_x"] - x0,
        p["centroid_y"] - y0,
        s=cfg.qa.marker_px**2 / 4,
        facecolors="none",
        edgecolors=GLINT_COLOR,
        linewidths=0.5,
    )
    hours = frame["elapsed_s"] / 3600.0
    ax.set_title(f"{frame['light']} +{hours:.1f} h | {len(p)} glints | thr {res.threshold:.0f}")
    ax.title.set_fontsize(7)
    ax.axis("off")
    return {"elapsed_h": hours, "glint_count": float(len(p)), "glint_thr": res.threshold}


def session_sheet(
    man: pd.DataFrame, session_id: str, sroi: np.ndarray, cfg: Config, figures_dir: Path
) -> pd.DataFrame:
    """Render and save ``qa_roi_<session_id>.png``; one SHEET_COLS row per panel."""
    import matplotlib.pyplot as plt

    frames = sheet_frames(man, session_id, cfg.qa.positions)
    ncol = len(cfg.qa.positions)
    fig, axes = plt.subplots(3, ncol, figsize=(PANEL_IN * ncol, PANEL_IN * 3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    rows = []
    for _, f in frames.iterrows():
        ax = axes[CHANNELS.index(f["light"]), cfg.qa.positions.index(f["position"])]
        stats = render_panel(ax, f, sroi, cfg)
        rows.append({"session_id": session_id, **f.drop(["path", "elapsed_s"]).to_dict(), **stats})
    fig.suptitle(
        f"{session_id}: session ROI (yellow), glint centroids (cyan); "
        "positions = start / 25 / 50 / 75 % / end of the usable frames",
        fontsize=9,
    )
    save_fig(fig, f"qa_roi_{session_id}", figures_dir, svg=False, dpi=cfg.qa.dpi)
    plt.close(fig)
    return pd.DataFrame(rows, columns=SHEET_COLS)
