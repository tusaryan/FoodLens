"""Start / middle / end comparison per session x light (P3.T2, figure F6).

Frames: the usable frames at ``eda.positions`` (0, 50, 100 % of each session x light's time-sorted
usable frames, the same rule as the QA sheets). Panels show the verified channel inside the
session-ROI bbox with ONE contrast stretch shared by the three frames (percentiles
``eda.stretch_pct`` of their pooled ROI pixels), so a brightness change between panels is real.
The 4th panel is end - start (float) in the ROI with a diverging colormap, range +/- the
``eda.diff_pct`` percentile of |diff|. The meat does not move within a session (LESSONS S04), so
a pixel difference is meaningful; a scene step between start and end is flagged in the title.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from foodlens.config import Config
from foodlens.qc.light_check import CHANNELS
from foodlens.qc.qa_sheets import sheet_frames

TABLE_METRICS = ["roi_median", "glint_count", "glint_area_frac", "rglint_count", "glcm_contrast"]
POS_NAMES = {0.0: "start", 0.5: "mid", 1.0: "end"}


def position_name(pos: float) -> str:
    return POS_NAMES.get(pos, f"p{round(100 * pos)}")


def compare_table(stats: pd.DataFrame, picks: pd.DataFrame, positions: list[float]) -> pd.DataFrame:
    """Wide table per session x light: TABLE_METRICS at each position + % change end vs start,
    and ``step_between`` (the frames lie in different scene segments)."""
    m = picks.merge(stats, on=["frame_id", "session_id", "light"], how="left")
    rows = []
    for (sid, light), g in m.groupby(["session_id", "light"]):
        g = g.set_index("position")
        row = {"session_id": sid, "light": light}
        for pos in positions:
            nm = position_name(pos)
            row[f"{nm}_h"] = float(g.loc[pos, "elapsed_h"])
            row |= {f"{c}_{nm}": float(g.loc[pos, c]) for c in TABLE_METRICS}
        first, last = position_name(positions[0]), position_name(positions[-1])
        for c in TABLE_METRICS:
            a, b = row[f"{c}_{first}"], row[f"{c}_{last}"]
            row[f"{c}_pct_change"] = 100 * (b - a) / abs(a) if a else np.nan
        row["step_between"] = g["scene_segment"].nunique() > 1
        rows.append(row)
    return pd.DataFrame(rows)


def _crop(img: np.ndarray, mask: np.ndarray, margin: float):
    from foodlens.qc.qa_sheets import crop_box

    y0, x0, y1, x1 = crop_box(mask, margin)
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def session_figure(
    picks: pd.DataFrame, sroi: np.ndarray, steps: bool, cfg: Config, figures_dir: Path
) -> Path:
    """F6 for one session: rows = lights, columns = positions + end - start difference."""
    import matplotlib.pyplot as plt

    from foodlens.io.images import load_rgb
    from foodlens.vision.roi import CHANNEL_INDEX, resize_mask
    from foodlens.viz import save_fig

    pos = cfg.eda.positions
    sid = picks["session_id"].iloc[0]
    fig, axes = plt.subplots(3, len(pos) + 1, figsize=(3.2 * (len(pos) + 1), 9.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, light in enumerate(CHANNELS):
        g = picks[picks["light"] == light].sort_values("position")
        if len(g) != len(pos):
            continue
        chans = [
            load_rgb(cfg.root / p, cfg.vision.work_long_side)[..., CHANNEL_INDEX[light]]
            for p in g["path"]
        ]
        mask = resize_mask(sroi, chans[0].shape)
        crops = [_crop(c.astype(np.float32), mask, cfg.qa.crop_margin) for c in chans]
        roi_px = np.concatenate([c[m] for c, m in crops])
        lo, hi = np.percentile(roi_px, cfg.eda.stretch_pct)
        for j, ((c, _), r) in enumerate(zip(crops, g.to_dict("records"), strict=True)):
            axes[i, j].imshow(c, cmap="gray", vmin=lo, vmax=hi)
            axes[i, j].set_title(
                f"{light} {position_name(r['position'])} +{r['elapsed_s'] / 3600:.1f} h", fontsize=8
            )
        diff = np.where(crops[0][1], crops[-1][0] - crops[0][0], np.nan)
        lim = max(float(np.nanpercentile(np.abs(diff), cfg.eda.diff_pct)), 1.0)
        im = axes[i, -1].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[i, -1].set_title(f"{light} end - start (grey levels)", fontsize=8)
        fig.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.02)
    note = " | scene step between start and end: difference includes it" if steps else ""
    fig.suptitle(f"F6 {sid}: start / mid / end with one shared stretch per light{note}", fontsize=9)
    png = save_fig(fig, f"F6_start_mid_end_{sid}", figures_dir, svg=False, dpi=cfg.qa.dpi)
    plt.close(fig)
    return png


def run_compare(man: pd.DataFrame, stats: pd.DataFrame, sessions: pd.DataFrame, cfg: Config):
    """Picks per session, the F6 figures and the comparison table (returned)."""
    from foodlens.vision.roi_batch import read_mask

    figures = cfg.path("reports_dir") / "figures"
    tables = []
    for sid, mp in zip(sessions["session_id"], sessions["sroi_mask_path"], strict=True):
        picks = sheet_frames(man, sid, cfg.eda.positions).assign(session_id=sid)
        t = compare_table(stats, picks, cfg.eda.positions)
        tables.append(t)
        session_figure(picks, read_mask(cfg.root / mp), bool(t["step_between"].any()), cfg, figures)
    return pd.concat(tables, ignore_index=True)
