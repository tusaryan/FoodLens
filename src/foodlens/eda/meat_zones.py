"""Meat zones per session: the lit ROI (the feature ROI from P2) and the shadowed meat (P3.T1).

Open question 6: the session ROI keeps only the well-lit 75-90 % of the meat. To show what the
dim, shadowed side does over time WITHOUT changing the feature ROI, the EDA measures a second
zone. Full meat = pixels of the smoothed session-median RED image above ``eda.shadow_min_level``
(the box background is ~0-4), opened with a disk of ``eda.shadow_open_r`` px (cuts thin lit
tray / glass strips), components that touch the lit ROI, holes filled, holder / tray removed
with the same BLUE/RED rule as the ROI (``vision.roi.exclude_foreign``), eroded by
``roi.erode_px`` (drops the meat rim), united with the lit ROI. Shadow zone = full meat minus
the lit ROI. All masks are at work resolution (``vision.work_long_side``).

Reference region (S06 control): per session x light, lit NON-meat pixels (tray, holder, box
walls): outside the full-meat mask grown by ``eda.ref_gap_px`` and brighter than
``eda.ref_min_level`` in that light's session-median image. If the LEDs or the camera drifted,
this region would change too; the meat-to-reference ratio cancels such common changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology

from foodlens.config import Config
from foodlens.logging_utils import get_logger
from foodlens.vision.roi import exclude_foreign, largest_component
from foodlens.vision.roi_batch import median_channel, read_mask, write_mask

log = get_logger("eda.meat_zones")


def full_meat_mask(
    red_med: np.ndarray, blue_med: np.ndarray, lit: np.ndarray, cfg: Config
) -> np.ndarray:
    """Bool full-meat mask (lit ROI + shadowed meat) from session-median RED/BLUE images."""
    if not red_med.shape == blue_med.shape == lit.shape:
        raise ValueError("red_med, blue_med and lit must have the same shape")
    r = filters.gaussian(red_med.astype(np.float32), sigma=cfg.roi.blur_sigma, preserve_range=True)
    cand = r > cfg.eda.shadow_min_level
    if cfg.eda.shadow_open_r:
        cand = ndi.binary_opening(cand, structure=morphology.disk(cfg.eda.shadow_open_r))
    lab, _ = ndi.label(cand)
    ids = np.unique(lab[lit])
    full = ndi.binary_fill_holes(np.isin(lab, ids[ids > 0]))
    full, _ = exclude_foreign(full, red_med, blue_med, cfg.roi)
    if cfg.roi.erode_px:
        full = ndi.binary_erosion(full, structure=morphology.disk(cfg.roi.erode_px))
    return largest_component(full | lit)


def session_zones(
    frames: pd.DataFrame, sessions: pd.DataFrame, cfg: Config, out_dir: Path
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """(lit ROI, shadow zone) masks per session + a table of their areas.

    ``frames``: usable frames with session_id, light, path; ``sessions``: ``roi_sessions.csv``
    rows (session_id, sroi_mask_path). Full-meat masks are written to ``out_dir/<session>.png``.
    """
    zones: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for sid, mp in zip(sessions["session_id"], sessions["sroi_mask_path"], strict=True):
        g = frames[frames["session_id"] == sid]
        lit = read_mask(cfg.root / mp)
        paths = {
            li: [cfg.root / p for p in g.loc[g["light"] == li, "path"]] for li in ("RED", "BLUE")
        }
        if not paths["RED"] or not paths["BLUE"]:
            log.warning("session %s: no usable RED or BLUE frame -> empty shadow zone", sid)
            full = lit.copy()
        else:
            red, blue = (median_channel(paths[li], li, cfg) for li in ("RED", "BLUE"))
            full = full_meat_mask(red, blue, lit, cfg)
        write_mask(full, out_dir / f"{sid}.png")
        shadow = full & ~lit
        zones[str(sid)] = (lit, shadow)
        rows.append(
            {
                "session_id": sid,
                "lit_px": int(lit.sum()),
                "shadow_px": int(shadow.sum()),
                "shadow_frac_of_meat": float(shadow.sum() / full.sum()),
                "full_mask_path": str((out_dir / f"{sid}.png").relative_to(cfg.root)),
            }
        )
        log.info(
            "session %s: shadow zone %.1f %% of the meat",
            sid,
            100 * rows[-1]["shadow_frac_of_meat"],
        )
    return zones, pd.DataFrame(rows)


def reference_masks(
    frames: pd.DataFrame, full: dict[str, np.ndarray], cfg: Config
) -> dict[tuple[str, str], np.ndarray]:
    """(session, light) -> bool reference mask (see module doc); work resolution."""
    out = {}
    for (sid, light), g in frames.groupby(["session_id", "light"]):
        if sid not in full:
            continue
        med = median_channel([cfg.root / p for p in g["path"]], light, cfg)
        far = ~ndi.binary_dilation(full[sid], iterations=cfg.eda.ref_gap_px)
        ref = far & (med > cfg.eda.ref_min_level)
        if ref.sum() < cfg.roi.foreign_min_area_px:
            log.warning("session %s %s: reference region too small (%d px)", sid, light, ref.sum())
            continue
        out[(str(sid), str(light))] = ref
    return out


def zones_figure(
    zones: dict[str, tuple[np.ndarray, np.ndarray]], images: dict[str, np.ndarray], figures_dir, dpi
) -> Path:
    """One panel per session: gamma-stretched RED image, lit ROI (yellow), shadow zone (cyan).

    ``images``: session -> RED uint8 image (work resolution). Saved as ``qa_zones.png``.
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    from foodlens.viz import save_fig

    fig, axes = plt.subplots(1, len(zones), figsize=(3.6 * len(zones), 4), squeeze=False)
    for ax, (sid, (lit, shadow)) in zip(axes[0], sorted(zones.items()), strict=True):
        img = images[sid].astype(np.float32)
        vis = np.clip(np.sqrt(img / max(np.percentile(img[lit], 99), 1.0)) * 255, 0, 255)
        vis = np.repeat(vis.astype(np.uint8)[..., None], 3, axis=2)
        vis[ndi.binary_dilation(find_boundaries(lit), iterations=1)] = (255, 255, 0)
        vis[ndi.binary_dilation(find_boundaries(lit | shadow), iterations=1)] = (0, 255, 255)
        ys, xs = np.nonzero(lit | shadow)
        pad = 30
        ax.imshow(
            vis[max(ys.min() - pad, 0) : ys.max() + pad, max(xs.min() - pad, 0) : xs.max() + pad]
        )
        frac = shadow.sum() / (lit | shadow).sum()
        ax.set_title(f"{sid}: shadow = {100 * frac:.0f} % of meat", fontsize=8)
        ax.axis("off")
    fig.suptitle(
        "Lit ROI (yellow, feature ROI) and full meat incl. shadowed side (cyan); RED, sqrt stretch",
        fontsize=9,
    )
    png = save_fig(fig, "qa_zones", figures_dir, svg=False, dpi=dpi)
    plt.close(fig)
    return png
