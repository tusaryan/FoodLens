"""Batch ROI over the manifest: per-frame masks, session ROIs, crops and the ROI index (P2.T3).

Layout under ``<data_dir>/interim/roi/<config_hash(roi, vision)>/``:
- ``masks/<frame_id>.png``: per-frame mask (0/255) at work resolution, used for QC only.
- ``session/<session_id>.png``: the session ROI (median of the ``roi.ref_lights`` masks, minus
  holder / tray material, see ``roi.exclude_foreign``), the ROI used for every frame of the session.
- ``crops/<frame_id>.png``: the verified light's channel (8-bit) inside the session ROI bbox.

Per-frame masks are cached (they depend only on the frame and the config); session ROIs and
crops are rebuilt on every run because they depend on the set of frames processed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from foodlens.config import Config
from foodlens.errors import E_NO_MEAT_FOUND, W_ROI_LOW_IOU
from foodlens.io.images import load_rgb
from foodlens.logging_utils import get_logger
from foodlens.qc.frame_qc import laplacian_variance
from foodlens.vision.roi import (
    CHANNEL_INDEX,
    exclude_foreign,
    mask_bbox,
    mask_centroid,
    mask_iou,
    resize_mask,
    segment_roi,
    session_reference_mask,
)

log = get_logger("vision.roi_batch")
ROI_HASH_SECTIONS = ["roi", "vision"]
OUTPUT_COLS = ["roi_iou_session", "crop_path", "sroi_mean", "lap_var_roi"]
FRAME_COLS = ["seg_ok", "seg_reason", "roi_area_frac", "roi_cx", "roi_cy", "work_h", "work_w"]


def read_mask(path: Path) -> np.ndarray:
    """Bool mask from a 0/255 PNG."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 0


def write_mask(mask: np.ndarray, path: Path) -> None:
    """Save a bool mask as a 0/255 8-bit PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def _mask_stats(mask: np.ndarray, ok: bool, reason: str) -> dict[str, Any]:
    cx, cy = mask_centroid(mask)
    h, w = mask.shape
    return dict(zip(FRAME_COLS, [ok, reason, float(mask.mean()), cx, cy, h, w], strict=True))


def frame_mask(path: Path, light: str, cfg: Config, mask_file: Path, force: bool) -> dict:
    """Segment one frame (or reuse its cached mask PNG); return its FRAME_COLS stats.

    ``roi_area_frac`` is a fraction of the frame; ``roi_cx``/``roi_cy`` are fractions (0-1) of
    width / height; ``work_h``/``work_w`` in px. A failed segmentation writes no PNG.
    """
    if mask_file.is_file() and not force:
        return _mask_stats(read_mask(mask_file), True, "")
    img = load_rgb(path, cfg.vision.work_long_side)
    res = segment_roi(img, light, cfg.roi)
    if not res.ok:
        log.warning("roi failed %s (%s): area_frac=%.3f", path.name, light, res.area_frac)
        out = _mask_stats(res.mask, False, res.reason)
        return out | {"roi_area_frac": res.area_frac}
    write_mask(res.mask, mask_file)
    return _mask_stats(res.mask, True, "")


def median_channel(paths: list[Path], light: str, cfg: Config) -> np.ndarray:
    """Per-pixel median (0-255) of ``light``'s channel over up to ``roi.foreign_max_frames``
    evenly spaced frames, at work resolution."""
    idx = np.unique(np.linspace(0, len(paths) - 1, cfg.roi.foreign_max_frames).round().astype(int))
    c = CHANNEL_INDEX[light]
    stack = np.stack([load_rgb(paths[i], cfg.vision.work_long_side)[..., c] for i in idx])
    return np.median(stack, axis=0)


def _foreign_removed(
    g: pd.DataFrame, sid: str, ref: np.ndarray, cfg: Config
) -> tuple[np.ndarray, int]:
    paths = {li: [cfg.root / p for p in g.loc[g["light"] == li, "path"]] for li in ("RED", "BLUE")}
    if not paths["RED"] or not paths["BLUE"]:
        log.warning("session %s: no RED or BLUE frames -> no holder exclusion", sid)
        return ref, 0
    red, blue = (median_channel(paths[li], li, cfg) for li in ("RED", "BLUE"))
    if red.shape != ref.shape or blue.shape != ref.shape:
        raise ValueError(f"session {sid}: median image shape differs from the ROI mask")
    return exclude_foreign(ref, red, blue, cfg.roi)


def session_rois(
    index: pd.DataFrame, cfg: Config, mask_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Session ROI per session (median of the seg-ok ``roi.ref_lights`` masks, holder removed).

    Returns (masks, px removed as foreign material). Sessions without any usable reference frame
    are absent from the result (logged).
    """
    refs: dict[str, np.ndarray] = {}
    removed: dict[str, int] = {}
    for sid, g in index.sort_values("frame_id").groupby("session_id", sort=True):
        ref = g[g["light"].isin(cfg.roi.ref_lights) & g["seg_ok"]]
        if ref.empty:
            log.warning("session %s: no %s frame with a mask -> no ROI", sid, cfg.roi.ref_lights)
            continue
        mask = session_reference_mask([read_mask(mask_dir / f"{f}.png") for f in ref["frame_id"]])
        refs[str(sid)], removed[str(sid)] = _foreign_removed(g, str(sid), mask, cfg)
        log.info("session %s: ROI %d px, %d px foreign removed", sid, mask.sum(), removed[str(sid)])
    return refs, removed


def frame_outputs(
    row: dict[str, Any], ref: np.ndarray | None, cfg: Config, mask_dir: Path, crop_dir: Path
) -> dict[str, Any]:
    """IoU of the frame mask vs the session ROI, the 8-bit crop of the verified channel, and
    inside the session ROI: ``sroi_mean`` (mean, 0-255) and ``lap_var_roi`` (Laplacian variance)."""
    out: dict[str, Any] = dict.fromkeys(OUTPUT_COLS, np.nan) | {"crop_path": ""}
    if ref is None:
        return out
    shape = (int(row["work_h"]), int(row["work_w"]))
    ref_f = resize_mask(ref, shape)
    if row["seg_ok"]:
        out["roi_iou_session"] = mask_iou(read_mask(mask_dir / f"{row['frame_id']}.png"), ref_f)
    y0, x0, y1, x1 = mask_bbox(ref_f)  # type: ignore[misc]  # session ROI is never empty
    img = load_rgb(cfg.root / row["path"], cfg.vision.work_long_side)
    if img.shape[:2] != shape:
        raise ValueError(f"{row['frame_id']}: image {img.shape[:2]} != mask {shape}")
    c = CHANNEL_INDEX[row["light"]]
    out["sroi_mean"] = float(img[..., c][ref_f].mean())
    out["lap_var_roi"] = laplacian_variance(img, c, ref_f)
    crop = img[y0:y1, x0:x1, c]
    crop_file = crop_dir / f"{row['frame_id']}.png"
    crop_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(crop)).save(crop_file)
    out["crop_path"] = str(crop_file.relative_to(cfg.root))
    return out


def session_table(
    refs: dict[str, np.ndarray], removed: dict[str, int], session_dir: Path, cfg: Config
) -> pd.DataFrame:
    """Write the session ROI PNGs; one row per session: area, foreign px removed, bbox, path."""
    rows = []
    for sid, ref in sorted(refs.items()):
        f = session_dir / f"{sid}.png"
        write_mask(ref, f)
        y0, x0, y1, x1 = mask_bbox(ref)  # type: ignore[misc]
        rows.append(
            {
                "session_id": sid,
                "sroi_area_frac": float(ref.mean()),
                "sroi_area_px": int(ref.sum()),
                "sroi_foreign_px": removed.get(sid, 0),
                "sroi_y0": y0,
                "sroi_x0": x0,
                "sroi_y1": y1,
                "sroi_x1": x1,
                "sroi_mask_path": str(f.relative_to(cfg.root)),
            }
        )
    return pd.DataFrame(rows)


def roi_status(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add ``roi_ok`` and ``roi_reason``.

    Reject (``roi_ok=False``, E_NO_MEAT_FOUND): the frame's own segmentation failed, or its
    session has no ROI. Warning (``roi_ok=True``, W_ROI_LOW_IOU): a ``roi.ref_lights`` frame
    whose mask IoU vs the session ROI is below ``roi.min_iou``. Non-reference lights are not
    judged on IoU (their masks leak into the lit tray/holder by design of the scene).
    """
    no_roi = df["sroi_mask_path"].isna()
    failed = ~df["seg_ok"].astype(bool) | no_roi
    low = df["light"].isin(cfg.roi.ref_lights) & (df["roi_iou_session"] < cfg.roi.min_iou)
    df = df.copy()
    df["roi_ok"] = ~failed
    df["roi_reason"] = np.where(failed, E_NO_MEAT_FOUND, np.where(low, W_ROI_LOW_IOU, ""))
    return df
