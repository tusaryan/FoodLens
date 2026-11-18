"""Meat-only crops (S08, user lesson: crop to the meat BEFORE any analysis).

Every feature since P2 is already computed only on pixels inside the meat masks, so the background
never entered a feature. This module makes that explicit and visible as standard preprocessing:
each frame is cut to the bounding box of its triplet's FULL meat mask (lit + shadowed side, from
``vision.triplet_roi``) plus ``qa.crop_pad_px``, and every pixel outside the mask is set to 0. The
crops feed the CNN track (P7A-P7D) and the visual review; ``crop_index.parquet`` lists them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from foodlens.config import Config, config_hash
from foodlens.logging_utils import get_logger

log = get_logger("vision.crop")
CROP_HASH_SECTIONS = ["roi", "vision", "eda", "features", "qa"]


@dataclass(frozen=True)
class Crop:
    """``image``: masked RGB crop (uint8, outside-meat = 0); ``mask``: bool crop of the mask;
    ``bbox``: (top, left, bottom, right) in the work-resolution frame (bottom, right exclusive)."""

    image: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


def crop_to_mask(img: np.ndarray, mask: np.ndarray, pad: int) -> Crop:
    """Crop ``img`` (H, W, 3) to the bbox of ``mask`` (H, W) + ``pad`` px; zero outside the mask.
    Raises ``ValueError`` on an empty mask or a shape mismatch."""
    if img.shape[:2] != mask.shape:
        raise ValueError(f"image {img.shape[:2]} and mask {mask.shape} differ")
    if not mask.any():
        raise ValueError("empty mask: nothing to crop")
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    h, w = mask.shape
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, h)
    left, right = max(cols[0] - pad, 0), min(cols[-1] + 1 + pad, w)
    m = mask[top:bottom, left:right]
    out = img[top:bottom, left:right].copy()
    out[~m] = 0
    return Crop(out, m, (int(top), int(left), int(bottom), int(right)))


def _triplet_crops(rows: list[dict], out_dir: Path, cfg: Config) -> list[dict]:
    """Masks from the triplet's RED / BLUE frames, then one PNG crop per frame."""
    from PIL import Image

    from foodlens.io.images import load_rgb
    from foodlens.vision.triplet_roi import triplet_masks

    r0 = rows[0]
    red = load_rgb(cfg.root / r0["red_path"], cfg.vision.work_long_side)
    blue = None
    if isinstance(r0["blue_path"], str):
        blue = load_rgb(cfg.root / r0["blue_path"], cfg.vision.work_long_side)[..., 2]
        blue = blue if blue.shape == red.shape[:2] else None
    tm = triplet_masks(red, blue, cfg)
    out = []
    for r in rows:
        base = {"frame_id": r["frame_id"], "session_id": r["session_id"], "light": r["light"]}
        if not tm.ok:
            log.warning("crops: frame %s rejected: %s", r["frame_id"], tm.reason)
            out.append(base | {"crop_reason": tm.reason})
            continue
        img = red if r["frame_id"] == r0["red_id"] else load_rgb(
            cfg.root / r["path"], cfg.vision.work_long_side
        )  # fmt: skip
        if img.shape[:2] != tm.full.shape:
            log.warning("crops: frame %s rejected: size differs from its RED frame", r["frame_id"])
            out.append(base | {"crop_reason": "E_NO_MEAT_FOUND"})
            continue
        c = crop_to_mask(img, tm.full, cfg.qa.crop_pad_px)
        path = out_dir / r["session_id"] / f"{r['frame_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(c.image).save(path)
        out.append(base | {"crop_reason": "", "crop_path": str(path.relative_to(cfg.root)),
                           "bbox": list(c.bbox), "meat_px": int(c.mask.sum()),
                           "lit_frac": float(tm.lit.sum() / tm.full.sum())})  # fmt: skip
    return out


def run_crops(cfg: Config, limit: int | None, force: bool) -> pd.DataFrame:
    """Crop every usable frame (``--limit``: first N frames); returns the crop index."""
    from joblib import Parallel, delayed

    from foodlens.eda.run import usable_frames
    from foodlens.features.run import mask_sources
    from foodlens.stages import _write_parquet

    frames = usable_frames(cfg)
    if limit is not None:
        frames = frames.head(limit)
    out_dir = cfg.path("data_dir") / "interim" / "crops" / config_hash(cfg, CROP_HASH_SECTIONS)
    index_path = out_dir / "crop_index.parquet"
    old = pd.read_parquet(index_path) if index_path.is_file() and not force else None
    src = mask_sources(frames, cfg)
    done = set() if old is None else set(old.loc[old["crop_reason"] == "", "frame_id"])
    todo = src[~src["frame_id"].isin(done)]
    groups = [g.to_dict("records") for _, g in todo.groupby(["session_id", "triplet"])]
    log.info("crops: %d frames to crop, %d cached", len(todo), len(done))
    res = Parallel(n_jobs=cfg.n_jobs)(delayed(_triplet_crops)(g, out_dir, cfg) for g in groups)
    new = pd.DataFrame([row for rows in res for row in rows])
    keep = old[old["frame_id"].isin(done)] if old is not None else None
    index = pd.concat([d for d in (keep, new) if d is not None and len(d)], ignore_index=True)
    index = index[index["frame_id"].isin(src["frame_id"])].sort_values("frame_id")
    _write_parquet(index.reset_index(drop=True), index_path, cfg)
    return index


def pick_review(index: pd.DataFrame, reviewed: set[str], n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` crops never reviewed before: the smallest and largest meat area (outliers:
    the likeliest segmentation errors) plus one random frame per session; column ``reason``."""
    pool = index[(index["crop_reason"] == "") & ~index["frame_id"].isin(reviewed)]
    if pool.empty:
        return pool.assign(reason=pd.Series(dtype=str))
    ext = pd.concat([pool.nsmallest(1, "meat_px").assign(reason="smallest_meat"),
                     pool.nlargest(1, "meat_px").assign(reason="largest_meat")])  # fmt: skip
    rest = pool[~pool["frame_id"].isin(ext["frame_id"])]
    rnd = rest.groupby("session_id").sample(n=1, random_state=seed).assign(reason="random")
    return pd.concat([ext, rnd]).drop_duplicates("frame_id").head(n)


def review_sheet(cfg: Config, index: pd.DataFrame, tag: str) -> tuple[Path, pd.DataFrame]:
    """Contact sheet original (bbox in yellow) vs meat-only crop for a NEW sample of frames;
    appends the sample to ``tables/visual_review_log.csv``. Returns (figure path, sample)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from foodlens.eda.visual_sample import reviewed_ids
    from foodlens.io.images import load_rgb
    from foodlens.stages import read_manifest, tables_dir
    from foodlens.viz import contrast_stretch, save_fig

    tables = tables_dir(cfg)
    sample = pick_review(index, reviewed_ids(tables), cfg.qa.crop_sample, cfg.seed)
    man = read_manifest(cfg).set_index("frame_id")
    fig, axes = plt.subplots(len(sample), 2, figsize=(7, 2.6 * max(len(sample), 1)), squeeze=False)
    for (ax0, ax1), (_, r) in zip(axes, sample.iterrows(), strict=True):
        img = contrast_stretch(load_rgb(cfg.root / man.at[r["frame_id"], "path"],
                                        cfg.vision.work_long_side))  # fmt: skip
        top, left, bottom, right = r["bbox"]
        ax0.imshow(img)
        ax0.add_patch(plt.Rectangle((left, top), right - left, bottom - top, fill=False,
                                    edgecolor="yellow", lw=1))  # fmt: skip
        ax1.imshow(contrast_stretch(np.asarray(Image.open(cfg.root / r["crop_path"]))))
        eh = man.at[r["frame_id"], "elapsed_s"] / 3600.0
        ax0.set_title(f"{r['frame_id']} {r['light']} {eh:.1f} h ({r['reason']})", fontsize=6)
        ax1.set_title(f"meat-only crop, lit {100 * r['lit_frac']:.0f} %", fontsize=6)
        ax0.axis("off")
        ax1.axis("off")
    fig.tight_layout()
    png = save_fig(fig, f"crops_sample_{tag}", cfg.path("reports_dir") / "figures", svg=False,
                   dpi=cfg.qa.dpi)  # fmt: skip
    plt.close(fig)
    eh = sample["frame_id"].map(man["elapsed_s"]) / 3600.0
    log_rows = sample.assign(elapsed_h=eh, tag=tag)[
        ["frame_id", "session_id", "light", "elapsed_h", "reason", "tag"]]  # fmt: skip
    log_path = tables / "visual_review_log.csv"
    log_rows.to_csv(log_path, mode="a", header=not log_path.is_file(), index=False)
    return png, sample
