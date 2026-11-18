"""Per-frame ROI statistics for every usable frame (P3.T1; P4 reuses these functions).

For one frame, on the verified light's channel at glint resolution (``vision.glint_long_side``):
- ``roi_*``: intensity statistics (0-255) inside the lit session ROI, glint pixels excluded
  (pixels of either glint rule), plus ``roi_sat_frac`` (share of ROI pixels >= ``glint.sat_level``).
- ``glint_*``: the P2 glint features (absolute floor ``glint.abs_min``) and ``glint_thr``.
- ``rglint_*``: the same features with the illumination-relative rule (open question 7).
- ``glcm_*``: GLCM texture of the lit ROI (``vision.texture``).
- ``shd_*``: the shadowed-meat zone (``eda.meat_zones``, open question 6): intensity median / mean
  / p90 and relative-glint density; ``full_*``: median / mean over lit + shadow meat.
Statistics are computed at glint resolution so glint pixels can be excluded exactly; medians and
means do not depend on the resolution. Glints are specular reflections (moisture), NOT bacteria.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from foodlens.config import Config
from foodlens.io.images import load_rgb
from foodlens.vision.glints import detect_glints, detect_glints_relative, glint_features
from foodlens.vision.roi import CHANNEL_INDEX, resize_mask
from foodlens.vision.texture import glcm_features

INTENSITY_KEYS = ("median", "p10", "p25", "p75", "p90", "iqr", "mean", "std")
ID_COLS = [
    "frame_id",
    "session_id",
    "light",
    "ts",
    "elapsed_s",
    "scene_segment",
    "qc_status",
    "qc_reasons",
    "temp_c",
]
PROP_KEEP = ["frame_id", "variant", "area", "centroid_y", "centroid_x", "mean_intensity"]


def intensity_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """``<prefix><key>`` for INTENSITY_KEYS of a 1-D array of pixel values (0-255); NaN if empty."""
    if values.size == 0:
        return {f"{prefix}{k}": float("nan") for k in INTENSITY_KEYS}
    v = values.astype(np.float32)
    p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    vals = [p50, p10, p25, p75, p90, p75 - p25, v.mean(), v.std()]
    return {f"{prefix}{k}": float(x) for k, x in zip(INTENSITY_KEYS, vals, strict=True)}


def shadow_stats(ch: np.ndarray, shadow: np.ndarray, cfg: Config) -> dict[str, float]:
    """``shd_median`` / ``shd_mean`` / ``shd_p90`` (0-255, relative glints excluded) and
    ``shd_rglint_density`` (count per 10^4 px) of the bool ``shadow`` zone; NaN if it is tiny."""
    nan = float("nan")
    if shadow.sum() < cfg.glint.min_area_px * 10:
        return {"shd_median": nan, "shd_mean": nan, "shd_p90": nan, "shd_rglint_density": nan}
    rel = detect_glints_relative(ch, shadow, cfg.glint)
    v = ch[shadow & (rel.labels == 0)].astype(np.float32)
    feats = glint_features(rel, int(shadow.sum()))
    return {
        "shd_median": float(np.median(v)) if v.size else nan,
        "shd_mean": float(v.mean()) if v.size else nan,
        "shd_p90": float(np.percentile(v, 90)) if v.size else nan,
        "shd_rglint_density": feats["glint_density"],
    }


def channel_features(
    ch: np.ndarray, lit: np.ndarray, shadow: np.ndarray, cfg: Config
) -> tuple[dict[str, float], pd.DataFrame]:
    """All T1 features of one uint8 channel with bool lit / shadow masks (same shape).

    Returns (feature dict, glint property table with a ``variant`` column: abs / rel).
    """
    if not lit.any():
        raise ValueError("lit ROI mask is empty")
    area = int(lit.sum())
    g_abs = detect_glints(ch, lit, cfg.glint)
    g_rel = detect_glints_relative(ch, lit, cfg.glint)
    clean = lit & (g_abs.labels == 0) & (g_rel.labels == 0)
    out: dict[str, float] = intensity_stats(ch[clean], "roi_")
    out["roi_sat_frac"] = float((ch[lit] >= cfg.glint.sat_level).mean())
    out |= glint_features(g_abs, area) | {"glint_thr": g_abs.threshold}
    out |= {f"r{k}": v for k, v in glint_features(g_rel, area).items()}
    out |= glcm_features(ch, lit, cfg.vision)
    out |= shadow_stats(ch, shadow, cfg)
    full = clean | shadow
    out |= {
        "full_median": float(np.median(ch[full])),
        "full_mean": float(ch[full].astype(np.float32).mean()),
    }
    props = pd.concat(
        [g_abs.props_df.assign(variant="abs"), g_rel.props_df.assign(variant="rel")],
        ignore_index=True,
    )
    return out, props


def frame_row(
    row: dict[str, Any],
    lit_small: np.ndarray,
    shadow_small: np.ndarray,
    cfg: Config,
    ref_small: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load one frame at glint resolution, resize the work-res masks, compute its features."""
    img = load_rgb(cfg.root / row["path"], cfg.vision.glint_long_side)
    ch = img[..., CHANNEL_INDEX[row["light"]]]
    lit = resize_mask(lit_small, ch.shape)
    shadow = resize_mask(shadow_small, ch.shape) & ~lit
    feats, props = channel_features(ch, lit, shadow, cfg)
    feats["ref_mean"] = float("nan")
    if ref_small is not None:
        feats["ref_mean"] = float(ch[resize_mask(ref_small, ch.shape)].astype(np.float32).mean())
    feats["roi_mean_rel_ref"] = (
        feats["roi_mean"] / feats["ref_mean"] if feats["ref_mean"] else np.nan
    )
    ids = {k: row[k] for k in ID_COLS}
    props = props.assign(frame_id=row["frame_id"])
    return ids | {"elapsed_h": row["elapsed_s"] / 3600.0} | feats, props[PROP_KEEP]


def compute_frame_stats(
    frames: pd.DataFrame,
    zones: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    refs: dict[tuple[str, str], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(frame_stats, glint_props) for the usable ``frames`` (manifest rows), in parallel.

    Frames of a session without zones are skipped (the caller logs them).
    """
    from joblib import Parallel, delayed

    refs = refs or {}
    todo = [r for r in frames.to_dict("records") if r["session_id"] in zones]
    res = Parallel(n_jobs=cfg.n_jobs)(
        delayed(frame_row)(r, *zones[r["session_id"]], cfg, refs.get((r["session_id"], r["light"])))
        for r in todo
    )
    stats = pd.DataFrame([r[0] for r in res])
    props = pd.concat([r[1] for r in res], ignore_index=True) if res else pd.DataFrame()
    return stats.sort_values(["session_id", "ts"]).reset_index(drop=True), props
