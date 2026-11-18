"""Per-frame QC flags with a reason per rejection (Phases P1, P2.T4).

Status rules:
- ``reject``: ingest reject (E_NO_TIME, E_NO_SESSION), decode failure (E_DECODE),
  ``DARK`` (E_DARK_FRAME) or ``MIXED`` light (E_MIXED_LIGHT); with ROI QC also no ROI
  (E_NO_MEAT_FOUND) or a transient scene change (E_SCENE_CHANGE, ``qc/scene_qc.py``).
- ``warn``: pixel light differs from the schedule (W_LIGHT_MISMATCH) or the Laplacian variance is
  below the session-light 5th percentile / ``qc.blur_div`` (W_BLUR).
- ``ok`` otherwise.
``qc_pre_roi`` is the status from the non-ROI rules only: the ROI stage selects frames on it, so
re-running ``roi`` after the ROI-based rejects gives the same frame set (idempotent).
The Laplacian variance is ``lap_var_roi`` (inside the session ROI) when every valid-light frame
has one, else the whole-frame ``lap_var`` (before the ROI stage has run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from foodlens.config import Config
from foodlens.errors import (
    E_DARK_FRAME,
    E_DECODE,
    E_MIXED_LIGHT,
    E_NO_MEAT_FOUND,
    E_SCENE_CHANGE,
    W_BLUR,
    W_LIGHT_MISMATCH,
    DataError,
)
from foodlens.io.images import load_rgb
from foodlens.logging_utils import get_logger
from foodlens.qc.light_check import CHANNELS, DARK, MIXED, VALID_LIGHTS, is_mismatch, light_stats
from foodlens.qc.scene_qc import ROI_QC_COLS

log = get_logger("qc.frame_qc")

BLUR_PERCENTILE = 5  # session-light percentile of lap_var used by the BLUR rule
STAT_COLS = [
    "p95_max",
    "global_mean_r",
    "global_mean_g",
    "global_mean_b",
    "lit_frac",
    "dom_channel",
    "dom_ratio",
    "pixel_light",
    "lap_var",
    "decode_error",
]


def laplacian_variance(
    img: np.ndarray, channel: int | None, mask: np.ndarray | None = None
) -> float:
    """Variance of the Laplacian (sharpness) of one channel (or the max channel if None).

    With a bool ``mask`` the Laplacian is computed on the whole plane (so the mask edge adds no
    artificial edge) and its variance is taken over the mask pixels only.
    """
    plane = img.max(axis=2) if channel is None else img[..., channel]
    lap = cv2.Laplacian(plane.astype(np.float32), cv2.CV_32F)
    return float(lap[mask].var() if mask is not None else lap.var())


def frame_stats(path: Path, cfg: Config) -> dict[str, Any]:
    """Light statistics + ``lap_var`` for one frame at ``vision.work_long_side``.

    A decode failure is returned as ``decode_error`` (never raised), so one bad file does not
    stop the stage.
    """
    try:
        img = load_rgb(path, cfg.vision.work_long_side)
    except DataError as exc:
        log.warning("decode failed %s: %s", path, exc.detail)
        return {c: None for c in STAT_COLS} | {"decode_error": exc.code}
    stats = light_stats(img, cfg.qc)
    ch = CHANNELS.index(stats["dom_channel"]) if stats["dom_channel"] else None
    stats["lap_var"] = laplacian_variance(img, ch)
    stats["decode_error"] = ""
    return stats


def compute_frame_stats(
    manifest: pd.DataFrame, cfg: Config, cached: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Stats for every ingest-ok frame, keyed by ``frame_id`` (plus ``sha1``).

    Rows of ``cached`` with the same frame_id and sha1 are reused.
    """
    todo = manifest[manifest["ingest_status"] == "ok"][["frame_id", "path", "sha1"]]
    reuse = pd.DataFrame()
    if cached is not None and len(cached):
        key = todo.merge(cached, on=["frame_id", "sha1"], how="inner")
        reuse = key.drop(columns=["path"])
        todo = todo[~todo["frame_id"].isin(reuse["frame_id"])]
    log.info("frame stats: %d cached, %d to compute", len(reuse), len(todo))
    rows = Parallel(n_jobs=cfg.n_jobs)(
        delayed(frame_stats)(cfg.root / p, cfg) for p in todo["path"]
    )
    fresh = pd.DataFrame(list(rows), columns=STAT_COLS)
    fresh.insert(0, "frame_id", todo["frame_id"].to_numpy())
    fresh.insert(1, "sha1", todo["sha1"].to_numpy())
    out = pd.concat([d for d in (reuse, fresh) if len(d)], ignore_index=True)
    return out.sort_values("frame_id").reset_index(drop=True)


def _blur_source(df: pd.DataFrame) -> str:
    """``lap_var_roi`` if every valid-light frame has it, else ``lap_var``."""
    if "lap_var_roi" not in df:
        return "lap_var"
    valid = df["pixel_light"].isin(VALID_LIGHTS)
    covered = valid.any() and df.loc[valid, "lap_var_roi"].notna().all()
    return "lap_var_roi" if covered else "lap_var"


def _blur_flags(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """True where the Laplacian variance < (session-light p5 / qc.blur_div), valid-light frames."""
    col = _blur_source(df)
    log.info("blur rule uses %s", col)
    valid = df["pixel_light"].isin(VALID_LIGHTS) & df[col].notna()
    lv = df[col].where(valid).astype(float)
    p5 = lv.groupby([df["session_id"], df["pixel_light"]]).transform(
        lambda s: np.nanpercentile(s, BLUR_PERCENTILE) if s.notna().any() else np.nan
    )
    return valid & (lv < p5 / cfg.qc.blur_div)


def _reasons(row: pd.Series) -> tuple[str, str, list[str]]:
    """(pre-ROI status, final status, reasons) of one merged row."""
    rejects, warns = [], []
    if row["ingest_status"] != "ok":
        rejects.append(row["ingest_reason"])
    elif row["decode_error"]:
        rejects.append(E_DECODE)
    elif row["pixel_light"] == DARK:
        rejects.append(E_DARK_FRAME)
    elif row["pixel_light"] == MIXED:
        rejects.append(E_MIXED_LIGHT)
    if row["mismatch"]:
        warns.append(W_LIGHT_MISMATCH)
    pre = "reject" if rejects else ("warn" if warns else "ok")
    if not rejects and row["roi_fail"]:
        rejects.append(E_NO_MEAT_FOUND)
    elif not rejects and row["scene_chg"]:
        rejects.append(E_SCENE_CHANGE)
    if row["blur"]:
        warns.append(W_BLUR)
    status = "reject" if rejects else ("warn" if warns else "ok")
    return pre, status, rejects + warns


QC_COLS = ["light", "qc_status", "qc_reasons", "qc_pre_roi", *ROI_QC_COLS]


def apply_qc(
    manifest: pd.DataFrame, stats: pd.DataFrame, cfg: Config, roi_qc: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Manifest + stats columns + ``pixel_light``, ``light``, ``qc_pre_roi``, ``qc_status``,
    ``qc_reasons`` and, with ``roi_qc`` (``scene_qc.roi_qc_table``), the ROI QC columns.

    Idempotent: all earlier QC columns are dropped first.
    """
    drop = [c for c in STAT_COLS + QC_COLS if c in manifest]
    df = manifest.drop(columns=drop).merge(stats.drop(columns=["sha1"]), on="frame_id", how="left")
    if roi_qc is not None:
        df = df.merge(roi_qc, on="frame_id", how="left")
        df["roi_ok"] = df["roi_ok"].astype("boolean")  # <NA> = frame not in the ROI index
        df["scene_change"] = df["scene_change"].astype("boolean").fillna(False).astype(bool)
    has_roi = "roi_ok" in df
    df["roi_fail"] = (~df["roi_ok"]).fillna(False).astype(bool) if has_roi else False
    df["scene_chg"] = df["scene_change"] if has_roi else False
    df["decode_error"] = df["decode_error"].fillna("")
    df["mismatch"] = [
        is_mismatch(p, s) for p, s in zip(df["pixel_light"], df["schedule_light"], strict=True)
    ]
    df["blur"] = _blur_flags(df, cfg)
    status_reasons = df.apply(_reasons, axis=1)
    df["qc_pre_roi"] = [p for p, _, _ in status_reasons]
    df["qc_status"] = [s for _, s, _ in status_reasons]
    df["qc_reasons"] = [";".join(r) for _, _, r in status_reasons]
    df["light"] = df["pixel_light"].where(
        df["pixel_light"].isin(VALID_LIGHTS) & (df["qc_status"] != "reject")
    )
    for fid, reasons in zip(df["frame_id"], df["qc_reasons"], strict=True):
        if reasons:
            log.debug("qc %s: %s", fid, reasons)
    return df.drop(columns=["mismatch", "blur", "roi_fail", "scene_chg"])


def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per session x reason counts, plus n frames / ok / warn / reject per session."""
    base = df.groupby(df["session_id"].fillna("unknown"))["qc_status"].value_counts().unstack()
    base = base.reindex(columns=["ok", "warn", "reject"], fill_value=0).fillna(0).astype(int)
    base.insert(0, "n_frames", base.sum(axis=1))
    exploded = df.assign(reason=df["qc_reasons"].str.split(";")).explode("reason")
    exploded = exploded[exploded["reason"].fillna("") != ""]
    counts = pd.crosstab(exploded["session_id"].fillna("unknown"), exploded["reason"])
    out = base.join(counts, how="left").fillna(0)
    out.loc["TOTAL"] = out.sum(axis=0)
    out = out.astype(int)
    out["usable"] = out["ok"] + out["warn"]
    return out.rename_axis("session_id").reset_index()


def qc_rejects(df: pd.DataFrame) -> pd.DataFrame:
    """frame_id, path and reasons of every rejected frame."""
    rej = df[df["qc_status"] == "reject"]
    return rej[["frame_id", "path", "session_id", "schedule_light", "qc_reasons"]].reset_index(
        drop=True
    )
