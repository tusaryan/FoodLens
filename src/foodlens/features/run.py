"""Feature stage runner (``foodlens features``, P4.T2-T4).

Outputs:
- ``<data_dir>/processed/features_frame.parquet``: one row per usable frame (schema.FRAME_META +
  frame features), W_BLUR glint/texture -> NaN, brightness step-corrected.
- ``<data_dir>/processed/features_triplet.parquet``: one row per session x triplet.
- ``<data_dir>/interim/features/<hash>/frame_raw.parquet``: per-frame cache (raw extraction),
  reused when the frame id and its mask source frames (``zone_key``) are unchanged.
Masks: per triplet from its own RED / BLUE frames (``vision/triplet_roi.py``, S07), NOT the P2
session ROI, because the meat moves / shrinks in the 16-22 h sessions.
- ``<reports_dir>/tables/feature_dictionary.csv`` (T5), ``feature_qc.csv``,
  ``feature_corr_pairs.csv``, ``feature_counts.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from foodlens.config import Config, config_hash
from foodlens.features.schema import FRAME_META, frame_specs, triplet_meta, triplet_specs
from foodlens.logging_utils import get_logger
from foodlens.stages import _write_parquet, tables_dir

log = get_logger("features.run")
FEATURE_HASH_SECTIONS = ["roi", "glint", "vision", "eda", "features"]
ID_IN = ["frame_id", "session_id", "experiment", "piece_id", "light", "ts", "elapsed_s",
         "temp_c", "scene_segment", "qc_status", "qc_reasons"]  # fmt: skip


def features_frame_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "processed" / "features_frame.parquet"


def features_triplet_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "processed" / "features_triplet.parquet"


def mask_sources(frames: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """``frames`` + ``triplet``, the RED / BLUE frame whose masks the frame uses (its own
    triplet's when present, else the nearest in time in the session) and ``roi_gap_s`` (s
    between the triplet and that RED frame)."""
    from foodlens.features.triplets import assign_triplets

    lights = cfg.capture.light_order
    f = frames.assign(triplet=assign_triplets(frames, lights, cfg.triplet.max_span_s))
    t = f.groupby(["session_id", "triplet"], as_index=False)["ts"].mean().sort_values("ts")
    for light, tag in (("RED", "red"), ("BLUE", "blue")):
        src = f.loc[f["light"] == light, ["session_id", "ts", "frame_id", "path"]].sort_values("ts")
        src = src.rename(columns={"frame_id": f"{tag}_id", "path": f"{tag}_path"})
        src[f"{tag}_ts"] = src["ts"]
        t = pd.merge_asof(t, src, on="ts", by="session_id", direction="nearest")
    t["roi_gap_s"] = (t["red_ts"] - t["ts"]).dt.total_seconds().abs()
    keep = ["session_id", "triplet", "red_id", "red_path", "blue_id", "blue_path", "roi_gap_s"]
    return f.merge(t[keep], on=["session_id", "triplet"], how="left", validate="many_to_one")


def _triplet_job(rows: list[dict], key: str, cfg: Config) -> list[dict]:
    """Masks from the triplet's RED / BLUE source frames, then features of each frame."""
    import cv2

    from foodlens.features.build import extract_frame_features
    from foodlens.io.images import load_rgb
    from foodlens.vision.roi import CHANNEL_INDEX
    from foodlens.vision.triplet_roi import frame_reference, triplet_masks

    r0, out = rows[0], []
    red = load_rgb(cfg.root / r0["red_path"], cfg.vision.work_long_side)
    blue = None
    if isinstance(r0["blue_path"], str):
        blue = load_rgb(cfg.root / r0["blue_path"], cfg.vision.work_long_side)[..., 2]
        blue = blue if blue.shape == red.shape[:2] else None
    tm = triplet_masks(red, blue, cfg)
    for r in rows:
        base = {"frame_id": r["frame_id"], "zone_key": key, "mask_reason": tm.reason}
        if not tm.ok:
            out.append(base)
            continue
        img = load_rgb(cfg.root / r["path"], cfg.vision.glint_long_side)
        h, w = tm.full.shape
        ch_work = cv2.resize(
            img[..., CHANNEL_INDEX[r["light"]]], (w, h), interpolation=cv2.INTER_AREA
        )
        ref = frame_reference(ch_work, tm.full, cfg)
        out.append(base | extract_frame_features(img, r["light"], tm.lit, cfg, tm.shadow, ref))
    return out


def extract_all(frames: pd.DataFrame, cfg: Config, force: bool) -> pd.DataFrame:
    """Raw frame features of ``frames`` + ``mask_reason`` ("" = ok), cached by config hash and
    the source RED / BLUE frame ids (``zone_key``). Adds ``roi_gap_s``."""
    from joblib import Parallel, delayed

    src = mask_sources(frames, cfg)
    src["zone_key"] = src["red_id"].astype(str) + "|" + src["blue_id"].astype(str)
    cache = cfg.path("data_dir") / "interim" / "features" / config_hash(cfg, FEATURE_HASH_SECTIONS)
    cache_file = cache / "frame_raw.parquet"
    old = pd.read_parquet(cache_file) if cache_file.is_file() and not force else None
    if old is not None:
        old = old[old["zone_key"] == old["frame_id"].map(src.set_index("frame_id")["zone_key"])]
    todo = src[~src["frame_id"].isin(set() if old is None else set(old["frame_id"]))]
    groups = [g.to_dict("records") for _, g in todo.groupby(["session_id", "triplet"])]
    log.info("features: %d frames (%d triplets) to extract, %d from cache",
             len(todo), len(groups), 0 if old is None else len(old))  # fmt: skip
    res = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_triplet_job)(g, g[0]["zone_key"], cfg) for g in groups
    )
    new = pd.DataFrame([row for rows in res for row in rows])
    raw = pd.concat([d for d in (old, new) if d is not None and len(d)], ignore_index=True)
    _write_parquet(raw, cache_file, cfg)
    raw = raw[raw["frame_id"].isin(src["frame_id"])].drop(columns="zone_key")
    return raw.merge(src[["frame_id", "roi_gap_s"]], on="frame_id", how="left")


def frame_table(frames: pd.DataFrame, raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Meta + features per frame, blur -> NaN, step-corrected; schema column order."""
    from foodlens.features.build import blur_to_nan, step_correct_frames

    bad = raw["mask_reason"].fillna("") != ""
    for fid, why in zip(raw.loc[bad, "frame_id"], raw.loc[bad, "mask_reason"], strict=True):
        log.warning("features: frame %s rejected: %s", fid, why)
    raw = raw[~bad].drop(columns="mask_reason")
    df = frames[ID_IN].merge(raw, on="frame_id", how="inner", validate="one_to_one")
    df["elapsed_h"] = df["elapsed_s"] / 3600.0
    df = step_correct_frames(blur_to_nan(df, cfg), cfg)
    names = [s.name for s in frame_specs(cfg)]
    return df[FRAME_META + names].sort_values(["session_id", "ts"]).reset_index(drop=True)


def triplet_table(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Wide triplets + cross-light ratios + session-relative variants; schema column order."""
    from foodlens.features.schema import relative_sources
    from foodlens.features.triplets import add_cross_light, add_relative, wide_triplets

    lights = cfg.capture.light_order
    names = [s.name for s in frame_specs(cfg)]
    t = wide_triplets(frame, names, lights, cfg.triplet.max_span_s)
    t = add_cross_light(t, cfg.features.ratio_sources, cfg.features.log_ratio_sources)
    t = add_relative(t, [s.name for s in relative_sources(cfg)])
    return t[triplet_meta(cfg) + [s.name for s in triplet_specs(cfg)]]


def counts_table(frame: pd.DataFrame, trip: pd.DataFrame) -> pd.DataFrame:
    """Rows per session: frames per light, triplets, complete / partial, blur-NaN frames."""
    f = frame.pivot_table(index="session_id", columns="light", values="frame_id", aggfunc="size")
    f.columns = [f"frames_{c}" for c in f.columns]
    t = trip.groupby("session_id").agg(
        triplets=("triplet", "size"),
        complete=("is_complete", "sum"),
        hours=("elapsed_h", "max"),
    )
    t["partial"] = t["triplets"] - t["complete"]
    t["blur_nan_frames"] = frame.groupby("session_id")["blur_nan"].sum()
    out = f.join(t).reset_index()
    total = out.drop(columns="session_id").sum(numeric_only=True)
    total["hours"] = np.nan
    return pd.concat([out, pd.DataFrame([{"session_id": "TOTAL", **total}])], ignore_index=True)


def run_features(cfg: Config, limit: int | None, force: bool) -> dict[str, int]:
    """P4.T2-T4 end to end. Returns row counts."""
    from foodlens.eda.run import usable_frames
    from foodlens.features.qc import feature_qc
    from foodlens.features.schema import export_feature_dictionary

    frames = usable_frames(cfg)
    if limit is not None:
        frames = frames.head(limit)
    frame = frame_table(frames, extract_all(frames, cfg, force), cfg)
    trip = triplet_table(frame, cfg)
    _write_parquet(frame, features_frame_path(cfg), cfg)
    _write_parquet(trip, features_triplet_path(cfg), cfg)
    tables = tables_dir(cfg)
    fcols, tcols = [s.name for s in frame_specs(cfg)], [s.name for s in triplet_specs(cfg)]
    qc, pairs = feature_qc(frame, fcols, trip, tcols, cfg.features.corr_max)
    qc.to_csv(tables / "feature_qc.csv", index=False)
    pairs.to_csv(tables / "feature_corr_pairs.csv", index=False)
    counts = counts_table(frame, trip)
    counts.to_csv(tables / "feature_counts.csv", index=False)
    export_feature_dictionary(cfg, tables / "feature_dictionary.csv")
    all_nan = qc.loc[qc["pct_nan"] >= 100, ["level", "column", "light"]]
    if len(all_nan):
        log.warning("100 %% NaN columns: %s", all_nan.to_dict("records"))
    log.info("features: %d frames, %d triplets (%d complete), %d high-|r| pairs",
             len(frame), len(trip), int(trip["is_complete"].sum()), len(pairs))  # fmt: skip
    return {"frames": len(frame), "triplets": len(trip), "all_nan_columns": len(all_nan)}
