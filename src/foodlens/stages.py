"""Stage runners called by the CLI: read inputs, run one stage, write its outputs (Phase P1+).

Output locations (all derived from config paths):
- ``<data_dir>/processed/manifest.parquet``: the one table every later stage reads.
- ``<data_dir>/interim/qc/<hash>/frame_stats.parquet``: cached per-frame QC statistics.
- ``<reports_dir>/tables/inventory.csv``, ``qc_summary.csv``; ``<reports_dir>/qc_rejects.csv``.
- ``<data_dir>/processed/roi_index.parquet`` + ``interim/roi/<hash>/``; ``tables/roi_*.csv`` (P2).
- ``<reports_dir>/tables/scene_steps.csv``: persistent scene steps per session (P2.T4).
- ``<reports_dir>/figures/qa_roi_<session>.png`` + ``tables/qa_sheet_frames.csv`` (P2.T5).

Stage order: ingest -> qc -> roi. ``qc`` applies the ROI-based rules (no ROI, scene change, ROI
blur) when ``roi_index.parquet`` exists, and ``roi`` re-runs ``qc`` at its end, so the manifest
QC columns always match the latest ROI index. Re-running either stage gives the same result.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from foodlens import __version__
from foodlens.config import Config, config_hash, load_samples
from foodlens.io.inventory import inventory_table
from foodlens.io.manifest import build_manifest
from foodlens.logging_utils import get_logger
from foodlens.qc.frame_qc import apply_qc, compute_frame_stats, qc_rejects, qc_summary

log = get_logger("stages")
QC_HASH_SECTIONS = ["qc", "vision"]


def manifest_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "processed" / "manifest.parquet"


def tables_dir(cfg: Config) -> Path:
    d = cfg.path("reports_dir") / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_parquet(df: pd.DataFrame, path: Path, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.attrs = {"foodlens_version": __version__, "config_hash": config_hash(cfg)}
    df.to_parquet(path, index=False)


def read_manifest(cfg: Config) -> pd.DataFrame:
    """The current manifest; raises ``FileNotFoundError`` with a hint if ingest has not run."""
    p = manifest_path(cfg)
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found: run `foodlens ingest` first")
    return pd.read_parquet(p)


def run_ingest(cfg: Config, limit: int | None, force: bool, with_temp: bool = True) -> int:
    """Build (or refresh) the manifest and the inventory table. Returns the number of rows."""
    samples = load_samples(cfg)
    prev = None
    if not force and manifest_path(cfg).is_file():
        prev = pd.read_parquet(manifest_path(cfg))
    df = build_manifest(cfg, samples, limit=limit, previous=prev)
    if with_temp:
        from foodlens.io.temperature import attach_temperature
        from foodlens.viz import plot_session_temperatures

        df = attach_temperature(df, cfg, samples)
        plot_session_temperatures(df, cfg.path("reports_dir") / "figures")
    _write_parquet(df, manifest_path(cfg), cfg)
    inventory_table(df).to_csv(tables_dir(cfg) / "inventory.csv", index=False)
    log.info("manifest: %d rows -> %s", len(df), manifest_path(cfg))
    return len(df)


def _roi_qc(man: pd.DataFrame, stats: pd.DataFrame, cfg: Config):
    """ROI QC table + scene steps from the ROI index, or (None, None) if there is no usable one."""
    from foodlens.qc.scene_qc import roi_qc_table

    p = roi_index_path(cfg)
    if not p.is_file():
        log.info("no %s yet: ROI-based QC rules skipped (run `foodlens roi`)", p.name)
        return None, None
    try:
        roi_qc, steps = roi_qc_table(man, stats, pd.read_parquet(p), cfg)
    except KeyError as exc:
        log.warning("ROI-based QC rules skipped: %s", exc)
        return None, None
    n_pre = int((man["frame_id"].isin(stats["frame_id"])).sum())
    log.info("ROI QC: %d of %d frames in the ROI index", len(roi_qc), n_pre)
    return roi_qc, steps


def run_qc(cfg: Config, limit: int | None, force: bool) -> pd.DataFrame:
    """Light verification + frame QC (+ ROI-based rules if the ROI index exists); rewrites the
    manifest with QC columns. Returns it."""
    man = read_manifest(cfg)
    if limit is not None:
        man = man.head(limit)
    cache = cfg.path("data_dir") / "interim" / "qc" / config_hash(cfg, QC_HASH_SECTIONS)
    cache_file = cache / "frame_stats.parquet"
    cached = None if force or not cache_file.is_file() else pd.read_parquet(cache_file)
    stats = compute_frame_stats(man, cfg, cached)
    if cached is not None:
        stats = pd.concat([cached[~cached["frame_id"].isin(stats["frame_id"])], stats])
    _write_parquet(stats, cache_file, cfg)
    roi_qc, steps = _roi_qc(man, stats, cfg)
    df = apply_qc(man, stats, cfg, roi_qc)
    _write_parquet(df, manifest_path(cfg), cfg)
    tables = tables_dir(cfg)
    if steps is not None:
        steps.to_csv(tables / "scene_steps.csv", index=False)
    qc_summary(df).to_csv(tables / "qc_summary.csv", index=False)
    qc_rejects(df).to_csv(cfg.path("reports_dir") / "qc_rejects.csv", index=False)
    inventory_table(df).to_csv(tables / "inventory.csv", index=False)
    return df


def roi_index_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "processed" / "roi_index.parquet"


ROI_INDEX_COLS = ["frame_id", "session_id", "light", "qc_pre_roi", "path"]


def run_roi(cfg: Config, limit: int | None, force: bool) -> pd.DataFrame:
    """ROI for every frame usable by the non-ROI QC rules (``qc_pre_roi`` ok/warn); writes
    ``roi_index.parquet``, then re-runs ``qc`` on the full manifest to apply the ROI rules."""
    from joblib import Parallel, delayed

    from foodlens.vision import roi_batch as rb

    man = read_manifest(cfg)
    if "qc_pre_roi" not in man:
        raise FileNotFoundError("manifest has no (current) QC columns: run `foodlens qc` first")
    frames = man[man["qc_pre_roi"].isin(["ok", "warn"])].sort_values("frame_id")
    frames = frames.assign(light=frames["pixel_light"])[ROI_INDEX_COLS].reset_index(drop=True)
    if limit is not None:
        frames = frames.head(limit)
    base = cfg.path("data_dir") / "interim" / "roi" / config_hash(cfg, rb.ROI_HASH_SECTIONS)
    mask_dir, crop_dir = base / "masks", base / "crops"
    log.info("roi: %d frames -> %s", len(frames), base)
    stats = Parallel(n_jobs=cfg.n_jobs)(
        delayed(rb.frame_mask)(cfg.root / p, li, cfg, mask_dir / f"{f}.png", force)
        for f, p, li in zip(frames["frame_id"], frames["path"], frames["light"], strict=True)
    )
    df = pd.concat([frames, pd.DataFrame(list(stats), columns=rb.FRAME_COLS)], axis=1)
    df["seg_ok"] = df["seg_ok"].astype(bool)
    refs, removed = rb.session_rois(df, cfg, mask_dir)
    sess = rb.session_table(refs, removed, base / "session", cfg)
    outs = Parallel(n_jobs=cfg.n_jobs)(
        delayed(rb.frame_outputs)(row, refs.get(row["session_id"]), cfg, mask_dir, crop_dir)
        for row in df.to_dict("records")
    )
    df = pd.concat([df, pd.DataFrame(list(outs))], axis=1)
    df = (
        df.merge(sess, on="session_id", how="left") if len(sess) else df.assign(sroi_mask_path=None)
    )
    df = rb.roi_status(df, cfg)
    _write_parquet(df, roi_index_path(cfg), cfg)
    sess.to_csv(tables_dir(cfg) / "roi_sessions.csv", index=False)
    summary = df.groupby(["session_id", "light"]).agg(
        n=("frame_id", "size"),
        roi_ok=("roi_ok", "sum"),
        low_iou=("roi_reason", lambda s: int((s == "W_ROI_LOW_IOU").sum())),
        iou_min=("roi_iou_session", "min"),
        iou_median=("roi_iou_session", "median"),
        area_frac_median=("roi_area_frac", "median"),
    )
    summary.reset_index().to_csv(tables_dir(cfg) / "roi_summary.csv", index=False)
    run_qc(cfg, None, force=False)  # refresh the manifest QC columns with the ROI rules
    return df


def run_qa_sheets(cfg: Config, limit: int | None) -> pd.DataFrame:
    """QA contact sheet per session (first ``limit`` sessions); returns the panel table."""
    from foodlens.qc.qa_sheets import session_sheet
    from foodlens.vision.roi_batch import read_mask

    man = read_manifest(cfg)
    sess_file = tables_dir(cfg) / "roi_sessions.csv"
    if "roi_ok" not in man or not sess_file.is_file():
        raise FileNotFoundError("no ROI QC in the manifest: run `foodlens roi` first")
    sessions = pd.read_csv(sess_file).sort_values("session_id")
    if limit is not None:
        sessions = sessions.head(limit)
    figures = cfg.path("reports_dir") / "figures"
    tables = [
        session_sheet(man, sid, read_mask(cfg.root / mp), cfg, figures)
        for sid, mp in zip(sessions["session_id"], sessions["sroi_mask_path"], strict=True)
    ]
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(tables_dir(cfg) / "qa_sheet_frames.csv", index=False)
    return out
