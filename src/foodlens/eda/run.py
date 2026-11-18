"""EDA stage runners called by the CLI (P3).

- ``run_frame_stats`` (``foodlens eda-stats``): ``<data_dir>/processed/frame_stats.parquet``
  (one row per usable frame) and ``glint_props.parquet`` (one row per glint, both rules), the
  meat-zone masks under ``<data_dir>/interim/eda/<hash>/zones/``, ``tables/eda_zones.csv`` and
  ``figures/qa_zones.png``.
- ``run_eda`` (``foodlens eda``): the P3.T2-T5 tables and figures from ``frame_stats.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from foodlens.config import Config, config_hash
from foodlens.logging_utils import get_logger
from foodlens.stages import _write_parquet, read_manifest, tables_dir

log = get_logger("eda.run")
ZONE_HASH_SECTIONS = ["roi", "vision", "eda"]
USABLE = ["ok", "warn"]


def frame_stats_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "processed" / "frame_stats.parquet"


def glint_props_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "processed" / "glint_props.parquet"


def read_frame_stats(cfg: Config) -> pd.DataFrame:
    """The current frame_stats table; raises ``FileNotFoundError`` with a hint if missing."""
    p = frame_stats_path(cfg)
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found: run `foodlens eda-stats` first")
    return pd.read_parquet(p)


def usable_frames(cfg: Config) -> pd.DataFrame:
    """Manifest rows with qc_status ok/warn, sorted by session and time."""
    man = read_manifest(cfg)
    if "qc_status" not in man or "roi_ok" not in man:
        raise FileNotFoundError("manifest has no ROI QC columns: run `foodlens roi` first")
    use = man[man["qc_status"].isin(USABLE)]
    return use.sort_values(["session_id", "ts"]).reset_index(drop=True)


def run_frame_stats(cfg: Config, limit: int | None) -> pd.DataFrame:
    """P3.T1: per-frame ROI statistics of every usable frame (first ``limit`` frames if given)."""
    from foodlens.eda.frame_stats import compute_frame_stats
    from foodlens.eda.meat_zones import reference_masks, session_zones, zones_figure
    from foodlens.io.images import load_rgb
    from foodlens.vision.roi_batch import read_mask

    frames = usable_frames(cfg)
    if limit is not None:
        frames = frames.head(limit)
    sessions = pd.read_csv(tables_dir(cfg) / "roi_sessions.csv")
    sessions = sessions[sessions["session_id"].isin(frames["session_id"])]
    zone_dir = cfg.path("data_dir") / "interim" / "eda" / config_hash(cfg, ZONE_HASH_SECTIONS)
    zones, ztable = session_zones(frames, sessions, cfg, zone_dir / "zones")
    missing = sorted(set(frames["session_id"]) - set(zones))
    if missing:
        log.warning("no session ROI for %s: their frames are skipped", missing)
    full = {
        s: read_mask(cfg.root / f)
        for s, f in zip(ztable["session_id"], ztable["full_mask_path"], strict=True)
    }
    stats, props = compute_frame_stats(frames, zones, cfg, reference_masks(frames, full, cfg))
    _write_parquet(stats, frame_stats_path(cfg), cfg)
    _write_parquet(props, glint_props_path(cfg), cfg)
    ztable.to_csv(tables_dir(cfg) / "eda_zones.csv", index=False)
    red = {}
    for sid in zones:
        g = frames[(frames["session_id"] == sid) & (frames["light"] == "RED")]
        if len(g):
            red[sid] = load_rgb(cfg.root / g["path"].iloc[len(g) // 2], cfg.vision.work_long_side)[
                ..., 0
            ]
    if len(red) == len(zones):
        zones_figure(zones, red, cfg.path("reports_dir") / "figures", cfg.qa.dpi)
    nan_cols = [c for c in stats.columns if stats[c].isna().all()]
    if nan_cols:
        log.warning("all-NaN columns in frame_stats: %s", nan_cols)
    log.info(
        "frame_stats: %d frames, %d glints -> %s", len(stats), len(props), frame_stats_path(cfg)
    )
    return stats


def run_visual_sample(cfg: Config, tag: str, per_sheet: int = 12) -> pd.DataFrame:
    """Pick a new varied frame sample (``eda.visual_sample``), render sheets of ``per_sheet``
    panels as ``figures/visual_sample_<tag>_<k>.png`` and append it to ``visual_review_log.csv``."""
    from foodlens.eda.visual_sample import LOG_COLS, pick_sample, render_sample, reviewed_ids
    from foodlens.vision.roi_batch import read_mask

    tables = tables_dir(cfg)
    stats = read_frame_stats(cfg)
    paths = usable_frames(cfg).set_index("frame_id")["path"]
    sample = pick_sample(stats, reviewed_ids(tables), cfg, cfg.eda.sample_outliers)
    sample["path"] = sample["frame_id"].map(paths)
    lit = pd.read_csv(tables / "roi_sessions.csv").set_index("session_id")["sroi_mask_path"]
    full = pd.read_csv(tables / "eda_zones.csv").set_index("session_id")["full_mask_path"]
    masks = {s: (read_mask(cfg.root / lit[s]), read_mask(cfg.root / full[s])) for s in full.index}
    for k in range(0, len(sample), per_sheet):
        render_sample(
            sample.iloc[k : k + per_sheet], masks, cfg, f"visual_sample_{tag}_{k // per_sheet + 1}"
        )
    log_file = tables / "visual_review_log.csv"
    new = sample.assign(tag=tag)[LOG_COLS]
    if log_file.is_file():
        new = pd.concat([pd.read_csv(log_file), new], ignore_index=True)
    new.to_csv(log_file, index=False)
    return sample


T3_METRICS = ["roi_median", "glint_area_frac", "glint_density", "glcm_contrast", "rglint_density"]
RATIO_SOURCES = ["roi_median", "roi_mean"]


def _step_hours(man: pd.DataFrame, tables: Path) -> dict[str, list[float]]:
    """Scene-step times (h since the session's first frame) from ``scene_steps.csv``."""
    f = tables / "scene_steps.csv"
    if not f.is_file():
        return {}
    steps = pd.read_csv(f)
    start = man.groupby("session_id")["ts"].min()
    out: dict[str, list[float]] = {}
    for sid, ts in zip(steps["session_id"], pd.to_datetime(steps["step_ts"]), strict=True):
        out.setdefault(sid, []).append((ts - start[sid]).total_seconds() / 3600.0)
    return out


def _step_gains(raw: pd.DataFrame, sc: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Median raw / corrected ratio of ``cols`` in segments > 0, per session x light."""
    later = raw["scene_segment"] > 0
    if not later.any():
        return pd.DataFrame()
    ratio = raw.loc[later, cols] / sc.loc[later, cols]
    ratio[["session_id", "light"]] = raw.loc[later, ["session_id", "light"]]
    return ratio.groupby(["session_id", "light"]).median().reset_index()


def _trends(stats: pd.DataFrame, sc: pd.DataFrame, feats: list[str], cfg: Config):
    from foodlens.eda.trends import mixed_table, relative_to_first, trend_table

    n = cfg.eda.min_frames
    trends = pd.concat(
        [
            trend_table(stats, feats, n).assign(variant="raw"),
            trend_table(sc, feats, n).assign(variant="step_corrected"),
        ],
        ignore_index=True,
    )
    rel = relative_to_first(sc, T3_METRICS, cfg.eda.rel_base_n)
    mixed = pd.concat(
        [
            mixed_table(sc, T3_METRICS, "raw units"),
            mixed_table(rel, T3_METRICS, "relative to start"),
        ],
        ignore_index=True,
    )
    return trends, rel, mixed


def _dynamics(stats: pd.DataFrame, cfg: Config, figures: Path) -> pd.DataFrame:
    from foodlens.eda.dynamics import dynamics_figure, dynamics_summary, pair_events, window_sizes

    props = pd.read_parquet(glint_props_path(cfg))
    out = []
    for variant in ("abs", "rel"):
        summ = dynamics_summary(
            pair_events(stats, props, variant, cfg), stats, variant, cfg.eda.alpha
        )
        sizes = window_sizes(stats, props, variant, cfg.eda.window_frac)
        dynamics_figure(sizes, summ, variant, figures)
        out.append(summ)
    return pd.concat(out, ignore_index=True)


def _signal(trends: pd.DataFrame, sc: pd.DataFrame, cfg: Config, figures: Path):
    from foodlens.eda.plots import signal_heatmap
    from foodlens.eda.signal import light_comparison, signal_ranking
    from foodlens.eda.trends import trend_table
    from foodlens.features.triplets import add_ratios, triplet_table

    trip = triplet_table(sc, RATIO_SOURCES, cfg.capture.light_order, cfg.triplet.max_span_s)
    trip, ratio_cols = add_ratios(trip, RATIO_SOURCES)
    trip_tr = trend_table(trip.assign(light="TRIPLET"), ratio_cols, cfg.eda.min_frames)
    frame_tr = trends[trends["variant"] == "step_corrected"]
    rank = signal_ranking(
        pd.concat([frame_tr, trip_tr], ignore_index=True),
        cfg.eda.alpha,
        cfg.eda.promising_min_sessions,
    )
    signal_heatmap(rank, figures)
    return rank, light_comparison(rank, sc, trends)


def run_eda(cfg: Config) -> dict[str, int]:
    """P3.T2-T5 from frame_stats.parquet: tables ``eda_*.csv`` / ``feature_signal_ranking.csv``
    and figures F5-F8. Returns row counts per output table."""
    from foodlens.eda.compare import run_compare
    from foodlens.eda.plots import relative_figure, timeseries_figure
    from foodlens.eda.signal import key_numbers
    from foodlens.eda.trends import (
        feature_columns,
        multiplicative_columns,
        rolling_median,
        step_correct,
    )

    stats, man = read_frame_stats(cfg), read_manifest(cfg)
    tables, figures = tables_dir(cfg), cfg.path("reports_dir") / "figures"
    sessions = pd.read_csv(tables / "roi_sessions.csv")
    sessions = sessions[sessions["session_id"].isin(stats["session_id"])]
    feats = feature_columns(stats)
    mult = multiplicative_columns(feats)
    sc = step_correct(stats, mult, cfg.qc.step_k)
    out = {"eda_start_mid_end": run_compare(man, stats, sessions, cfg)}
    smooth = rolling_median(sc, T3_METRICS, cfg.eda.rolling_window)
    steps = _step_hours(man, tables)
    for sid in sorted(stats["session_id"].unique()):
        timeseries_figure(stats, smooth, sid, T3_METRICS, steps.get(sid, []), figures)
    trends, rel, mixed = _trends(stats, sc, feats, cfg)
    relative_figure(rolling_median(rel, T3_METRICS, cfg.eda.rolling_window), T3_METRICS, figures)
    out |= {"eda_trends": trends, "eda_mixed_models": mixed}
    out["eda_step_gains"] = _step_gains(stats, sc, mult)
    out["eda_glint_dynamics"] = _dynamics(stats, cfg, figures)
    out["feature_signal_ranking"], out["eda_light_comparison"] = _signal(trends, sc, cfg, figures)
    out["eda_key_numbers"] = key_numbers(
        sc, trends, mixed, out["feature_signal_ranking"], out["eda_glint_dynamics"]
    )
    for name, df in out.items():
        df.to_csv(tables / f"{name}.csv", index=False)
        log.info("%s: %d rows", name, len(df))
    return {k: len(v) for k, v in out.items()}
