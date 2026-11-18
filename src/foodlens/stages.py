"""Stage runners called by the CLI: read inputs, run one stage, write its outputs (Phase P1+).

Output locations (all derived from config paths):
- ``<data_dir>/processed/manifest.parquet``: the one table every later stage reads.
- ``<data_dir>/interim/qc/<hash>/frame_stats.parquet``: cached per-frame QC statistics.
- ``<reports_dir>/tables/inventory.csv``, ``qc_summary.csv``; ``<reports_dir>/qc_rejects.csv``.
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


def run_qc(cfg: Config, limit: int | None, force: bool) -> pd.DataFrame:
    """Light verification + frame QC; rewrites the manifest with QC columns. Returns it."""
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
    df = apply_qc(man, stats, cfg)
    _write_parquet(df, manifest_path(cfg), cfg)
    tables = tables_dir(cfg)
    qc_summary(df).to_csv(tables / "qc_summary.csv", index=False)
    qc_rejects(df).to_csv(cfg.path("reports_dir") / "qc_rejects.csv", index=False)
    inventory_table(df).to_csv(tables / "inventory.csv", index=False)
    return df
