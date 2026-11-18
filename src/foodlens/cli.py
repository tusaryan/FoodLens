"""``foodlens`` command-line interface (typer). One command per pipeline stage.

Every stage takes ``--config``, ``--limit`` and ``--force``. An implemented stage must end by
printing ``STAGE DONE <stage> rows=<n>``. Stages not yet implemented are stubs.
"""

from __future__ import annotations

from pathlib import Path

import typer

from foodlens.config import load_config

app = typer.Typer(
    help="FoodLens: room-temperature age / freshness of raw chicken from R/G/B-LED photos.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOpt = typer.Option(
    None, "--config", help="Path to the config YAML (default: configs/default.yaml)."
)
LimitOpt = typer.Option(None, "--limit", min=1, help="Process only the first N items (smoke test).")
ForceOpt = typer.Option(False, "--force", help="Recompute even if cached outputs exist.")


def stage_done(stage: str, rows: int) -> None:
    """Print the success marker that check scripts and agents wait for."""
    typer.echo(f"STAGE DONE {stage} rows={rows}")


def _stub(stage: str, config: Path | None, limit: int | None, force: bool) -> None:
    load_config(config)  # validate config even for stubs, so bad configs fail early
    typer.echo(f"{stage}: not implemented yet (limit={limit}, force={force})")


SkipTempOpt = typer.Option(
    False, "--skip-temp", help="Do not look up the ambient temperature (offline smoke tests)."
)


@app.command()
def ingest(
    config: Path | None = ConfigOpt,
    limit: int | None = LimitOpt,
    force: bool = ForceOpt,
    skip_temp: bool = SkipTempOpt,
):
    """Build the frame manifest (+ temperature per frame) from the raw sessions (P1)."""
    from foodlens.stages import run_ingest

    rows = run_ingest(load_config(config), limit, force, with_temp=not skip_temp)
    stage_done("ingest", rows)


@app.command()
def qc(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Verify lights; flag dark / mixed-light / blurred frames (P1) and, once `roi` has run, no-ROI
    and scene-change frames (P2.T4)."""
    from foodlens.stages import run_qc

    df = run_qc(load_config(config), limit, force)
    counts = df["qc_status"].value_counts().to_dict()
    typer.echo(f"qc status counts: {counts}")
    stage_done("qc", len(df))


@app.command()
def roi(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Segment the meat ROI per frame, build session ROIs and channel crops (P2)."""
    from foodlens.stages import run_roi

    df = run_roi(load_config(config), limit, force)
    typer.echo(f"roi ok: {int(df['roi_ok'].sum())}/{len(df)}")
    typer.echo(f"roi reasons: {df['roi_reason'].replace('', 'none').value_counts().to_dict()}")
    stage_done("roi", len(df))


@app.command("eda-stats")
def eda_stats(
    config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt
):
    """Per-frame ROI statistics of every usable frame -> frame_stats.parquet (P3.T1).

    Always recomputed (``--force`` has no effect); ``--limit`` = first N usable frames.
    """
    from foodlens.eda.run import run_frame_stats

    df = run_frame_stats(load_config(config), limit)
    typer.echo(f"frame_stats: {len(df)} frames, {df['session_id'].nunique()} sessions")
    stage_done("eda-stats", len(df))


@app.command()
def features(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Frame + triplet feature tables, feature dictionary and feature QC (P4).

    Cached per frame by config hash; ``--force`` re-extracts; ``--limit`` = first N usable frames.
    """
    from foodlens.features.run import run_features

    out = run_features(load_config(config), limit, force)
    typer.echo(f"features: {out}")
    stage_done("features", out["triplets"])


@app.command()
def labels(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Build label columns (hours, equivalent hours, freshness level) per triplet (P5.T1)."""
    from foodlens.evaluation.run import run_labels

    lab = run_labels(load_config(config), limit)
    typer.echo(lab.groupby("freshness_class").size().to_string())
    stage_done("labels", len(lab))


@app.command()
def train(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Train models under grouped CV (P6)."""
    _stub("train", config, limit, force)


@app.command()
def evaluate(
    config: Path | None = ConfigOpt,
    limit: int | None = LimitOpt,
    force: bool = ForceOpt,
    model: str = typer.Option(
        "baseline_*", "--model", help="Glob over model names (baseline_*, leakage_demo, *)."
    ),
):
    """Evaluate models with grouped CV, CIs, permutation tests and baselines (P5.T5).
    ``--limit N`` keeps N triplets per session (smoke test)."""
    from foodlens.evaluation.run import run_evaluate

    table = run_evaluate(load_config(config), model, limit)
    if len(table):
        typer.echo(table[["run_id", "metric", "value", "ci_lo", "ci_hi"]].to_string(index=False))
    stage_done("evaluate", len(table))


@app.command()
def predict(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Predict age and freshness class from R, G, B photos (P8)."""
    _stub("predict", config, limit, force)


@app.command("eda-sheets")
def eda_sheets(
    config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt
):
    """QA contact sheets per session: session ROI + glints on start..end frames per light (P2.T5).

    ``--limit`` = number of sessions; sheets are always re-rendered (``--force`` has no effect).
    """
    from foodlens.stages import run_qa_sheets

    df = run_qa_sheets(load_config(config), limit)
    floor = int((df["glint_thr"] <= load_config(config).glint.abs_min).sum())
    typer.echo(f"qa sheets: {df['session_id'].nunique()} sessions, {len(df)} panels")
    typer.echo(f"glint threshold at glint.abs_min floor: {floor}/{len(df)} panels")
    stage_done("eda-sheets", len(df))


@app.command()
def eda(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """EDA tables and figures F5-F8 from frame_stats.parquet (P3.T2-T5).

    Always recomputed; ``--limit`` / ``--force`` have no effect (it reads the existing table).
    """
    from foodlens.eda.run import run_eda

    counts = run_eda(load_config(config))
    typer.echo(f"eda tables: {counts}")
    stage_done("eda", sum(counts.values()))


@app.command("eda-sample")
def eda_sample(
    config: Path | None = ConfigOpt,
    tag: str = typer.Option(..., "--tag", help="Session tag for the review log, e.g. S06."),
):
    """Render a NEW varied frame sample for visual review (random + outliers + special frames)."""
    from foodlens.eda.run import run_visual_sample

    df = run_visual_sample(load_config(config), tag)
    typer.echo(df["reason"].str.split(":").str[0].str.split(" z=").str[0].value_counts().to_dict())
    stage_done("eda-sample", len(df))


if __name__ == "__main__":
    app()


@app.command()
def crops(
    config: Path | None = ConfigOpt,
    limit: int | None = LimitOpt,
    force: bool = ForceOpt,
    tag: str = typer.Option("", "--tag", help="Render a NEW review sample tagged e.g. S08."),
):
    """Meat-only crops (full meat = lit + shadow, background set to 0) of every usable frame
    (S08); ``--tag`` also renders original vs crop for frames never reviewed before."""
    from foodlens.vision.crop import review_sheet, run_crops

    cfg = load_config(config)
    index = run_crops(cfg, limit, force)
    if tag:
        png, sample = review_sheet(cfg, index, tag)
        typer.echo(f"review sheet: {png} ({len(sample)} frames)")
    ok = int((index["crop_reason"] == "").sum())
    typer.echo(f"crops: {ok} ok, {len(index) - ok} rejected")
    stage_done("crops", ok)
