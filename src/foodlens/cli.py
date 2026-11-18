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
    """Verify lights and flag dark / mixed-light / blurred frames (P1)."""
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


@app.command()
def features(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Compute per-frame and per-triplet features (P4)."""
    _stub("features", config, limit, force)


@app.command()
def labels(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Build label columns (hours, equivalent hours, freshness class) (P5)."""
    _stub("labels", config, limit, force)


@app.command()
def train(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Train models under grouped CV (P6)."""
    _stub("train", config, limit, force)


@app.command()
def evaluate(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Evaluate models with grouped CV, CIs and baselines (P5/P6)."""
    _stub("evaluate", config, limit, force)


@app.command()
def predict(config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt):
    """Predict age and freshness class from R, G, B photos (P8)."""
    _stub("predict", config, limit, force)


@app.command("eda-sheets")
def eda_sheets(
    config: Path | None = ConfigOpt, limit: int | None = LimitOpt, force: bool = ForceOpt
):
    """Render EDA contact sheets per session and light (P3)."""
    _stub("eda-sheets", config, limit, force)


if __name__ == "__main__":
    app()
