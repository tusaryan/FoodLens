"""``foodlens train --experiment <name>`` (P6.T2): run one experiment YAML.

An experiment (``configs/experiments/<name>.yaml``) = models x feature sets x targets x CV
schemes on the P5 harness (``evaluation.run.run_model``). Outputs:
- ``reports/metrics/<model>__<feature set>__<target>__<scheme>.json`` (RunReport per run),
- rows merged into ``reports/tables/model_comparison.csv``,
- ``reports/tables/<table>.csv`` (this experiment's runs) and, with ``paired: true``,
  ``reports/tables/<table>_paired.csv`` (per-held-out-group wins vs ``reference_set``),
- out-of-fold predictions ``<data_dir>/processed/oof/<name>__<task>.parquet`` (figures).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from foodlens.config import Config, config_hash
from foodlens.errors import ConfigError
from foodlens.evaluation.paired import paired_wins, per_group_metric
from foodlens.evaluation.run import eval_table, run_model, update_comparison
from foodlens.evaluation.stats import git_commit
from foodlens.logging_utils import get_logger
from foodlens.models.classical import classical_specs
from foodlens.models.feature_sets import feature_sets
from foodlens.stages import _write_parquet, tables_dir

log = get_logger("models.train")


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    models: list[str] = Field(min_length=1)  # names from models.zoo
    feature_sets: list[str] = Field(min_length=1)  # names from models.feature_sets
    targets: list[str] = Field(min_length=1)  # label columns (evaluation.targets / class_target)
    schemes: list[str] = Field(min_length=1)  # e.g. LOGO_piece_id, LOGO_experiment
    permute: bool = False  # grouped permutation test on the headline grouping
    paired: bool = False  # per-held-out-group comparison of the feature sets
    reference_set: str = "RGB_all"
    table: str


def load_experiment(cfg: Config, name: str) -> Experiment:
    path = cfg.root / cfg.train.experiments_dir / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"experiment file not found: {path}")
    try:
        return Experiment.model_validate(yaml.safe_load(path.read_text()))
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def oof_path(cfg: Config, name: str, task: str) -> Path:
    return cfg.path("data_dir") / "processed" / "oof" / f"{name}__{task}.parquet"


def limited_table(cfg: Config, limit: int | None) -> pd.DataFrame:
    """The evaluation table; with ``limit``, that many rows per session spread over time."""
    df = eval_table(cfg)
    if limit is None:
        return df
    pick = df.groupby("session_id")["ts"].transform(
        lambda s: np.isin(np.arange(len(s)), np.linspace(0, len(s) - 1, limit).round())
    )
    return df[pick.astype(bool)].reset_index(drop=True)


def run_experiment(cfg: Config, name: str, limit: int | None) -> pd.DataFrame:
    """Run experiment ``name``; returns its table (one row per run)."""
    from foodlens.labels.labels import label_definition

    exp = load_experiment(cfg, name)
    df = limited_table(cfg, limit)
    sets = feature_sets(exp.feature_sets, cfg)
    common = {"label_definition": label_definition(cfg), "config_hash": config_hash(cfg),
              "git_commit": git_commit(cfg.root)}  # fmt: skip
    select = {"targets": exp.targets, "schemes": exp.schemes, "permute": exp.permute,
              "n_permutations": cfg.train.n_permutations}  # fmt: skip
    rows, oof = [], []
    for spec in classical_specs(cfg, sets, exp.models):
        t0 = time.perf_counter()
        r = run_model(spec, df, cfg, common, select, oof)
        fam, fs = spec.name.split("__", 1)
        rows += [x | {"family": fam, "feature_set": fs, "n_features": len(spec.features)}
                 for x in r]  # fmt: skip
        log.info("%s: %d runs in %.0f s", spec.name, len(r), time.perf_counter() - t0)
    table = pd.DataFrame(rows)
    update_comparison(table.drop(columns=["family", "feature_set", "n_features"]), cfg)
    table.to_csv(tables_dir(cfg) / f"{exp.table}.csv", index=False)
    oof_df = pd.concat(oof, ignore_index=True)
    is_cls = oof_df["target"] == cfg.evaluation.class_target
    for task, part in (("classification", oof_df[is_cls]), ("regression", oof_df[~is_cls])):
        if len(part):  # one file per task: y is numeric (hours) or a class name
            cast = str if task == "classification" else float
            part = part.astype({"y": cast, "yhat": cast}).dropna(axis=1, how="all")
            _write_parquet(part.reset_index(drop=True), oof_path(cfg, exp.name, task), cfg)
    if exp.paired:
        paired_tables(cfg, exp, oof_df).to_csv(
            tables_dir(cfg) / f"{exp.table}_paired.csv", index=False
        )
    return table


def paired_tables(cfg: Config, exp: Experiment, oof: pd.DataFrame) -> pd.DataFrame:
    """Per-group wins of every feature set, per model family / target / scheme."""
    out = []
    for target in exp.targets:
        task = "classification" if target == cfg.evaluation.class_target else "regression"
        pg = per_group_metric(oof[oof["target"] == target], task)
        parts = pg["model"].str.split("__", n=1, expand=True)
        pg = pg.assign(family=parts[0], feature_set=parts[1])
        out.append(paired_wins(pg, task, exp.reference_set, cfg.cv.n_bootstrap, cfg.seed,
                               cfg.cv.ci_level))  # fmt: skip
    return pd.concat(out, ignore_index=True)
