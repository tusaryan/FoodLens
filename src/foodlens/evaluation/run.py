"""``foodlens labels`` and ``foodlens evaluate`` stages (P5.T1, P5.T5).

Outputs:
- ``<data_dir>/processed/labels_triplet.parquet`` + ``tables/label_summary.csv``,
  ``tables/level_clock_ranges.csv`` (labels).
- ``<reports_dir>/metrics/<run_id>.json`` (one per model x target x CV scheme),
  ``tables/model_comparison.csv``, ``tables/leakage_demo.csv``, ``figures/F10b_leakage_demo``.
Schemes: leave-one-group-out for every ``evaluation.group_cols`` entry (first = headline), the
temporal holdout (secondary) and, for the headline grouping, a sensitivity run without
post-step triplets. The permutation test runs for the headline grouping only.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd

from foodlens.config import Config, config_hash, load_samples
from foodlens.errors import ConfigError
from foodlens.evaluation.cv import HEADLINE, evaluate
from foodlens.evaluation.splits import (
    assert_no_group_overlap,
    get_groups,
    logo_splits,
    temporal_holdout_splits,
)
from foodlens.evaluation.stats import RunReport, git_commit
from foodlens.logging_utils import get_logger
from foodlens.stages import _write_parquet, tables_dir

log = get_logger("evaluation.run")
CLOCK_TEMPS_C = (20.0, 25.0, 30.0, 35.0)  # °C rows of the indicative level -> clock-hour table


def labels_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "processed" / "labels_triplet.parquet"


def run_labels(cfg: Config, limit: int | None) -> pd.DataFrame:
    """Build and save the label table from the triplet feature table's meta columns."""
    from foodlens.features.run import features_triplet_path
    from foodlens.labels.labels import build_labels, level_clock_ranges

    meta = ["session_id", "experiment", "triplet", "ts", "elapsed_s", "temp_c", "scene_segment"]
    trip = pd.read_parquet(features_triplet_path(cfg), columns=meta)
    if limit is not None:
        trip = trip.sort_values(["session_id", "ts"]).head(limit)
    lab = build_labels(trip, load_samples(cfg), cfg)
    _write_parquet(lab, labels_path(cfg), cfg)
    summary = lab.groupby("session_id").agg(
        piece_id=("piece_id", "first"), n=("triplet", "size"),
        hours_max=("hours_since_start", "max"), equiv_h_max=("equiv_hours_25C", "max"),
        equiv_h_max_tmin_alt=("equiv_hours_25C_tmin_alt", "max"), post_step=("post_step", "sum"),
    )  # fmt: skip
    summary["tmin_alt_diff_pct"] = 100 * (
        summary["equiv_h_max_tmin_alt"] / summary["equiv_h_max"] - 1
    )
    counts = lab.pivot_table(index="session_id", columns="freshness_class", values="triplet",
                             aggfunc="size", fill_value=0)  # fmt: skip
    summary = summary.join(counts.reindex(columns=cfg.labels.classes, fill_value=0))
    summary.reset_index().to_csv(tables_dir(cfg) / "label_summary.csv", index=False)
    if cfg.labels.class_definition == "B":
        ranges = pd.concat([level_clock_ranges(t, cfg) for t in CLOCK_TEMPS_C], ignore_index=True)
        ranges.to_csv(tables_dir(cfg) / "level_clock_ranges.csv", index=False)
    return lab


def eval_table(cfg: Config) -> pd.DataFrame:
    """Triplet features joined with labels (excluded sessions are absent from the labels)."""
    from foodlens.features.run import features_triplet_path

    feats = pd.read_parquet(features_triplet_path(cfg)).drop(columns=["piece_id"])
    lab = pd.read_parquet(labels_path(cfg)).drop(columns=["experiment", "ts"])
    df = feats.merge(lab, on=["session_id", "triplet"], how="inner", validate="one_to_one")
    if cfg.evaluation.complete_only:
        df = df[df["is_complete"]]
    return df.sort_values(["session_id", "ts"]).reset_index(drop=True)


def single_shot_features(cfg: Config) -> list[str]:
    """All triplet features usable from one R/G/B capture (no __rel, no time, no ids), minus
    ``evaluation.drop_feature_prefixes`` (setup proxies)."""
    from foodlens.features.schema import triplet_specs

    drop = tuple(cfg.evaluation.drop_feature_prefixes)
    return [s.name for s in triplet_specs(cfg) if s.single_shot_ok and not s.name.startswith(drop)]


def _tasks(cfg: Config) -> list[tuple[str, str]]:
    ev = cfg.evaluation
    return [("regression", t) for t in ev.targets] + [("classification", ev.class_target)]


def _schemes(df: pd.DataFrame, cfg: Config) -> list[tuple[str, str, pd.DataFrame, bool]]:
    """(scheme name, group col, rows, permute?) for every CV scheme."""
    head = cfg.evaluation.group_cols[0]
    out = [(f"LOGO_{g}", g, df, g == head) for g in cfg.evaluation.group_cols]
    out.append(("temporal_60_40", "session_id", df, False))
    out.append(
        (f"LOGO_{head}_no_post_step", head, df[~df["post_step"]].reset_index(drop=True), False)
    )
    return out


def _report_row(rep: RunReport, res: dict, task: str, scheme: str) -> dict:
    name, _ = HEADLINE[task]
    lo, hi = res["ci"][name]
    perm = res["permutation"]
    return {"run_id": rep.run_id, "model": rep.model, "target": rep.target, "task": task,
            "scheme": scheme, "group_col": rep.group_col, "n_groups": rep.n_groups,
            "n_test": res["n_test"], "metric": name, "value": res["metrics"][name],
            "ci_lo": lo, "ci_hi": hi, "p_value": perm.get("p_value", np.nan)}  # fmt: skip


def run_model(spec, df: pd.DataFrame, cfg: Config, common: dict, select: dict | None = None,
              oof_sink: list | None = None) -> list[dict]:  # fmt: skip
    """All tasks x schemes of one model spec; writes one RunReport JSON per run.

    ``select`` (P6 experiments) may hold ``targets`` / ``schemes`` lists to restrict the runs,
    ``permute`` (bool, headline grouping only) and ``n_permutations``. ``oof_sink`` collects the
    out-of-fold predictions (with run_id / model / target / scheme columns)."""
    sel = select or {}
    rows = []
    for task, target in _tasks(cfg):
        if sel.get("targets") and target not in sel["targets"]:
            continue
        for scheme, gcol, d, permute in _schemes(df, cfg):
            if sel.get("schemes") and scheme not in sel["schemes"]:
                continue
            permute = permute and sel.get("permute", True)
            groups = get_groups(d, cfg, [gcol])
            if scheme.startswith("temporal"):
                splits = temporal_holdout_splits(d, cfg)
            else:
                splits = logo_splits(groups)
                assert_no_group_overlap(splits, groups)
            res = evaluate(spec.make, task, d, spec.features, target, groups, splits, cfg,
                           permute, sel.get("n_permutations"))  # fmt: skip
            rep = RunReport(
                run_id=f"{spec.name}__{target}__{scheme}", model=spec.name, target=target,
                task=task, cv_scheme=scheme, group_col=gcol, n_groups=len(np.unique(groups)),
                n_samples=len(d), feature_set=spec.features, metrics=res["metrics"],
                ci={k: list(v) for k, v in res["ci"].items()}, permutation=res["permutation"],
                notes=spec.description, **common,
            )  # fmt: skip
            rep.save(cfg.path("reports_dir") / "metrics")
            rows.append(_report_row(rep, res, task, scheme))
            if oof_sink is not None:
                oof_sink.append(res["oof"].assign(run_id=rep.run_id, model=spec.name,
                                                  target=target, scheme=scheme))  # fmt: skip
            log.info("%s: %s = %.3f", rep.run_id, rows[-1]["metric"], rows[-1]["value"])
    return rows


def update_comparison(rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Merge ``rows`` into ``tables/model_comparison.csv`` (other runs' rows are kept), sorted by
    task, scheme and the grouped metric (best first)."""
    if not len(rows):
        return rows
    path = tables_dir(cfg) / "model_comparison.csv"
    table = rows
    if path.is_file():
        old = pd.read_csv(path)
        table = pd.concat([old[~old["run_id"].isin(rows["run_id"])], rows], ignore_index=True)
    better = table["task"].map(lambda t: HEADLINE[t][1])
    table = table.assign(_k=np.where(better, -table["value"], table["value"]))
    table = table.sort_values(["task", "target", "scheme", "_k"]).drop(columns="_k")
    table.to_csv(path, index=False)
    return rows


def run_evaluate(cfg: Config, model_pattern: str, limit: int | None) -> pd.DataFrame:
    """Evaluate every baseline matching ``model_pattern`` (glob) and/or the leakage demo."""
    from foodlens.evaluation.leakage_demo import leakage_figure, run_leakage_demo
    from foodlens.labels.labels import label_definition
    from foodlens.models.baselines import baseline_specs

    df = eval_table(cfg)
    if limit is not None:  # smoke test: `limit` rows per session, spread evenly over its time
        pick = df.groupby("session_id")["ts"].transform(
            lambda s: np.isin(np.arange(len(s)), np.linspace(0, len(s) - 1, limit).round())
        )
        df = df[pick.astype(bool)].reset_index(drop=True)
    common = {"label_definition": label_definition(cfg), "config_hash": config_hash(cfg),
              "git_commit": git_commit(cfg.root)}  # fmt: skip
    specs = {k: v for k, v in baseline_specs(cfg).items() if fnmatch.fnmatch(k, model_pattern)}
    demo = fnmatch.fnmatch("leakage_demo", model_pattern)
    if not specs and not demo:
        raise ConfigError(f"no model matches {model_pattern!r}")
    rows = [r for s in specs.values() for r in run_model(s, df, cfg, common)]
    table = update_comparison(pd.DataFrame(rows), cfg)
    if demo:
        feats = single_shot_features(cfg)
        groups = get_groups(df, cfg, [cfg.evaluation.group_cols[0]])
        lk = run_leakage_demo(df, feats, groups, cfg)
        lk.to_csv(tables_dir(cfg) / "leakage_demo.csv", index=False)
        leakage_figure(lk, cfg.path("reports_dir") / "figures")
        log.info("leakage demo:\n%s", lk.to_string(index=False))
    return table
