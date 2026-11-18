"""Paired per-held-out-group comparison of feature sets (P6.T2 light ablation).

Every feature set is scored on the SAME held-out groups, so per-group metrics can be compared
pair-wise: which set wins on how many held-out pieces, and the mean per-group difference to a
reference set (with a group-bootstrap CI of that difference).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foodlens.evaluation import metrics as M
from foodlens.evaluation.cv import HEADLINE


def per_group_metric(oof: pd.DataFrame, task: str) -> pd.DataFrame:
    """Headline metric per held-out group of every run in ``oof`` (columns run_id, model,
    target, scheme, group, y, yhat). Regression: MAE; classification: balanced accuracy over
    the classes present in that group."""
    name, _ = HEADLINE[task]
    fn = M.mae if task == "regression" else M.balanced_accuracy
    cast = float if task == "regression" else str
    keys = ["run_id", "model", "target", "scheme", "group"]
    rows = [
        {**dict(zip(keys, k, strict=True)), "metric": name,
         "value": fn(g["y"].to_numpy(cast), g["yhat"].to_numpy(cast)), "n": len(g)}
        for k, g in oof.groupby(keys, sort=True)
    ]  # fmt: skip
    return pd.DataFrame(rows)


def _diff_ci(d: np.ndarray, n_boot: int, seed: int, level: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)]
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def paired_wins(pg: pd.DataFrame, task: str, ref: str, n_boot: int, seed: int,
                level: float) -> pd.DataFrame:  # fmt: skip
    """Per (model family, target, scheme) and feature set: number of held-out groups on which
    the set is the best of all sets, and its mean per-group difference to ``ref``.

    ``pg`` = ``per_group_metric`` output with ``family`` and ``feature_set`` columns."""
    _, greater = HEADLINE[task]
    out = []
    for (fam, target, scheme), d in pg.groupby(["family", "target", "scheme"], sort=True):
        wide = d.pivot_table(index="group", columns="feature_set", values="value")
        best = wide.idxmax(axis=1) if greater else wide.idxmin(axis=1)
        for fs in wide.columns:
            diff = (wide[fs] - wide[ref]).dropna().to_numpy() if ref in wide else np.array([])
            lo, hi = _diff_ci(diff, n_boot, seed, level) if len(diff) else (np.nan, np.nan)
            out.append({
                "family": fam, "target": target, "scheme": scheme, "feature_set": fs,
                "metric": HEADLINE[task][0], "n_groups": int(wide[fs].notna().sum()),
                "wins": int((best == fs).sum()), "mean_per_group": float(wide[fs].mean()),
                "mean_diff_vs_ref": float(diff.mean()) if len(diff) else np.nan,
                "diff_ci_lo": lo, "diff_ci_hi": hi, "ref": ref,
                "better_than_ref_groups": int(((diff > 0) if greater else (diff < 0)).sum()),
            })  # fmt: skip
    return pd.DataFrame(out)


def per_piece_table(oof: pd.DataFrame, model: str, cfg) -> pd.DataFrame:
    """Failure analysis: per held-out piece, the headline metric of ``model`` for every target
    in ``oof`` (LOGO_piece_id rows), with the piece's n and true-class mix."""
    d = oof[(oof["model"] == model) & (oof["scheme"] == f"LOGO_{cfg.evaluation.group_cols[0]}")]
    out = []
    for target, part in d.groupby("target"):
        task = "classification" if target == cfg.evaluation.class_target else "regression"
        pg = per_group_metric(part, task)[["group", "metric", "value", "n"]]
        out.append(pg.assign(target=target))
    return (
        pd.concat(out, ignore_index=True)
        .pivot_table(index="group", columns="target", values="value")
        .reset_index()
    )
