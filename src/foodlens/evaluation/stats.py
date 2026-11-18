"""Group bootstrap CIs, grouped permutation tests and the RunReport JSON (P5.T3)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_group_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    groups: np.ndarray,
    metric: Metric,
    n: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI of ``metric`` by resampling GROUPS with replacement (all rows of a drawn
    group are kept). Replicates whose metric is NaN (e.g. one class only) are skipped."""
    y, yhat, groups = np.asarray(y), np.asarray(yhat), np.asarray(groups)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = np.concatenate([rows[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        v = metric(y[idx], yhat[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def permute_within_groups(
    y: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """``y`` shuffled inside each group: breaks the feature-label link in time while keeping
    every group's label distribution (the grouped null hypothesis)."""
    out = np.asarray(y).copy()
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        out[i] = out[rng.permutation(i)]
    return out


def group_permutation_test(
    score_fn: Callable[[np.ndarray], float],
    y: np.ndarray,
    groups: np.ndarray,
    n: int,
    seed: int,
    greater_is_better: bool,
) -> dict:
    """p-value of the observed ``score_fn(y)`` (which must re-run the full CV) against ``n``
    within-group label permutations; (count as extreme + 1) / (n + 1)."""
    rng = np.random.default_rng(seed)
    obs = score_fn(np.asarray(y))
    null = np.array([score_fn(permute_within_groups(y, groups, rng)) for _ in range(n)])
    extreme = null >= obs if greater_is_better else null <= obs
    return {
        "observed": float(obs),
        "null_mean": float(np.mean(null)),
        "p_value": float((extreme.sum() + 1) / (n + 1)),
        "n_permutations": int(n),
    }


def git_commit(root: Path) -> str:
    """Current ``git rev-parse HEAD`` (``"unknown"`` outside a git checkout)."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return res.stdout.strip()


@dataclass
class RunReport:
    """Everything needed to trust (or reproduce) one evaluated run."""

    run_id: str
    model: str
    target: str
    task: str  # "regression" | "classification"
    cv_scheme: str
    group_col: str
    n_groups: int
    n_samples: int
    label_definition: str
    feature_set: list[str]
    config_hash: str
    git_commit: str
    metrics: dict
    ci: dict = field(default_factory=dict)
    permutation: dict = field(default_factory=dict)
    notes: str = ""

    def save(self, out_dir: Path) -> Path:
        """Write ``<out_dir>/<run_id>.json`` and return the path."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True, default=str))
        return path
