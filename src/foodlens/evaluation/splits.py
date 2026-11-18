"""Grouped CV splitters (P5.T2): the ONLY allowed source of train/test splits.

Each splitter returns a list of ``(train_idx, test_idx)`` positional index arrays.
- ``logo_splits``: leave one group out (headline scheme).
- ``nested_inner_splits``: inner GroupKFold on the training groups (tuning only); None when the
  training fold has fewer than ``cv.min_groups_for_tuning`` groups (-> fixed default config).
- ``temporal_holdout_splits``: within EACH session, the first ``cv.temporal_train_frac`` of time
  trains and the rest tests. Same sessions on both sides BY DESIGN (an extrapolation-in-time
  sanity check, never the headline number); there is no time overlap.
- ``random_frame_splits_LEAKY``: shuffled KFold over rows. Leakage demo ONLY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from foodlens.config import Config
from foodlens.errors import SplitError

Splits = list[tuple[np.ndarray, np.ndarray]]


def group_column(df: pd.DataFrame, cfg: Config, prefer: list[str] | None = None) -> str:
    """First column of ``prefer`` (default ``cv.group_col_preference``) that exists and is fully
    non-null in ``df``."""
    for col in prefer or cfg.cv.group_col_preference:
        if col in df.columns and df[col].notna().all():
            return col
    raise SplitError(
        f"no fully non-null group column among {prefer or cfg.cv.group_col_preference}"
    )


def get_groups(df: pd.DataFrame, cfg: Config, prefer: list[str] | None = None) -> np.ndarray:
    """Group label per row (``piece_id`` if fully non-null, else ``session_id``)."""
    return df[group_column(df, cfg, prefer)].astype(str).to_numpy()


def logo_splits(groups: np.ndarray) -> Splits:
    """Leave-one-group-out; needs >= 2 groups."""
    if len(np.unique(groups)) < 2:
        raise SplitError("leave-one-group-out needs at least 2 groups")
    x = np.zeros(len(groups))
    return [(tr, te) for tr, te in LeaveOneGroupOut().split(x, groups=groups)]


def nested_inner_splits(train_groups: np.ndarray, cfg: Config) -> Splits | None:
    """Inner GroupKFold over the training fold's groups (indices relative to the fold), or None
    if it has fewer than ``cv.min_groups_for_tuning`` groups (use a fixed default config)."""
    return inner_group_kfold(train_groups, cfg.cv.inner_splits, cfg.cv.min_groups_for_tuning)


def inner_group_kfold(train_groups: np.ndarray, n_splits: int, min_groups: int) -> Splits | None:
    """GroupKFold with ``min(n_splits, n_groups)`` folds, or None below ``min_groups`` groups.
    Used by ``nested_inner_splits`` and the tuned models of ``models.classical``."""
    n = len(np.unique(train_groups))
    if n < min_groups:
        return None
    x = np.zeros(len(train_groups))
    gkf = GroupKFold(n_splits=min(n_splits, n))
    return [(tr, te) for tr, te in gkf.split(x, groups=train_groups)]


def temporal_holdout_splits(
    df: pd.DataFrame, cfg: Config, session_col: str = "session_id"
) -> Splits:
    """One split: per session, rows with time fraction < ``cv.temporal_train_frac`` of the
    session's span train, the rest test (by ``ts``)."""
    frac = cfg.cv.temporal_train_frac
    ts = pd.to_datetime(df["ts"])
    g = ts.groupby(df[session_col].to_numpy())
    t0 = g.transform("min").to_numpy()
    span = (g.transform("max") - g.transform("min")).dt.total_seconds().to_numpy()
    rel = (ts.to_numpy() - t0).astype("timedelta64[ns]").astype(np.int64) / 1e9
    pos = np.divide(rel, span, out=np.zeros_like(rel, dtype=float), where=span > 0)
    train = np.flatnonzero(pos < frac)
    test = np.flatnonzero(pos >= frac)
    if not len(train) or not len(test):
        raise SplitError("temporal holdout produced an empty train or test set")
    return [(train, test)]


def random_frame_splits_LEAKY(n_rows: int, cfg: Config) -> Splits:  # noqa: N802
    """LEAKY shuffled KFold over rows: neighbouring frames of one piece land in train AND test.
    Used ONLY by ``evaluation.leakage_demo`` to show how much a random split inflates scores."""
    kf = KFold(n_splits=cfg.evaluation.leaky_folds, shuffle=True, random_state=cfg.seed)
    return [(tr, te) for tr, te in kf.split(np.zeros(n_rows))]


def assert_no_group_overlap(splits: Splits, groups: np.ndarray) -> None:
    """Raise ``SplitError`` if any group is in both train and test of a split."""
    for i, (tr, te) in enumerate(splits):
        both = set(groups[tr]) & set(groups[te])
        if both:
            raise SplitError(f"split {i}: groups in train and test: {sorted(both)}")
