"""Feature sanity report (P4.T4): missing values, constant columns, redundant pairs.

Nothing is dropped here: redundant / constant columns are only REPORTED. Dropping happens later
inside each training fold's sklearn Pipeline (AGENTS.md §1), never globally on all data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foodlens.features.schema import SEP


def column_qc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per column: n, % NaN, number of distinct non-NaN values, constant flag."""
    x = df[cols]
    nun = x.nunique(dropna=True)
    return pd.DataFrame(
        {
            "column": cols,
            "n": len(x),
            "pct_nan": (100.0 * x.isna().mean()).to_numpy(),
            "n_unique": nun.to_numpy(),
            "constant": (nun <= 1).to_numpy(),
        }
    )


def high_corr_pairs(df: pd.DataFrame, cols: list[str], corr_max: float) -> pd.DataFrame:
    """Pairs of ``cols`` with |Pearson r| > ``corr_max`` (pairwise-complete rows)."""
    usable = [c for c in cols if df[c].nunique(dropna=True) > 1]
    r = df[usable].corr().to_numpy()
    iu = np.triu_indices_from(r, k=1)
    a = np.abs(r[iu])
    hit = a > corr_max
    return pd.DataFrame(
        {
            "a": np.asarray(usable)[iu[0][hit]],
            "b": np.asarray(usable)[iu[1][hit]],
            "abs_r": a[hit],
        }
    ).sort_values("abs_r", ascending=False, ignore_index=True)


def feature_qc(
    frame: pd.DataFrame,
    frame_cols: list[str],
    trip: pd.DataFrame,
    trip_cols: list[str],
    corr_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(QC table, redundant pairs). Frame rows are per feature x light; triplet rows add the
    number of |r| > ``corr_max`` partners and the strongest partner."""
    parts = []
    for light, g in frame.groupby("light"):
        parts.append(column_qc(g, frame_cols).assign(level="frame", light=light))
    tq = column_qc(trip, trip_cols).assign(level="triplet")
    tq["light"] = [c.split(SEP)[1] if c.count(SEP) >= 1 else "" for c in trip_cols]
    pairs = high_corr_pairs(trip, trip_cols, corr_max)
    both = pd.concat(
        [pairs.rename(columns={"a": "column", "b": "partner"}),
         pairs.rename(columns={"b": "column", "a": "partner"})]
    )  # fmt: skip
    n_hi = both.groupby("column").size()
    best = both.sort_values("abs_r", ascending=False).drop_duplicates("column").set_index("column")
    tq["n_high_corr"] = tq["column"].map(n_hi).fillna(0).astype(int)
    tq["max_abs_r_partner"] = tq["column"].map(best["partner"])
    tq["max_abs_r"] = tq["column"].map(best["abs_r"])
    out = pd.concat(parts + [tq], ignore_index=True)
    first = ["level", "column", "light"]
    return out[first + [c for c in out.columns if c not in first]], pairs
