"""Group consecutive R, G, B frames into triplets with cross-light ratios (P3.T5; P4 reuses).

Within a session, sorted by time, a triplet is a run of frames whose lights follow the capture
order (``capture.light_order``) and whose timestamps span <= ``triplet.max_span_s``. A frame whose
light does not come later in the order than the previous frame's starts a new triplet, so a
missing frame (rejected / dark) never shifts later cycles. Missing lights give a partial triplet
(NaN values, ``has_<light>`` = False). A triplet's time = the mean of its frames' times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATIO_PAIRS = [("RED", "GREEN"), ("GREEN", "BLUE"), ("RED", "BLUE")]


def assign_triplets(df: pd.DataFrame, light_order: list[str], max_span_s: float) -> pd.Series:
    """Triplet index (int, unique within a session) for each row of ``df`` (needs session_id,
    light, ts). Returned in ``df``'s index order."""
    rank = {light: i for i, light in enumerate(light_order)}
    out = pd.Series(-1, index=df.index, dtype=int)
    for _, g in df.sort_values("ts").groupby("session_id"):
        tid, start, last = -1, None, None
        for idx, light, ts in zip(g.index, g["light"], g["ts"], strict=True):
            new = last is None or rank[light] <= last or (ts - start).total_seconds() > max_span_s
            if new:
                tid, start = tid + 1, ts
            out[idx], last = tid, rank[light]
    return out


def triplet_table(df: pd.DataFrame, cols: list[str], light_order: list[str], max_span_s: float):
    """Wide table: one row per session x triplet with ``<col>_<LIGHT>``, ``has_<LIGHT>``, mean
    ``ts`` and ``elapsed_h``."""
    d = df.assign(triplet=assign_triplets(df, light_order, max_span_s))
    keys = ["session_id", "triplet"]
    wide = d.pivot_table(index=keys, columns="light", values=cols, aggfunc="first")
    wide.columns = [f"{c}_{light}" for c, light in wide.columns]
    t = d.groupby(keys).agg(ts=("ts", "mean"), elapsed_h=("elapsed_h", "mean"))
    t = t.join(wide).reset_index()
    for light in light_order:
        present = d[d["light"] == light].set_index(keys).index
        t[f"has_{light}"] = t.set_index(keys).index.isin(present)
    return t


def add_ratios(t: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add ``<col>_ratio_<A><B>`` = col_A / col_B for RATIO_PAIRS (NaN if B is 0 or missing).
    Returns (table, new column names). Ratios compare separately lit frames of one cycle, which
    carries spectral (absorption) information and cancels a common brightness change."""
    t = t.copy()
    new = []
    for c in cols:
        for a, b in RATIO_PAIRS:
            name = f"{c}_ratio_{a[0]}{b[0]}"
            den = t[f"{c}_{b}"].where(t[f"{c}_{b}"] != 0)
            t[name] = t[f"{c}_{a}"] / den
            new.append(name)
    t[new] = t[new].replace([np.inf, -np.inf], np.nan)
    return t, new
