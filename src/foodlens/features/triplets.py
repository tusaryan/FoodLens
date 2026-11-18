"""Group consecutive R, G, B frames into triplets with cross-light ratios (P3.T5, P4.T3).

Within a session, sorted by time, a triplet is a run of frames whose lights follow the capture
order (``capture.light_order``) and whose timestamps span <= ``triplet.max_span_s``. A frame whose
light does not come later in the order than the previous frame's starts a new triplet, so a
missing frame (rejected / dark) never shifts later cycles. Missing lights give a partial triplet
(NaN values, ``has_<light>`` = False). A triplet's time = the mean of its frames' times.

P4 (``wide_triplets`` / ``add_cross_light`` / ``add_relative``): the model table uses
``<feature>__<LIGHT>`` columns (``schema.SEP``), every light's columns exist even if no frame of
that light is present, and ``__rel`` = value / value in the session's first COMPLETE triplet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foodlens.features.schema import RATIO_PAIRS, SEP, ratio_names


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


def _meta(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = d.groupby(keys)
    t = g.agg(
        experiment=("experiment", "first"),
        piece_id=("piece_id", "first"),
        ts=("ts", "mean"),
        elapsed_s=("elapsed_s", "mean"),
        temp_c=("temp_c", "mean"),
        scene_segment=("scene_segment", "max"),
        n_lights=("light", "nunique"),
        blur_nan_any=("blur_nan", "any"),
    )
    t["elapsed_h"] = t["elapsed_s"] / 3600.0
    return t


def wide_triplets(frames: pd.DataFrame, cols: list[str], lights: list[str], max_span_s: float):
    """One row per session x triplet: meta, ``has_<L>``, ``frame_id__<L>``, ``<col>__<L>``.

    ``frames`` needs session_id, experiment, piece_id, light, ts, elapsed_s, temp_c,
    scene_segment, blur_nan, frame_id and ``cols``."""
    keys = ["session_id", "triplet"]
    d = frames.assign(triplet=assign_triplets(frames, lights, max_span_s))
    wide = d.pivot(index=keys, columns="light", values=["frame_id", *cols])
    full_cols = pd.MultiIndex.from_product([["frame_id", *cols], lights])
    wide = wide.reindex(columns=full_cols)
    wide.columns = [f"{c}{SEP}{li}" for c, li in wide.columns]
    t = _meta(d, keys).join(wide)
    for li in lights:
        t[f"has_{li}"] = t[f"frame_id{SEP}{li}"].notna()
    t["is_complete"] = t[[f"has_{li}" for li in lights]].all(axis=1)
    t = t.reset_index()
    t[cols_of(cols, lights)] = t[cols_of(cols, lights)].astype(float)
    return t.sort_values(["session_id", "ts"]).reset_index(drop=True)


def cols_of(cols: list[str], lights: list[str]) -> list[str]:
    """``<col>__<L>`` names for every col x light."""
    return [f"{c}{SEP}{li}" for c in cols for li in lights]


def add_cross_light(t: pd.DataFrame, ratio_src: list[str], log_src: list[str]) -> pd.DataFrame:
    """Add ``<src>__ratio_<A><B>`` (A / B) and ``<src>__logratio_<A><B>`` (ln A / B) per
    RATIO_PAIRS; NaN where a value is missing, B <= 0 or (log) A <= 0."""
    t = t.copy()
    for kind, srcs in (("ratio", ratio_src), ("logratio", log_src)):
        for src in srcs:
            for name, (a, b) in zip(ratio_names(src, kind), RATIO_PAIRS, strict=True):
                num, den = t[f"{src}{SEP}{a}"], t[f"{src}{SEP}{b}"]
                r = num / den.where(den > 0)
                t[name] = np.log(r.where(r > 0)) if kind == "logratio" else r
    return t


def add_relative(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add ``<col>__rel`` = col / col in the session's first complete triplet (by time; per
    column the first complete triplet where it is not NaN). NaN if there is no such triplet or
    the base is 0. EDA / experiments only (needs the start of the session)."""
    t = t.sort_values(["session_id", "ts"]).copy()
    base = t[t["is_complete"]].groupby("session_id")[cols].first()
    b = base.reindex(t["session_id"]).set_axis(t.index)
    rel = t[cols] / b.where(b != 0)
    rel.columns = [f"{c}{SEP}rel" for c in cols]
    return pd.concat([t, rel.replace([np.inf, -np.inf], np.nan)], axis=1)
