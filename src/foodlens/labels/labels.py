"""Label columns per triplet (P5.T1): what the models predict.

- ``hours_since_start``: elapsed clock hours since the session's first frame.
- ``hours_since_piece_start``: clock hours since the first frame of the piece (= the session's
  own start unless ``continues`` chains it to earlier sessions; storage gaps are counted).
- ``equiv_hours_25C``: temperature-equivalent hours at ``thermal.T_ref`` since the piece start
  (``thermal_age``); a chained session adds its parents' total plus the storage gap at
  ``storage_temp_c`` (0 with a warning if unknown). ``equiv_hours_25C_tmin_alt``: same with
  ``thermal.T_min_sensitivity``.
- ``age_h_total``: ``pre_age_h`` + ``hours_since_piece_start``; NaN unless pre_age_h is known.
- ``storage_day``: samples.yaml, else the position in the piece chain when chained, else NA.
- ``freshness_class``: definition A (storage_day) or B (equiv hour thresholds), see config.
- ``post_step``: triplet after a scene step (``scene_segment`` > 0) in one of
  ``evaluation.step_sensitivity_sessions`` (steps that change the meat) = sensitivity case.
Excluded sessions (``exclude: true``) are dropped here with a logged reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foodlens.config import Config, SamplesConfig
from foodlens.errors import LabelError
from foodlens.labels.thermal_age import clock_hours_for, equiv_hours, relative_rate
from foodlens.logging_utils import get_logger

log = get_logger("labels")
LABEL_COLS = [
    "session_id", "triplet", "piece_id", "experiment", "ts", "hours_since_start",
    "hours_since_piece_start", "equiv_hours_25C", "equiv_hours_25C_tmin_alt", "age_h_total",
    "storage_day", "freshness_class", "post_step", "label_definition",
]  # fmt: skip


def label_definition(cfg: Config) -> str:
    """Human-readable label definition stored in every metrics JSON."""
    lab, th = cfg.labels, cfg.thermal
    if lab.class_definition == "A":
        return f"A: storage_day 1..{len(lab.classes)} -> {lab.classes}"
    return (
        f"B: equiv_hours_25C (T_min {th.T_min} °C, T_ref {th.T_ref} °C) thresholds "
        f"{lab.equiv_h_thresholds} h -> {lab.classes}"
    )


def _session_series(s: pd.DataFrame, cfg: Config, t_min: float) -> np.ndarray:
    """Equivalent hours since the session's first FRAME (the first triplet is elapsed_s in)."""
    th = cfg.thermal
    eq = equiv_hours(s["ts"], s["temp_c"], t_min, th.T_ref, th.indoor_offset_c)
    first = relative_rate(s["temp_c"].iloc[0] + th.indoor_offset_c, t_min, th.T_ref)
    return eq + float(first) * s["elapsed_s"].iloc[0] / 3600.0


def _offsets(sid: str, per: dict, samples: SamplesConfig, cfg: Config) -> tuple[float, ...]:
    """(clock h, equiv h, equiv h alt) at the start of ``sid`` since its piece's start."""
    spec = samples.sessions.get(sid)
    if spec is None or spec.continues is None:
        return 0.0, 0.0, 0.0
    parent = spec.continues
    if parent not in per:
        raise LabelError(f"{sid} continues {parent}, which has no triplets (excluded?)")
    pc, pe, pa = _offsets(parent, per, samples, cfg)
    p = per[parent]
    gap_h = (per[sid]["start"] - p["end"]).total_seconds() / 3600.0
    if gap_h < 0:
        raise LabelError(f"{sid} starts before its parent {parent} ends")
    st, th = spec.storage_temp_c, cfg.thermal
    if st is None:
        log.warning("%s: storage_temp_c unknown, storage gap of %.1f h adds 0 equiv h", sid, gap_h)
    rate = 0.0 if st is None else float(relative_rate(st, th.T_min, th.T_ref))
    rate_a = 0.0 if st is None else float(relative_rate(st, th.T_min_sensitivity, th.T_ref))
    clock = pc + p["hours_end"] + gap_h
    return clock, pe + p["eq_end"] + rate * gap_h, pa + p["eqa_end"] + rate_a * gap_h


def assign_class(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Freshness class per row by the configured definition (A: storage_day, B: equiv hours)."""
    lab = cfg.labels
    if lab.class_definition == "A":
        if df["storage_day"].isna().any():
            raise LabelError("class_definition A needs storage_day for every session")
        idx = df["storage_day"].astype(int).clip(1, len(lab.classes)) - 1
    else:
        idx = np.digitize(df["equiv_hours_25C"].to_numpy(), lab.equiv_h_thresholds)
    return pd.Series(np.asarray(lab.classes)[np.asarray(idx)], index=df.index, dtype="string")


def drop_excluded(trip: pd.DataFrame, samples: SamplesConfig) -> pd.DataFrame:
    """Rows of sessions marked ``exclude: true`` removed (reason logged per session)."""
    out = trip
    for sid, spec in sorted(samples.sessions.items()):
        if spec.exclude and (trip["session_id"] == sid).any():
            log.warning("labels: session %s excluded: %s", sid, spec.exclude_reason)
            out = out[out["session_id"] != sid]
    return out


def build_labels(trip: pd.DataFrame, samples: SamplesConfig, cfg: Config) -> pd.DataFrame:
    """Label table (``LABEL_COLS``) for the triplet rows of non-excluded sessions."""
    t = drop_excluded(trip, samples).sort_values(["session_id", "ts"]).reset_index(drop=True)
    if t["temp_c"].isna().any():
        raise LabelError(
            "temp_c missing for some triplets: rerun `foodlens ingest` with temperature"
        )
    th, per = cfg.thermal, {}
    for sid, s in t.groupby("session_id", sort=True):
        eq, eqa = _session_series(s, cfg, th.T_min), _session_series(s, cfg, th.T_min_sensitivity)
        t.loc[s.index, "_eq"], t.loc[s.index, "_eqa"] = eq, eqa
        per[sid] = {"start": s["ts"].iloc[0] - pd.Timedelta(seconds=float(s["elapsed_s"].iloc[0])),
                    "end": s["ts"].iloc[-1], "hours_end": s["elapsed_s"].iloc[-1] / 3600.0,
                    "eq_end": eq[-1], "eqa_end": eqa[-1]}  # fmt: skip
    t["hours_since_start"] = t["elapsed_s"] / 3600.0
    piece, day, pre = {}, {}, {}
    for sid in per:
        chain = samples.chain(sid) if sid in samples.sessions else [sid]
        spec = samples.sessions.get(sid)
        root = samples.sessions.get(chain[0])
        piece[sid] = (spec.piece_id if spec and spec.piece_id else None) or chain[0]
        day[sid] = spec.storage_day if spec and spec.storage_day else (
            len(chain) if len(chain) > 1 else pd.NA)  # fmt: skip
        pre[sid] = root.pre_age_h if root and root.pre_age_h is not None else np.nan
        clock, eq0, eqa0 = _offsets(sid, per, samples, cfg)
        m = t["session_id"] == sid
        t.loc[m, "hours_since_piece_start"] = clock + t.loc[m, "hours_since_start"]
        t.loc[m, "equiv_hours_25C"] = eq0 + t.loc[m, "_eq"]
        t.loc[m, "equiv_hours_25C_tmin_alt"] = eqa0 + t.loc[m, "_eqa"]
    t["piece_id"] = t["session_id"].map(piece).astype("string")
    t["storage_day"] = t["session_id"].map(day).astype("Int64")
    t["age_h_total"] = t["session_id"].map(pre).astype(float) + t["hours_since_piece_start"]
    t["freshness_class"] = assign_class(t, cfg)
    listed = t["session_id"].isin(cfg.evaluation.step_sensitivity_sessions)
    t["post_step"] = listed & (t["scene_segment"].fillna(0) > 0)
    t["label_definition"] = label_definition(cfg)
    return t[LABEL_COLS].reset_index(drop=True)


def level_clock_ranges(temp_c: float, cfg: Config) -> pd.DataFrame:
    """Per freshness level: equivalent-hour range and the clock-hour range at a constant
    ``temp_c`` (°C). Indicative only (the app "i" button); depends on the storage temperature."""
    th, lab = cfg.thermal, cfg.labels
    if lab.class_definition != "B":
        raise LabelError("clock ranges need class_definition B (equivalent hours)")
    edges = [0.0, *lab.equiv_h_thresholds, float("inf")]
    rows = []
    for name, lo, hi in zip(lab.classes, edges[:-1], edges[1:], strict=True):
        rows.append({"level": name, "temp_c": temp_c, "equiv_h_from": lo, "equiv_h_to": hi,
                     "clock_h_from": clock_hours_for(lo, temp_c, th.T_min, th.T_ref),
                     "clock_h_to": clock_hours_for(hi, temp_c, th.T_min, th.T_ref)})  # fmt: skip
    return pd.DataFrame(rows)
