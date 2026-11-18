"""Scene-change, drift and scene-step QC from the per-frame ROI statistics (Phase P2.T4).

All statistics are computed per session x light on time-sorted frames, so a light whose masks
leak into the lit tray (BLUE) is only compared with itself.

- **Scene change** (transient outlier, reject ``E_SCENE_CHANGE``): robust z of each feature
  against a centred rolling median (``qc.scene_window`` frames), scaled by
  max(1.4826 x MAD of the residuals, a per-feature floor). |z| > ``qc.scene_z`` on any feature.
  Features: log lit-channel global mean and log mean of the lit channel inside the session
  ROI (all lights), plus the per-frame mask's log area fraction and centroid x / y (px at work
  resolution) for ``roi.ref_lights`` only: GREEN/BLUE per-frame masks join lit tray / box strips
  from frame to frame (S05: D1A BLUE 14:27:51), so their geometry is not a scene signal.
  A rolling (not session) median follows slow trends (D2B meat settling) and persistent steps.
- **Drift** ``drift_px``: distance (px at work resolution) of the per-frame ROI centroid from
  the session x light median centroid.
- **Scene step** (persistent shift, kept): the median log global mean of the ``qc.step_k`` frames
  after a frame differs by >= ``qc.step_min_log`` from the ``step_k`` frames before. Steps of
  different lights within ``qc.step_merge_s`` seconds are one session step; ``scene_segment``
  counts the session steps at or before each frame (0 = before any step).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foodlens.config import Config, QCConfig
from foodlens.logging_utils import get_logger

log = get_logger("qc.scene_qc")

MAD_TO_SD = 1.4826  # MAD -> standard deviation for normal data
STEP_TIE_TOL = 1e-9  # |shift| values this close count as tied (log units)
GEOMETRY = ("log_area", "cx_px", "cy_px")
INTENSITY = ("log_mean", "log_roi")
FEATURES = (*GEOMETRY, *INTENSITY)
SCENE_COLS = ["scene_z", "scene_feature", "scene_change", "drift_px", "scene_segment"]


def local_robust_z(x: pd.Series, window: int, floor: float) -> pd.Series:
    """(x - centred rolling median) / max(1.4826 x MAD of the residuals, ``floor``).

    ``x`` must be time-sorted. NaN stays NaN; same units as ``floor`` before scaling.
    """
    med = x.rolling(window, center=True, min_periods=1).median()
    resid = x - med
    if resid.notna().sum() == 0:
        return resid
    mad = float(np.nanmedian(np.abs(resid - np.nanmedian(resid))))
    return resid / max(MAD_TO_SD * mad, floor)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The scene features from ROI-index + QC-stats columns.

    Needs ``roi_area_frac`` (0-1), ``roi_cx``/``roi_cy`` (fractions of width/height),
    ``work_w``/``work_h`` (px), ``lit_mean`` (lit-channel global mean, 0-255) and
    ``sroi_mean`` (lit-channel mean inside the session ROI, 0-255; a median of uint8
    values is quantised to whole grey levels, ~8 % at the dim BLUE level of ~12).
    """
    return pd.DataFrame(
        {
            "log_area": np.log(df["roi_area_frac"].clip(lower=1e-6)),
            "cx_px": df["roi_cx"] * df["work_w"],
            "cy_px": df["roi_cy"] * df["work_h"],
            "log_mean": np.log(df["lit_mean"].clip(lower=1e-3)),
            "log_roi": np.log(df["sroi_mean"].clip(lower=1e-3)),
        },
        index=df.index,
    )


def detect_steps(x: np.ndarray, k: int, min_log: float) -> list[tuple[int, float]]:
    """Persistent level shifts in a time-sorted series ``x`` (log units).

    Returns (index of the first frame after the step, shift) pairs. A step at i needs
    |median(x[i:i+k]) - median(x[i-k:i])| >= ``min_log``; among adjacent candidate indices the
    largest |shift| wins (the middle one if several tie). A single-frame spike does not move a
    median of k >= 3 frames.
    """
    n = len(x)
    shifts = np.full(n, np.nan)
    for i in range(k, n - k + 1):
        shifts[i] = np.median(x[i : i + k]) - np.median(x[i - k : i])
    cand = np.flatnonzero(np.abs(np.nan_to_num(shifts)) >= min_log)
    steps: list[tuple[int, float]] = []
    for run in np.split(cand, np.flatnonzero(np.diff(cand) > 1) + 1) if len(cand) else []:
        mag = np.abs(shifts[run])
        top = run[mag >= mag.max() - STEP_TIE_TOL]  # a clean step ties at i-1, i, i+1
        best = int(top[len(top) // 2])
        steps.append((best, float(shifts[best])))
    return steps


def session_steps(df: pd.DataFrame, cfg: QCConfig) -> pd.DataFrame:
    """One row per session step: session_id, step_ts (earliest first-after frame), lights,
    shift_log (per light, ``LIGHT:+0.25``). ``df`` needs session_id, light, ts, lit_mean."""
    found = []
    for (sid, light), g in df.sort_values("ts").groupby(["session_id", "light"], sort=True):
        x = np.log(g["lit_mean"].clip(lower=1e-3).to_numpy(dtype=float))
        for i, shift in detect_steps(x, cfg.step_k, cfg.step_min_log):
            found.append({"session_id": sid, "ts": g["ts"].iloc[i], "light": light, "s": shift})
    cols = ["session_id", "step_ts", "lights", "shift_log"]
    if not found:
        return pd.DataFrame(columns=cols)
    groups: list[list[dict]] = []
    for rec in pd.DataFrame(found).sort_values(["session_id", "ts"]).to_dict("records"):
        prev = groups[-1][0] if groups else None
        same = prev is not None and prev["session_id"] == rec["session_id"]
        if same and (rec["ts"] - prev["ts"]).total_seconds() <= cfg.step_merge_s:
            groups[-1].append(rec)
        else:
            groups.append([rec])
    rows = [
        {
            "session_id": grp[0]["session_id"],
            "step_ts": grp[0]["ts"],
            "lights": ",".join(sorted(r["light"] for r in grp)),
            "shift_log": ";".join(f"{r['light']}:{r['s']:+.3f}" for r in grp),
        }
        for grp in groups
    ]
    return pd.DataFrame(rows, columns=cols)


def _segments(df: pd.DataFrame, steps: pd.DataFrame) -> pd.Series:
    seg = pd.Series(0, index=df.index, dtype=int)
    for sid, ts in zip(steps["session_id"], steps["step_ts"], strict=True):
        seg += ((df["session_id"] == sid) & (df["ts"] >= ts)).astype(int)
    return seg


def scene_qc(
    df: pd.DataFrame, cfg: QCConfig, ref_lights: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SCENE_COLS for every row of ``df`` (index kept) and the session-step table.

    Geometry features are scored only for ``ref_lights`` (``roi.ref_lights``).

    ``df``: one row per frame with session_id, light, ts and the ``feature_frame`` inputs.
    ``scene_feature`` names the feature with the largest |z|.
    """
    feats = feature_frame(df)
    floors = {
        "log_area": cfg.scene_floor_log_area,
        "cx_px": cfg.scene_floor_px,
        "cy_px": cfg.scene_floor_px,
        "log_mean": cfg.scene_floor_log_mean,
        "log_roi": cfg.scene_floor_log_mean,
    }
    z = pd.DataFrame(np.nan, index=df.index, columns=list(FEATURES))
    drift = pd.Series(np.nan, index=df.index)
    for (_, light), g in df.sort_values("ts").groupby(["session_id", "light"], sort=True):
        for f in FEATURES if light in ref_lights else INTENSITY:
            z.loc[g.index, f] = local_robust_z(feats.loc[g.index, f], cfg.scene_window, floors[f])
        c = feats.loc[g.index, ["cx_px", "cy_px"]]
        drift.loc[g.index] = np.hypot(*(c - c.median()).to_numpy().T)
    absz = z.abs()
    out = pd.DataFrame(index=df.index)
    out["scene_z"] = absz.max(axis=1, skipna=True)
    out["scene_feature"] = absz.fillna(-1).idxmax(axis=1).where(out["scene_z"].notna(), "")
    out["scene_change"] = out["scene_z"] > cfg.scene_z
    out["drift_px"] = drift
    steps = session_steps(df, cfg)
    out["scene_segment"] = _segments(df, steps)
    for fid, zz, f in zip(df["frame_id"], out["scene_z"], out["scene_feature"], strict=True):
        if zz > cfg.scene_z:
            log.info("scene change %s: |z|=%.1f on %s", fid, zz, f)
    return out[SCENE_COLS], steps


LIT_MEAN_COL = {"RED": "global_mean_r", "GREEN": "global_mean_g", "BLUE": "global_mean_b"}
ROI_INPUT_COLS = [
    "frame_id",
    "roi_ok",
    "roi_area_frac",
    "roi_cx",
    "roi_cy",
    "work_w",
    "work_h",
    "sroi_mean",
    "lap_var_roi",
]
ROI_QC_COLS = ["roi_ok", "sroi_mean", "lap_var_roi", *SCENE_COLS]


def roi_qc_table(
    manifest: pd.DataFrame, stats: pd.DataFrame, roi: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-frame ROI QC columns (``frame_id`` + ROI_QC_COLS) and the session-step table.

    ``manifest``: frame_id, session_id, ts; ``stats``: frame stats (pixel_light, global means);
    ``roi``: the ROI index (ROI_INPUT_COLS). Only frames present in ``roi`` get a row; scene
    statistics use the ``roi_ok`` frames with a valid pixel light, others get NaN / False.
    Raises ``KeyError`` if ``roi`` lacks a ROI_INPUT_COLS column (index from an older version).
    """
    missing = [c for c in ROI_INPUT_COLS if c not in roi]
    if missing:
        raise KeyError(f"roi_index lacks {missing}: rerun `foodlens roi`")
    base = manifest[["frame_id", "session_id", "ts"]].merge(
        stats[["frame_id", "pixel_light", *LIT_MEAN_COL.values()]], on="frame_id"
    )
    m = base.merge(roi[ROI_INPUT_COLS], on="frame_id", how="inner")
    m["roi_ok"] = m["roi_ok"].astype(bool)
    ok = m[m["roi_ok"] & m["pixel_light"].isin(list(LIT_MEAN_COL))].reset_index(drop=True)
    ok["light"] = ok["pixel_light"]
    ok["lit_mean"] = [ok.at[i, LIT_MEAN_COL[li]] for i, li in enumerate(ok["light"])]
    scene, steps = scene_qc(ok, cfg.qc, list(cfg.roi.ref_lights))
    scored = pd.concat([ok[["frame_id"]], scene], axis=1)
    out = m[["frame_id", "roi_ok", "sroi_mean", "lap_var_roi"]].merge(
        scored, on="frame_id", how="left"
    )
    out["scene_change"] = out["scene_change"].astype("boolean").fillna(False).astype(bool)
    out["scene_feature"] = out["scene_feature"].fillna("")
    return out[["frame_id", *ROI_QC_COLS]], steps
