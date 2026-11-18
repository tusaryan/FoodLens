"""Per-session inventory table (``reports/tables/inventory.csv``) from the manifest (Phase P1)."""

from __future__ import annotations

import pandas as pd

CAMERA_COLS = ["exposure_s", "iso", "fnumber", "wb_mode", "exposure_mode", "focal_mm", "model"]
SECONDS_PER_HOUR = 3600.0


def _uniques(s: pd.Series) -> str:
    vals = sorted({str(v) for v in s.dropna()})
    return "|".join(vals)


def inventory_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per session: n frames, n ingest rejects, start, end, duration (h), frames per
    schedule light (and per pixel light if QC ran), max capture gap (s), unique camera settings.
    """
    df = manifest[manifest["session_id"].notna()]
    rows = []
    for sid, g in df.groupby("session_id", sort=True):
        ok = g[g["ingest_status"] == "ok"].sort_values("ts")
        row = {
            "session_id": sid,
            "experiment": g["experiment"].iloc[0],
            "n_frames": len(g),
            "n_ingest_reject": int((g["ingest_status"] != "ok").sum()),
            "start": ok["ts"].min(),
            "end": ok["ts"].max(),
            "duration_h": (ok["ts"].max() - ok["ts"].min()).total_seconds() / SECONDS_PER_HOUR,
            "max_gap_s": ok["ts"].diff().dt.total_seconds().max(),
        }
        for light, n in ok["schedule_light"].value_counts().sort_index().items():
            row[f"n_sched_{light}"] = int(n)
        if "pixel_light" in ok:
            for light, n in ok["pixel_light"].value_counts().sort_index().items():
                row[f"n_pixel_{light}"] = int(n)
        for col in CAMERA_COLS:
            row[col] = _uniques(g[col])
        rows.append(row)
    out = pd.DataFrame(rows)
    count_cols = [c for c in out.columns if c.startswith(("n_sched_", "n_pixel_"))]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    return out
