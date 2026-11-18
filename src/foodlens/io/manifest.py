"""Build the frame manifest (one row per image, Parquet) from ``paths.raw_dir`` (Phase P1).

Every raw image gets a row, including rejected ones (``ingest_status='reject'`` plus a code in
``ingest_reason``), so nothing disappears silently. Session ids come from ``samples.yaml``
``folder`` entries first; a folder not listed there falls back to ``capture.session_pattern``
(with a warning and default metadata).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from foodlens.config import CaptureConfig, Config, SamplesConfig
from foodlens.errors import E_DUPLICATE, E_NO_SESSION, ConfigError, DataError
from foodlens.io.exif import file_sha1, read_frame_meta
from foodlens.logging_utils import get_logger

log = get_logger("io.manifest")

META_COLS = [
    "ts",
    "ts_source",
    "exposure_s",
    "iso",
    "fnumber",
    "wb_mode",
    "exposure_mode",
    "focal_mm",
    "make",
    "model",
    "width",
    "height",
    "orientation",
]
SAMPLE_COLS = ["piece_id", "storage_day", "pre_age_h", "temp_mode"]
DAY_PIECE_RE = re.compile(r"_(D\d+)([A-Z])$")  # exp1_D1A -> D1, A


def discover_images(raw_dir: Path, cap: CaptureConfig) -> list[Path]:
    """All image files under ``raw_dir`` (recursive, sorted), skipping ignored folders."""
    exts = {e.lower() for e in cap.image_exts}
    skip = cap.ignore_dir_substr.lower()
    out = []
    for p in raw_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if any(skip in part.lower() for part in p.relative_to(raw_dir).parts[:-1]):
            continue
        out.append(p)
    return sorted(out)


def _rel(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def locate_session(
    rel_to_raw: Path, cap: CaptureConfig, folder_map: dict[str, str]
) -> tuple[str | None, str | None]:
    """(experiment, session_id) for a file path relative to ``raw_dir``; None if unknown."""
    parts = rel_to_raw.parts[:-1]
    exp = next((m.group(0) for p in parts if (m := re.match(cap.experiment_pattern, p))), None)
    session = folder_map.get(Path(*parts).as_posix()) if parts else None
    if session is None:
        found = next((p for p in reversed(parts) if re.fullmatch(cap.session_pattern, p)), None)
        session = normalise_session_id(found) if found else None
    return exp, session


def normalise_session_id(folder: str) -> str:
    """Session id from a folder name: hyphens dropped (``exp-5_D1A`` -> ``exp5_D1A``, S07)."""
    return folder.replace("-", "")


def _frame_row(
    path: Path, cfg: Config, folder_map: dict[str, str], cached: dict[str, dict]
) -> dict[str, Any]:
    """One manifest row. EXIF fields are reused from ``cached`` when the sha1 is unchanged."""
    raw = cfg.path("raw_dir")
    exp, session = locate_session(path.relative_to(raw), cfg.capture, folder_map)
    rel = _rel(path, cfg.root)
    row: dict[str, Any] = {
        "path": rel,
        "sha1": file_sha1(path),
        "experiment": exp,
        "session_id": session,
        "stem": path.stem,
        "ingest_status": "ok",
        "ingest_reason": "",
    }
    if session is None:
        row.update(ingest_status="reject", ingest_reason=E_NO_SESSION)
        return row
    prev = cached.get(rel)
    if prev is not None and prev["sha1"] == row["sha1"]:
        row.update({k: prev[k] for k in META_COLS + ["ingest_status", "ingest_reason"]})
        return row
    try:
        row.update(read_frame_meta(path, cfg.capture, cfg.location.tz))
    except DataError as exc:
        row.update(ingest_status="reject", ingest_reason=exc.code)
    return row


def _add_sequence(df: pd.DataFrame, cap: CaptureConfig) -> pd.DataFrame:
    """seq_idx (0-based by ts within session), elapsed_s and schedule_light for ok rows."""
    df = df.copy()
    df["seq_idx"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["elapsed_s"] = float("nan")
    df["schedule_light"] = None
    ok = df["ingest_status"] == "ok"
    order = df[ok].sort_values(["session_id", "ts", "path"])
    seq = order.groupby("session_id").cumcount()
    start = order.groupby("session_id")["ts"].transform("min")
    df.loc[order.index, "seq_idx"] = seq.astype("Int64")
    df.loc[order.index, "elapsed_s"] = (order["ts"] - start).dt.total_seconds()
    lights = cap.light_order
    df.loc[order.index, "schedule_light"] = [lights[i % len(lights)] for i in seq]
    return df


def _merge_samples(df: pd.DataFrame, samples: SamplesConfig) -> pd.DataFrame:
    df = df.copy()
    for col in SAMPLE_COLS:
        df[col] = None
    for sid in sorted(set(df["session_id"].dropna())):
        mask = df["session_id"] == sid
        spec = samples.sessions.get(sid)
        if spec is None:
            log.warning("session %s not in samples.yaml: using defaults (temp mode weather)", sid)
            df.loc[mask, "temp_mode"] = "weather"
            continue
        df.loc[mask, "piece_id"] = spec.piece_id
        df.loc[mask, "storage_day"] = spec.storage_day
        df.loc[mask, "pre_age_h"] = spec.pre_age_h
        df.loc[mask, "temp_mode"] = spec.temp.mode
    codes = df["session_id"].fillna("").str.extract(DAY_PIECE_RE)
    df["day_code"], df["piece_code"] = codes[0], codes[1]
    return df.astype(
        {
            "piece_id": "string",
            "storage_day": "Int64",
            "pre_age_h": "Float64",
            "temp_mode": "string",
        }
    )


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Reject byte-identical copies (same sha1) of an ok file: the shortest file name is kept
    (``IMG_x.jpg`` over ``IMG_x - Copy.jpg`` / ``IMG_x(1).jpg``), the others get E_DUPLICATE."""
    df = df.copy()
    ok = df[df["ingest_status"] == "ok"]
    order = ok.assign(_n=ok["path"].str.len()).sort_values(["sha1", "_n", "path"])
    dup = order.index[order.duplicated("sha1", keep="first")]
    for i in dup:
        log.warning("ingest reject %s: %s", df.at[i, "path"], E_DUPLICATE)
    df.loc[dup, "ingest_status"], df.loc[dup, "ingest_reason"] = "reject", E_DUPLICATE
    return df


def _finalise(rows: list[dict[str, Any]], cfg: Config, samples: SamplesConfig) -> pd.DataFrame:
    df = mark_duplicates(pd.DataFrame(rows))
    for col in META_COLS:
        if col not in df:
            df[col] = None
    df["ts"] = pd.to_datetime(df["ts"])
    df["frame_id"] = df["session_id"].fillna("unknown") + "__" + df["stem"]
    df = _add_sequence(df, cfg.capture)
    df = _merge_samples(df, samples)
    first = ["frame_id", "path", "sha1", "experiment", "session_id", "day_code", "piece_code"]
    rest = [c for c in df.columns if c not in first and c != "stem"]
    return df[first + rest].sort_values(["session_id", "ts", "path"]).reset_index(drop=True)


def build_manifest(
    cfg: Config,
    samples: SamplesConfig,
    limit: int | None = None,
    previous: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Manifest DataFrame for all (or the first ``limit``) images under ``paths.raw_dir``.

    Rows of ``previous`` whose file sha1 is unchanged are reused instead of re-reading EXIF.
    """
    raw = cfg.path("raw_dir")
    if not raw.is_dir():
        raise ConfigError(f"paths.raw_dir does not exist: {raw}")
    files = discover_images(raw, cfg.capture)[:limit]
    folder_map = {spec.folder: sid for sid, spec in samples.sessions.items()}
    cached: dict[str, dict] = {}
    if previous is not None:
        keep = ["path", "sha1", "ingest_status", "ingest_reason"] + META_COLS
        ok = previous[~previous["ingest_reason"].isin([E_NO_SESSION, E_DUPLICATE])]
        cached = {r["path"]: r for r in ok[keep].to_dict("records")}
    rows = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_frame_row)(p, cfg, folder_map, cached) for p in files
    )
    for r in rows:
        if r["ingest_status"] == "reject":
            log.warning("ingest reject %s: %s", r["path"], r["ingest_reason"])
    return _finalise(list(rows), cfg, samples)
