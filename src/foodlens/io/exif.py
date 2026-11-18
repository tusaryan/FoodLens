"""Read EXIF metadata (capture time, ISO, exposure, orientation) from raw images (Phase P1).

Only the file header is read (no full decode), so this is fast enough for every frame.
Widths/heights are reported *after* ``exif_transpose`` (orientations 5-8 swap the axes).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import ExifTags, Image, UnidentifiedImageError

from foodlens.config import CaptureConfig
from foodlens.errors import E_DECODE, E_NO_TIME, DataError

EXIF_IFD = 0x8769  # pointer to the Exif sub-IFD (DateTimeOriginal, ISO, ...)
EXIF_TIME_FORMAT = "%Y:%m:%d %H:%M:%S"
SWAPPED_ORIENTATIONS = frozenset({5, 6, 7, 8})  # EXIF orientations that rotate by 90°
SHA1_CHUNK = 1 << 20  # bytes read per hashing step
_TAG = {name: code for code, name in ExifTags.TAGS.items()}
WB_MODES = {0: "auto", 1: "manual"}
EXPOSURE_MODES = {0: "auto", 1: "manual", 2: "auto_bracket"}


def file_sha1(path: Path) -> str:
    """SHA-1 hex digest of the file bytes (used for change detection / caching)."""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while chunk := fh.read(SHA1_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def parse_exif_time(value: str | None, subsec: str | None, tz: str) -> pd.Timestamp | None:
    """EXIF ``DateTimeOriginal`` (+ ``SubsecTimeOriginal`` digits) -> tz-aware Timestamp."""
    if not value or not str(value).strip():
        return None
    try:
        dt = datetime.strptime(str(value).strip(), EXIF_TIME_FORMAT)
    except ValueError:
        return None
    ts = pd.Timestamp(dt).tz_localize(tz)
    digits = re.sub(r"\D", "", str(subsec or ""))
    if digits:
        ts += pd.Timedelta(seconds=float(f"0.{digits}"))
    return ts


def parse_filename_time(stem: str, cap: CaptureConfig, tz: str) -> pd.Timestamp | None:
    """Capture time from a file name like ``IMG_20260828_121950`` (config pattern), or None."""
    m = re.search(cap.filename_time_pattern, stem)
    if m is None:
        return None
    try:
        dt = datetime.strptime(m.group(1), cap.filename_time_format)
    except ValueError:
        return None
    return pd.Timestamp(dt).tz_localize(tz)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _camera_fields(base: dict, sub: dict) -> dict[str, Any]:
    iso = sub.get(_TAG["ISOSpeedRatings"])
    if isinstance(iso, tuple):
        iso = iso[0] if iso else None
    wb, em = sub.get(_TAG["WhiteBalance"]), sub.get(_TAG["ExposureMode"])
    return {
        "exposure_s": _num(sub.get(_TAG["ExposureTime"])),
        "iso": _num(iso),
        "fnumber": _num(sub.get(_TAG["FNumber"])),
        "wb_mode": WB_MODES.get(wb) if wb is not None else None,
        "exposure_mode": EXPOSURE_MODES.get(em) if em is not None else None,
        "focal_mm": _num(sub.get(_TAG["FocalLength"])),
        "make": base.get(_TAG["Make"]),
        "model": base.get(_TAG["Model"]),
    }


def read_frame_meta(path: Path, cap: CaptureConfig, tz: str) -> dict[str, Any]:
    """Header metadata for one frame.

    Returns ``ts`` (tz-aware), ``ts_source`` (exif/filename), camera fields (None if missing),
    and ``width``/``height``/``orientation`` after ``exif_transpose``.
    Raises ``DataError`` ``E_DECODE`` (not an image) or ``E_NO_TIME`` (no time anywhere).
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
            exif = im.getexif()
    except (UnidentifiedImageError, OSError) as exc:
        raise DataError(E_DECODE, f"{path.name}: {exc}") from exc
    base = dict(exif.items())
    sub = dict(exif.get_ifd(EXIF_IFD).items())
    orientation = int(base.get(_TAG["Orientation"], 1) or 1)
    if orientation in SWAPPED_ORIENTATIONS:
        w, h = h, w
    ts = parse_exif_time(sub.get(_TAG["DateTimeOriginal"]), sub.get(_TAG["SubsecTimeOriginal"]), tz)
    source = "exif"
    if ts is None:
        ts, source = parse_filename_time(path.stem, cap, tz), "filename"
    if ts is None:
        raise DataError(E_NO_TIME, f"{path.name}: no EXIF DateTimeOriginal and no time in name")
    meta = {"ts": ts, "ts_source": source, **_camera_fields(base, sub)}
    meta.update({"width": w, "height": h, "orientation": orientation})
    return meta
