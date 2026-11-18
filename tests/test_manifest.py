"""Tests for io/exif.py and io/manifest.py on a synthetic folder tree (no real data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from foodlens.config import load_config, load_samples
from foodlens.errors import E_NO_SESSION, E_NO_TIME
from foodlens.io.exif import parse_exif_time, parse_filename_time, read_frame_meta
from foodlens.io.manifest import build_manifest, discover_images

TZ = "Asia/Kolkata"
SESSION = {
    "exp1_D1A": {
        "experiment": "EXP-1",
        "folder": "EXP-1_x/exp1_D1A",
        "piece_id": "P1",
        "storage_day": 0,
        "temp": {"mode": "fixed", "value_c": 28.0},
    }
}


def _jpeg(path: Path, when: str | None = None, orientation: int = 1, size=(40, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.full((size[1], size[0], 3), 10, np.uint8))
    exif = Image.Exif()
    if orientation != 1:
        exif[0x0112] = orientation
    if when is not None:
        sub = exif.get_ifd(0x8769)
        sub[36867] = when  # DateTimeOriginal
        sub[37521] = "500000"  # SubsecTimeOriginal -> +0.5 s
        sub[34855] = 100  # ISO
        sub[41987] = 0  # WhiteBalance auto
    img.save(path, exif=exif)


def _tree(root: Path) -> Path:
    s = root / "experiment" / "EXP-1_x" / "exp1_D1A"
    _jpeg(s / "IMG_20260828_120200.jpg", "2026:08:28 12:02:00")
    _jpeg(s / "IMG_20260828_120000.jpg", "2026:08:28 12:00:00")
    _jpeg(s / "IMG_20260828_120100.JPG", "2026:08:28 12:01:00", orientation=6)
    _jpeg(s / "IMG_20260828_120300.jpg")  # no EXIF -> filename time
    _jpeg(s / "photo_bad_name.jpg")  # no EXIF and no time in name -> E_NO_TIME
    _jpeg(root / "experiment" / "misc" / "IMG_20260828_120400.jpg", "2026:08:28 12:04:00")
    _jpeg(s / "old_sorted" / "IMG_20260828_130000.jpg", "2026:08:28 13:00:00")  # ignored
    (s / ".DS_Store").write_bytes(b"x")
    return root / "experiment"


def test_parse_times():
    ts = parse_exif_time("2026:08:28 12:19:50", "765805", TZ)
    assert ts == pd.Timestamp("2026-08-28 12:19:50.765805", tz=TZ)
    assert parse_exif_time("garbage", None, TZ) is None
    cfg = load_config()
    assert parse_filename_time("IMG_20260828_121950", cfg.capture, TZ).hour == 12
    assert parse_filename_time("photo", cfg.capture, TZ) is None


def test_discover_skips_sorted_and_non_images(tmp_path):
    raw = _tree(tmp_path)
    files = discover_images(raw, load_config().capture)
    assert len(files) == 6
    assert not any("_sorted" in str(f.relative_to(raw)) for f in files)


def test_read_frame_meta_orientation_swaps_size(tmp_path):
    p = tmp_path / "a.jpg"
    _jpeg(p, "2026:08:28 12:00:00", orientation=6, size=(40, 30))
    meta = read_frame_meta(p, load_config().capture, TZ)
    assert (meta["width"], meta["height"], meta["orientation"]) == (30, 40, 6)
    assert meta["iso"] == 100 and meta["wb_mode"] == "auto" and meta["ts_source"] == "exif"


def test_build_manifest_columns_and_rejects(tmp_path, write_tmp_config):
    _tree(tmp_path)
    cfg = load_config(write_tmp_config(tmp_path, sessions=SESSION))
    df = build_manifest(cfg, load_samples(cfg))
    assert len(df) == 6
    by_stem = df.set_index(df["path"].map(lambda p: Path(p).stem))
    assert by_stem.loc["photo_bad_name", "ingest_reason"] == E_NO_TIME
    assert by_stem.loc["IMG_20260828_120400", "ingest_reason"] == E_NO_SESSION
    assert by_stem.loc["IMG_20260828_120300", "ts_source"] == "filename"
    ok = df[df["ingest_status"] == "ok"].sort_values("seq_idx")
    assert ok["seq_idx"].tolist() == [0, 1, 2, 3]
    assert ok["schedule_light"].tolist() == ["RED", "GREEN", "BLUE", "RED"]
    assert ok["elapsed_s"].tolist() == [0.0, 60.0, 120.0, 179.5]
    assert ok["frame_id"].iloc[0] == "exp1_D1A__IMG_20260828_120000"
    assert set(ok["day_code"]) == {"D1"} and set(ok["piece_code"]) == {"A"}
    assert set(ok["piece_id"]) == {"P1"} and set(ok["temp_mode"]) == {"fixed"}
    assert ok["path"].iloc[0].startswith("experiment/EXP-1_x/exp1_D1A/")


def test_build_manifest_reuses_cache_and_limit(tmp_path, write_tmp_config):
    _tree(tmp_path)
    cfg = load_config(write_tmp_config(tmp_path, sessions=SESSION))
    samples = load_samples(cfg)
    first = build_manifest(cfg, samples)
    fake = first.copy()
    fake.loc[fake["ingest_status"] == "ok", "iso"] = 999.0  # marker: proves rows were reused
    again = build_manifest(cfg, samples, previous=fake)
    assert (again.loc[again["ingest_status"] == "ok", "iso"] == 999.0).all()
    assert len(build_manifest(cfg, samples, limit=2)) == 2


def test_unknown_session_gets_defaults(tmp_path, write_tmp_config):
    _jpeg(tmp_path / "experiment" / "EXP-3_y" / "exp3_D9C" / "IMG_1.jpg", "2026:08:28 12:00:00")
    cfg = load_config(write_tmp_config(tmp_path, sessions=SESSION))
    df = build_manifest(cfg, load_samples(cfg))
    assert df.loc[0, "session_id"] == "exp3_D9C" and df.loc[0, "experiment"] == "EXP-3"
    assert df.loc[0, "temp_mode"] == "weather" and pd.isna(df.loc[0, "piece_id"])


def test_locate_session_hyphenated_folder():
    """S07: folders exp-5_D1A fall back to the pattern and get the id exp5_D1A."""
    from foodlens.io.manifest import locate_session

    cap = load_config().capture
    exp, sid = locate_session(Path("EXP-5_06-09-2026/exp-5_D1A/IMG_1.jpg"), cap, {})
    assert (exp, sid) == ("EXP-5", "exp5_D1A")
    exp, sid = locate_session(Path("EXP-1_x/exp1_D1B/IMG_1.jpg"), cap, {})
    assert sid == "exp1_D1B"


def test_duplicate_copy_rejected_and_sequence_unchanged(tmp_path, write_tmp_config):
    """S07: a byte-identical "- Copy" file is rejected (E_DUPLICATE) and does not shift lights."""
    import shutil

    from foodlens.errors import E_DUPLICATE

    s = _tree(tmp_path) / "EXP-1_x" / "exp1_D1A"
    shutil.copy(s / "IMG_20260828_120100.JPG", s / "IMG_20260828_120100 - Copy.JPG")
    cfg = load_config(write_tmp_config(tmp_path, sessions=SESSION))
    df = build_manifest(cfg, load_samples(cfg))
    by_stem = df.set_index(df["path"].map(lambda p: Path(p).stem))
    assert by_stem.loc["IMG_20260828_120100 - Copy", "ingest_reason"] == E_DUPLICATE
    assert by_stem.loc["IMG_20260828_120100", "ingest_status"] == "ok"
    ok = df[df["ingest_status"] == "ok"].sort_values("seq_idx")
    assert ok["schedule_light"].tolist() == ["RED", "GREEN", "BLUE", "RED"]
