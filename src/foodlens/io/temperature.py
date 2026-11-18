"""Ambient temperature per frame (weather archive, fixed value or logger CSV, °C) (Phase P1).

Sources implement ``TemperatureSource.get_series(window)`` and return a DataFrame indexed by a
tz-aware timestamp with columns ``temp_c`` (°C) and ``rh`` (%, NaN if unknown).
- ``WeatherSource``: Open-Meteo archive (hourly ``temperature_2m``, ``relative_humidity_2m``),
  cached as JSON under ``<data_dir>/<temperature.cache_subdir>``. ``thermal.indoor_offset_c`` is
  added because outdoor reanalysis is not room temperature.
- ``ManualSource``: the constant ``temp.value_c`` from samples.yaml (mode ``fixed``).
- ``LogSource``: a CSV ``timestamp,temp_c[,rh]`` (mode ``log``).
Per-frame values are linearly interpolated in time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import requests

from foodlens.config import Config, SamplesConfig, TempSpec
from foodlens.errors import E_TEMP_UNAVAILABLE, DataError
from foodlens.logging_utils import get_logger

log = get_logger("io.temperature")
HOURLY_VARS = "temperature_2m,relative_humidity_2m"


@dataclass(frozen=True)
class SessionWindow:
    """Time span (tz-aware) of one capture session."""

    session_id: str
    start: pd.Timestamp
    end: pd.Timestamp


class TemperatureSource(Protocol):
    def get_series(self, window: SessionWindow) -> pd.DataFrame: ...


class ManualSource:
    """Constant temperature (°C) for the whole session; humidity unknown."""

    def __init__(self, value_c: float) -> None:
        self.value_c = value_c

    def get_series(self, window: SessionWindow) -> pd.DataFrame:
        idx = pd.DatetimeIndex([window.start, window.end])
        return pd.DataFrame({"temp_c": [self.value_c] * 2, "rh": [np.nan] * 2}, index=idx)


class LogSource:
    """Logger CSV with columns ``timestamp,temp_c[,rh]`` (naive times are taken as ``tz``)."""

    def __init__(self, path: Path, tz: str) -> None:
        self.path, self.tz = path, tz

    def get_series(self, window: SessionWindow) -> pd.DataFrame:
        if not self.path.is_file():
            raise DataError(E_TEMP_UNAVAILABLE, f"temperature log not found: {self.path}")
        df = pd.read_csv(self.path)
        if not {"timestamp", "temp_c"} <= set(df.columns):
            raise DataError(E_TEMP_UNAVAILABLE, f"{self.path} needs columns timestamp,temp_c")
        ts = pd.to_datetime(df["timestamp"])
        ts = ts.dt.tz_localize(self.tz) if ts.dt.tz is None else ts.dt.tz_convert(self.tz)
        rh = df.get("rh", np.nan)
        out = pd.DataFrame({"temp_c": df["temp_c"].astype(float), "rh": rh})
        return out.set_index(pd.DatetimeIndex(ts)).sort_index()


class WeatherSource:
    """Open-Meteo hourly archive with a JSON file cache (no network when cached)."""

    def __init__(self, cfg: Config, session: requests.Session | None = None) -> None:
        self.cfg = cfg
        self.http = session or requests.Session()
        self.cache_dir = cfg.path("data_dir") / cfg.temperature.cache_subdir

    def _cache_file(self, start: str, end: str) -> Path:
        loc = self.cfg.location
        return self.cache_dir / f"{loc.lat}_{loc.lon}_{start}_{end}.json"

    def _download(self, start: str, end: str) -> dict:
        loc, t = self.cfg.location, self.cfg.temperature
        params = {"latitude": loc.lat, "longitude": loc.lon, "start_date": start,
                  "end_date": end, "hourly": HOURLY_VARS, "timezone": loc.tz}  # fmt: skip
        last: Exception | None = None
        for attempt in range(t.retries + 1):
            try:
                resp = self.http.get(t.archive_url, params=params, timeout=t.timeout_s)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                log.warning("weather download attempt %d failed: %s", attempt + 1, exc)
        raise DataError(
            E_TEMP_UNAVAILABLE,
            f"weather archive unreachable and no cache for {start}..{end} ({last}). "
            "Set temp.mode: fixed with temp.value_c (manual) or mode: log in samples.yaml.",
        )

    def get_series(self, window: SessionWindow) -> pd.DataFrame:
        start, end = window.start.strftime("%Y-%m-%d"), window.end.strftime("%Y-%m-%d")
        cache = self._cache_file(start, end)
        if cache.is_file():
            data = json.loads(cache.read_text())
        else:
            data = self._download(start, end)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        hourly = data.get("hourly") or {}
        if "time" not in hourly or "temperature_2m" not in hourly:
            raise DataError(E_TEMP_UNAVAILABLE, f"unexpected weather response in {cache}")
        idx = pd.DatetimeIndex(pd.to_datetime(hourly["time"])).tz_localize(self.cfg.location.tz)
        temp = np.asarray(hourly["temperature_2m"], dtype=float)
        rh = np.asarray(hourly.get("relative_humidity_2m", [np.nan] * len(idx)), dtype=float)
        temp = temp + self.cfg.thermal.indoor_offset_c
        return pd.DataFrame({"temp_c": temp, "rh": rh}, index=idx)


def make_source(spec: TempSpec | None, cfg: Config) -> TemperatureSource:
    """Source for a session's ``temp`` spec (None -> weather)."""
    if spec is None or spec.mode == "weather":
        return WeatherSource(cfg)
    if spec.mode == "fixed":
        return ManualSource(float(spec.value_c))
    path = Path(spec.log_file)
    return LogSource(path if path.is_absolute() else cfg.root / path, cfg.location.tz)


def interpolate_at(series: pd.DataFrame, ts: pd.Series) -> pd.DataFrame:
    """Linear-in-time interpolation of ``temp_c``/``rh`` at frame times (clamped at the ends)."""
    x = series.index.as_unit("ns").asi8.astype(float)
    xi = pd.DatetimeIndex(ts).as_unit("ns").asi8.astype(float)
    out = {}
    for col in ("temp_c", "rh"):
        y = series[col].to_numpy(dtype=float)
        ok = ~np.isnan(y)
        out[col] = np.interp(xi, x[ok], y[ok]) if ok.any() else np.full(len(xi), np.nan)
    return pd.DataFrame(out, index=ts.index)


def attach_temperature(
    manifest: pd.DataFrame, cfg: Config, samples: SamplesConfig, sources: dict | None = None
) -> pd.DataFrame:
    """Add ``temp_c`` (°C) and ``rh`` (%) per ingest-ok frame. ``sources`` overrides per session."""
    df = manifest.copy()
    df["temp_c"], df["rh"] = np.nan, np.nan
    ok = df[(df["ingest_status"] == "ok") & df["session_id"].notna()]
    for sid, g in ok.groupby("session_id", sort=True):
        window = SessionWindow(sid, g["ts"].min(), g["ts"].max())
        spec = samples.sessions[sid].temp if sid in samples.sessions else None
        src = (sources or {}).get(sid) or make_source(spec, cfg)
        vals = interpolate_at(src.get_series(window), g["ts"])
        df.loc[g.index, ["temp_c", "rh"]] = vals.to_numpy()
        log.info("%s: temp %.1f-%.1f °C", sid, vals["temp_c"].min(), vals["temp_c"].max())
    return df
