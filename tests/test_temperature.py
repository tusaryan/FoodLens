"""Tests for io/temperature.py: mocked HTTP (never the network), manual, log, offline error."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from foodlens.config import load_config, load_samples
from foodlens.errors import E_TEMP_UNAVAILABLE, DataError
from foodlens.io.temperature import (
    LogSource,
    ManualSource,
    SessionWindow,
    WeatherSource,
    attach_temperature,
)

TZ = "Asia/Kolkata"
WIN = SessionWindow(
    "s", pd.Timestamp("2026-08-28 12:00", tz=TZ), pd.Timestamp("2026-08-28 14:00", tz=TZ)
)
PAYLOAD = {
    "hourly": {
        "time": ["2026-08-28T12:00", "2026-08-28T13:00", "2026-08-28T14:00"],
        "temperature_2m": [30.0, 32.0, 34.0],
        "relative_humidity_2m": [70.0, 60.0, 50.0],
    }
}


class _Resp:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return PAYLOAD


class _Http:
    def __init__(self, fail: bool = False) -> None:
        self.calls, self.fail = 0, fail

    def get(self, url, params, timeout):
        self.calls += 1
        if self.fail:
            raise requests.ConnectionError("offline")
        assert params["hourly"] == "temperature_2m,relative_humidity_2m" and timeout == 20
        return _Resp()


def _cfg(tmp_path, write_tmp_config, offset=0.0):
    return load_config(write_tmp_config(tmp_path, thermal={"indoor_offset_c": offset}))


def test_weather_source_mocked_and_cached(tmp_path, write_tmp_config):
    cfg = _cfg(tmp_path, write_tmp_config, offset=-2.0)
    http = _Http()
    s = WeatherSource(cfg, session=http).get_series(WIN)
    assert s["temp_c"].tolist() == [28.0, 30.0, 32.0] and s["rh"].iloc[0] == 70.0
    assert list((tmp_path / "data" / "cache" / "weather").glob("*_2026-08-28_2026-08-28.json"))
    WeatherSource(cfg, session=_Http(fail=True)).get_series(WIN)  # served from cache
    assert http.calls == 1


def test_weather_offline_without_cache_raises(tmp_path, write_tmp_config):
    cfg = _cfg(tmp_path, write_tmp_config)
    http = _Http(fail=True)
    with pytest.raises(DataError) as exc:
        WeatherSource(cfg, session=http).get_series(WIN)
    assert exc.value.code == E_TEMP_UNAVAILABLE and "fixed" in exc.value.detail
    assert http.calls == cfg.temperature.retries + 1


def test_manual_and_log_sources(tmp_path):
    assert ManualSource(27.5).get_series(WIN)["temp_c"].tolist() == [27.5, 27.5]
    csv = tmp_path / "log.csv"
    csv.write_text("timestamp,temp_c,rh\n2026-08-28 12:00,25,40\n2026-08-28 14:00,27,50\n")
    s = LogSource(csv, TZ).get_series(WIN)
    assert s["temp_c"].tolist() == [25.0, 27.0]
    with pytest.raises(DataError):
        LogSource(tmp_path / "missing.csv", TZ).get_series(WIN)


def test_attach_temperature_interpolates(tmp_path, write_tmp_config):
    cfg = _cfg(tmp_path, write_tmp_config)
    ts = pd.to_datetime(["2026-08-28 12:30", "2026-08-28 13:30", None]).tz_localize(TZ)
    man = pd.DataFrame({"session_id": ["exp1_D1A"] * 3, "ts": ts,
                        "ingest_status": ["ok", "ok", "reject"]})  # fmt: skip
    src = WeatherSource(cfg, session=_Http())
    out = attach_temperature(man, cfg, load_samples(cfg), sources={"exp1_D1A": src})
    assert out["temp_c"].iloc[:2].tolist() == [31.0, 33.0]
    assert out["rh"].iloc[0] == 65.0 and pd.isna(out["temp_c"].iloc[2])
