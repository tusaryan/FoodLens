"""P5.T1 labels: thermal age (hand calculations), piece chains, exclusion, freshness levels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foodlens.config import load_config
from foodlens.errors import ConfigError, LabelError
from foodlens.labels.labels import build_labels, level_clock_ranges
from foodlens.labels.thermal_age import clock_hours_for, equiv_hours, relative_rate
from foodlens.samples_config import SamplesConfig

CFG = load_config()


def _ts(n: int, step_min: float = 60.0, start: str = "2026-09-01 10:00") -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq=pd.Timedelta(minutes=step_min)))


def test_constant_25c_equals_clock_hours():
    eq = equiv_hours(_ts(4), pd.Series([25.0] * 4), t_min=-8.0, t_ref=25.0)
    assert np.allclose(eq, [0, 1, 2, 3])


def test_33c_one_hour_hand_calculation():
    eq = equiv_hours(_ts(2), pd.Series([33.0, 33.0]), t_min=-8.0, t_ref=25.0)
    assert eq[-1] == pytest.approx((41 / 33) ** 2, rel=1e-9)  # ≈ 1.544


def test_below_t_min_contributes_zero_and_offset_applies():
    assert relative_rate(-9.0, -8.0, 25.0) == 0.0
    eq = equiv_hours(_ts(2), pd.Series([20.0, 20.0]), -8.0, 25.0, offset_c=5.0)
    assert eq[-1] == pytest.approx(1.0)


def test_equiv_hours_rejects_nan_and_unsorted():
    with pytest.raises(ValueError):
        equiv_hours(_ts(2), pd.Series([25.0, np.nan]), -8.0, 25.0)
    with pytest.raises(ValueError):
        equiv_hours(_ts(3)[::-1].reset_index(drop=True), pd.Series([25.0] * 3), -8.0, 25.0)


def test_clock_hours_for():
    assert clock_hours_for(2.0, 25.0, -8.0, 25.0) == pytest.approx(2.0)
    assert clock_hours_for(2.0, -20.0, -8.0, 25.0) == float("inf")


def _samples(**extra) -> SamplesConfig:
    base = {"experiment": "E", "folder": "x", "temp": {"mode": "weather"}}
    sessions = {"s1": dict(base), "s2": dict(base)}
    for sid, kv in extra.items():
        sessions[sid].update(kv)
    return SamplesConfig.model_validate({"sessions": sessions})


def _trip(sid: str, n: int, start: str, temp: float = 25.0) -> pd.DataFrame:
    ts = _ts(n, 60.0, start)
    return pd.DataFrame({"session_id": sid, "experiment": "E", "triplet": range(n), "ts": ts,
                         "elapsed_s": (ts - ts.iloc[0]).dt.total_seconds(), "temp_c": temp,
                         "scene_segment": 0.0})  # fmt: skip


def test_build_labels_standalone_sessions_and_classes():
    trip = pd.concat([_trip("s1", 4, "2026-09-01 10:00"), _trip("s2", 15, "2026-09-02 10:00")])
    lab = build_labels(trip, _samples(), CFG)
    s2 = lab[lab["session_id"] == "s2"]
    assert list(lab["piece_id"].unique()) == ["s1", "s2"]  # every session its own piece
    assert np.allclose(s2["equiv_hours_25C"], np.arange(15))
    assert s2["freshness_class"].tolist()[:3] == ["fresh", "fresh", "use_soon"]
    assert s2["freshness_class"].iloc[-1] == "spoiled_risk"
    assert lab["age_h_total"].isna().all() and lab["storage_day"].isna().all()
    assert lab["label_definition"].str.startswith("B:").all()


def test_build_labels_chain_continues_piece():
    smp = _samples(s2={"continues": "s1", "storage_temp_c": 4.0})
    trip = pd.concat([_trip("s1", 4, "2026-09-01 10:00"), _trip("s2", 2, "2026-09-02 10:00")])
    lab = build_labels(trip, smp, CFG).set_index(["session_id", "triplet"])
    assert lab.loc[("s2", 0), "piece_id"] == "s1"
    assert lab.loc[("s2", 0), "storage_day"] == 2
    gap_h = 21.0  # s1 ends 13:00, s2 starts 10:00 next day
    want = 3.0 + relative_rate(4.0, CFG.thermal.T_min, CFG.thermal.T_ref) * gap_h
    assert lab.loc[("s2", 0), "equiv_hours_25C"] == pytest.approx(want)
    assert lab.loc[("s2", 1), "hours_since_piece_start"] == pytest.approx(3 + gap_h + 1)


def test_excluded_session_dropped_and_needs_reason():
    trip = pd.concat([_trip("s1", 3, "2026-09-01 10:00"), _trip("s2", 3, "2026-09-02 10:00")])
    lab = build_labels(trip, _samples(s2={"exclude": True, "exclude_reason": "bad"}), CFG)
    assert set(lab["session_id"]) == {"s1"}
    with pytest.raises(Exception, match="exclude_reason"):
        _samples(s2={"exclude": True})


def test_chain_validation_and_missing_temperature():
    with pytest.raises(Exception, match="unknown session"):
        _samples(s2={"continues": "nope"})
    with pytest.raises(Exception, match="cycle"):
        _samples(s1={"continues": "s2"}, s2={"continues": "s1"})
    bad = _trip("s1", 3, "2026-09-01 10:00").assign(temp_c=np.nan)
    with pytest.raises(LabelError):
        build_labels(bad, _samples(), CFG)


def test_level_clock_ranges_adapt_to_temperature():
    warm, cool = level_clock_ranges(35.0, CFG), level_clock_ranges(20.0, CFG)
    assert list(warm["level"]) == CFG.labels.classes
    assert (warm["clock_h_to"].iloc[:-1] < cool["clock_h_to"].iloc[:-1]).all()


def test_real_samples_yaml_excludes_exp3():
    from foodlens.config import load_samples

    assert load_samples(CFG).sessions["exp3_D1A"].exclude
    with pytest.raises(ConfigError):
        load_config(CFG.root / "configs" / "missing.yaml")
