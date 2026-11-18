"""Tests for foodlens.config."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from foodlens.config import DEFAULT_CONFIG_PATH, config_hash, load_config, load_samples
from foodlens.errors import E_CONFIG, ConfigError


def _write_variant(tmp_path: Path, section: str, key: str, value) -> Path:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    data[section][key] = value
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    p = cfg_dir / "bad.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_default_config_loads():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.roi.min_frac < cfg.roi.max_frac
    assert cfg.thermal.T_ref == 25.0
    assert cfg.path("data_dir").is_absolute()


def test_samples_load_with_null_piece_ids():
    cfg = load_config()
    samples = load_samples(cfg)
    assert len(samples.sessions) == 5
    assert all(s.piece_id is None for s in samples.sessions.values())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("roi", "min_frac", -0.1),
        ("roi", "min_frac", 0.9),  # >= max_frac
        ("labels", "class_definition", "C"),
        ("labels", "equiv_h_thresholds", [4.0, 2.0]),
        ("thermal", "T_min", 30.0),
        ("qc", "unknown_key", 1),  # typo / unknown keys are rejected
    ],
)
def test_invalid_values_raise_config_error(tmp_path, section, key, value):
    with pytest.raises(ConfigError) as exc:
        load_config(_write_variant(tmp_path, section, key, value))
    assert exc.value.code == E_CONFIG


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_unknown_path_key_raises():
    with pytest.raises(ConfigError):
        load_config().path("nope")


def test_hash_stable_and_section_sensitive(tmp_path):
    cfg = load_config()
    h = config_hash(cfg)
    assert h == config_hash(load_config())
    assert len(h) == 10
    assert config_hash(cfg, ["roi"]) != config_hash(cfg, ["glint"])
    changed = load_config(_write_variant(tmp_path, "roi", "erode_px", 7))
    assert config_hash(changed, ["roi"]) != config_hash(cfg, ["roi"])
    assert config_hash(changed, ["glint"]) == config_hash(cfg, ["glint"])
    assert config_hash(changed, ["roi"]) == config_hash(changed, ["roi"])


def test_hash_ignores_config_location(tmp_path):
    # Same values from a different folder -> same hash (root is excluded).
    moved = load_config(_write_variant(tmp_path, "roi", "erode_px", load_config().roi.erode_px))
    assert config_hash(moved) == config_hash(load_config())


def test_hash_stable_across_processes():
    code = "from foodlens.config import load_config, config_hash; print(config_hash(load_config()))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == config_hash(load_config())


def test_unknown_hash_section_raises():
    with pytest.raises(ConfigError):
        config_hash(load_config(), ["nope"])
