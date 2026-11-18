"""Named feature sets for the P6 experiments (light ablation, feature groups, relative).

Light sets use the SAME recipe for every light: all single-shot features measured under the
lights of the set plus the cross-light ratios among exactly those lights. So RED vs GREEN vs
RG differ only in which LED illuminated the meat (user question S06).
- ``RED`` / ``GREEN`` / ``BLUE``: one light (no cross-light ratio is possible).
- ``RG`` / ``RB`` / ``GB``: two lights + their ratio / log-ratio columns.
- ``RGB_all``: all single-shot features (three lights + all cross-light ratios).
Group sets (cumulative, all lights): ``intensity`` -> ``+glint`` -> ``+texture`` -> ``+ratio``.
Session-normalised: ``relative`` (``__rel`` columns only; need the session's first photo) and
``single_shot+relative``.
"""

from __future__ import annotations

from foodlens.config import Config
from foodlens.errors import ConfigError
from foodlens.evaluation.run import single_shot_features
from foodlens.features.schema import triplet_specs

LIGHT_CODES = {"R": "RED", "G": "GREEN", "B": "BLUE"}
LIGHT_SETS = ["RED", "GREEN", "BLUE", "RG", "RB", "GB", "RGB_all"]
GROUP_SETS = ["intensity", "intensity+glint", "intensity+glint+texture", "RGB_all"]
RELATIVE_SETS = ["RGB_all", "relative", "single_shot+relative"]


def _lights_of(code: str) -> set[str]:
    if code in LIGHT_CODES.values():
        return {code}
    if code == "RGB_all":
        return set(LIGHT_CODES.values())
    if not code or any(c not in LIGHT_CODES for c in code):
        raise ConfigError(f"unknown light set {code!r}")
    return {LIGHT_CODES[c] for c in code}


def _spec_lights(light: str) -> set[str]:
    """Lights a feature needs (``RED/GREEN`` -> both)."""
    return set(light.split("/"))


def feature_set(name: str, cfg: Config) -> list[str]:
    """Column list of feature set ``name`` (see module docstring), in table order."""
    ss = set(single_shot_features(cfg))
    specs = [s for s in triplet_specs(cfg) if s.name in ss]
    if name in GROUP_SETS[:-1]:
        groups = set(name.split("+"))
        return [s.name for s in specs if s.group in groups]
    if name == "relative":
        drop = tuple(cfg.evaluation.drop_feature_prefixes)
        return [s.name for s in triplet_specs(cfg)
                if s.group == "relative" and not s.name.startswith(drop)]  # fmt: skip
    if name == "single_shot+relative":
        return feature_set("RGB_all", cfg) + feature_set("relative", cfg)
    lights = _lights_of(name)
    return [s.name for s in specs if _spec_lights(s.light) <= lights]


def feature_sets(names: list[str], cfg: Config) -> dict[str, list[str]]:
    """``{name: columns}``; raises ``ConfigError`` for an empty set."""
    out = {n: feature_set(n, cfg) for n in names}
    empty = [n for n, c in out.items() if not c]
    if empty:
        raise ConfigError(f"empty feature set(s): {empty}")
    return out
