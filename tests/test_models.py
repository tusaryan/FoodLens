"""P6: model zoo, nested grouped tuning, feature sets, paired comparison, OOD, calibration,
importance and the selection rule (tiny synthetic data, no real dataset)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foodlens.config import load_config
from foodlens.errors import ConfigError, SplitError
from foodlens.evaluation.cv import oof_predict
from foodlens.evaluation.importance import (
    grouped_permutation_importance,
    partial_dependence,
    summarize_importance,
)
from foodlens.evaluation.paired import paired_wins, per_group_metric
from foodlens.evaluation.run import single_shot_features
from foodlens.evaluation.splits import logo_splits
from foodlens.inference.ood import MahalanobisOOD, abstention_summary
from foodlens.models.calibration import calibrated_factory, swap_lights
from foodlens.models.classical import TopKSelector, build_pipeline, model_grid, tuned_factory
from foodlens.models.feature_sets import feature_set
from foodlens.models.select import select_model

CFG = load_config().model_copy(update={"n_jobs": 1})
F1, F2 = "roi_median__RED", "roi_median__GREEN"  # real names: families are looked up by spec


def _data(n_groups: int = 5, n: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    h = np.tile(np.linspace(0, 14, n), n_groups)
    cls = np.array(CFG.labels.classes)[np.digitize(h, CFG.labels.equiv_h_thresholds)]
    return pd.DataFrame({F1: -h + rng.normal(0, 0.5, h.size), F2: rng.normal(0, 1, h.size),
                         "y": h, "c": cls,
                         "g": np.repeat([f"p{i}" for i in range(n_groups)], n)})  # fmt: skip


def test_topk_selector_keeps_informative_and_clips_k():
    df = _data()
    x = df[[F2, F1]].to_numpy()
    sel = TopKSelector(k=1, task="regression").fit(x, df["y"].to_numpy())
    assert sel.get_support().tolist() == [False, True]
    assert TopKSelector(k=50).fit(x, df["y"].to_numpy()).get_support().all()


def test_grid_guard_and_defaults():
    for name in CFG.models.zoo:
        for task in ("regression", "classification"):
            assert build_pipeline(name, task, CFG).steps[-1][0] == "model"
            assert model_grid(name, task, CFG)
    big = CFG.models.model_copy(update={"max_grid_combos": 2})
    with pytest.raises(ConfigError):
        model_grid("linear", "regression", CFG.model_copy(update={"models": big}))


class _Spy:
    """Records the groups it was fitted on (to prove the test fold is never passed)."""

    uses_groups = True
    seen: list = []

    def fit(self, x, y, groups=None):
        _Spy.seen.append(set(groups))
        self.m = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.m)


def test_oof_passes_only_training_groups():
    df = _data()
    g = df["g"].to_numpy()
    splits = logo_splits(g)
    _Spy.seen = []
    oof_predict(lambda t: _Spy(), "regression", df[[F1]].to_numpy(), df["y"].to_numpy(), splits,
                CFG.labels.classes, g)  # fmt: skip
    for (_, te), seen in zip(splits, _Spy.seen, strict=True):
        assert set(g[te]).isdisjoint(seen) and len(seen) == 4


def test_tuned_model_tunes_with_groups_and_falls_back():
    df = _data()
    x, y, g = df[[F1, F2]].to_numpy(), df["y"].to_numpy(), df["g"].to_numpy()
    m = tuned_factory("linear", CFG)("regression").fit(x, y, groups=g)
    assert m.tuned_ and set(m.best_params_) == set(model_grid("linear", "regression", CFG))
    two = np.isin(g, ["p0", "p1"])
    m2 = tuned_factory("linear", CFG)("regression").fit(x[two], y[two], groups=g[two])
    assert not m2.tuned_ and m2.best_params_ == {}
    assert np.abs(m.predict(x) - y).mean() < 1.5


def test_feature_sets_same_recipe_per_light():
    red, rg, allf = (feature_set(n, CFG) for n in ("RED", "RG", "RGB_all"))
    assert red and all("__RED" in c and "GREEN" not in c for c in red)
    assert any(c.endswith("_RG") for c in rg) and not any(c.endswith("_RB") for c in rg)
    assert set(red) < set(rg) < set(allf) and allf == single_shot_features(CFG)
    rel = feature_set("relative", CFG)
    assert rel and all(c.endswith("__rel") and not c.startswith("shd_") for c in rel)
    with pytest.raises(ConfigError):
        feature_set("RX", CFG)


def test_paired_wins_counts_groups():
    oof = pd.DataFrame({"y": [0.0] * 4, "yhat": [1.0, 1.0, 2.0, 0.5], "group": ["a", "b"] * 2,
                        "target": "t", "scheme": "s", "run_id": ["r1", "r1", "r2", "r2"],
                        "model": ["lin__A", "lin__A", "lin__B", "lin__B"]})  # fmt: skip
    pg = per_group_metric(oof, "regression")
    parts = pg["model"].str.split("__", n=1, expand=True)
    out = paired_wins(pg.assign(family=parts[0], feature_set=parts[1]), "regression", "A",
                      50, 0, 0.95).set_index("feature_set")  # fmt: skip
    assert out.loc["A", "wins"] == 1 and out.loc["B", "wins"] == 1
    assert out.loc["B", "mean_diff_vs_ref"] == pytest.approx(0.25)


def test_ood_flags_far_points_and_summary():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 4))
    det = MahalanobisOOD(99.0, mode="in_sample").fit(x)
    assert det.is_ood(x).mean() == pytest.approx(0.01, abs=0.01)
    assert det.is_ood(np.full((1, 4), 8.0))[0]
    # grouped threshold: groups with shifted means -> a new group from the same spread is kept
    g = np.repeat(np.arange(6), 50)
    xg = x + g[:, None] * 0.6
    grouped = MahalanobisOOD(99.0).fit(xg, g)
    insample = MahalanobisOOD(99.0, mode="in_sample").fit(xg, g)
    assert grouped.threshold_ > insample.threshold_
    with pytest.raises(SplitError):
        MahalanobisOOD(99.0).fit(x)
    s = abstention_summary(np.array(["a", "b", "a"]), np.array(["a", "a", "a"]),
                           np.array([False, True, False]))  # fmt: skip
    assert s["abstention_rate"] == pytest.approx(1 / 3) and s["accuracy_kept"] == 1.0


def test_calibrated_model_probabilities():
    df = _data(n_groups=6)
    x, y, g = df[[F1, F2]].to_numpy(), df["c"].to_numpy(), df["g"].to_numpy()
    m = calibrated_factory("linear", CFG)("classification").fit(x, y, groups=g)
    p = m.predict_proba(x)
    assert m.calibrated_ and np.allclose(p.sum(axis=1), 1.0)
    assert (m.predict(x) == y).mean() > 0.5


def test_importance_ranks_informative_feature_and_pdp():
    df = _data()
    g = df["g"].to_numpy()
    imp = grouped_permutation_importance(tuned_factory("linear", CFG), "regression", df,
                                         [F1, F2], "y", g, CFG)  # fmt: skip
    s = summarize_importance(imp).set_index("item")
    assert s.loc[F1, "mean"] > 1.0 > abs(s.loc[F2, "mean"])
    assert "light:RED" in s.index and s.loc[F1, "folds_positive"] == 5
    m = tuned_factory("linear", CFG)("regression").fit(df[[F1, F2]].to_numpy(), df["y"], groups=g)
    pdp = partial_dependence(m, df, [F1, F2], F1, 10)
    assert len(pdp) == 10 and pdp["mean_prediction"].is_monotonic_decreasing


def _sel_table(vals: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"model": k, "task": "classification", "scheme": "LOGO_piece_id",
                          "metric": "balanced_accuracy", "value": v, "ci_lo": lo, "ci_hi": v + .1}
                         for k, (v, lo) in vals.items()])  # fmt: skip


def test_selection_rule_ties_and_baseline():
    t = _sel_table({"baseline_red_median": (0.30, 0.2), "baseline_mean": (0.1, 0.0),
                    "linear__RGB_all": (0.45, 0.35), "rf__RGB_all": (0.46, 0.36),
                    "svm__RGB_all": (0.60, 0.25)})  # fmt: skip
    d = select_model(t, CFG)  # svm best but its CI does not beat 0.30; rf/linear tie -> linear
    assert d["model"] == "linear__RGB_all" and d["meets_rule"] and d["tied_with"] == ["rf__RGB_all"]
    t2 = _sel_table({"baseline_red_median": (0.5, 0.4), "rf__RGB_all": (0.45, 0.3)})
    assert not select_model(t2, CFG)["meets_rule"]


def test_swap_lights_exchanges_columns():
    df = pd.DataFrame({"a__RED": [1.0], "a__BLUE": [2.0], "a__ratio_RB": [0.5]})
    out = swap_lights(df, list(df.columns))
    assert out.iloc[0].tolist() == [2.0, 1.0, 0.5] and df.iloc[0].tolist() == [1.0, 2.0, 0.5]


def test_slope_pct_per_h():
    from foodlens.evaluation.light_check import _slope_pct_per_h

    t = np.linspace(0, 5, 20)
    assert _slope_pct_per_h(t, 50 * np.exp(-0.04 * t)) == pytest.approx(-4.0)
    assert np.isnan(_slope_pct_per_h(t[:2], np.ones(2)))
