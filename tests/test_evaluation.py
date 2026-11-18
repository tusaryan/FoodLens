"""P5.T2-T4: splitters never leak, metrics/statistics are correct, baselines run in a Pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foodlens.config import load_config
from foodlens.errors import SplitError
from foodlens.evaluation import metrics as M
from foodlens.evaluation.cv import evaluate
from foodlens.evaluation.splits import (
    assert_no_group_overlap,
    get_groups,
    logo_splits,
    nested_inner_splits,
    random_frame_splits_LEAKY,
    temporal_holdout_splits,
)
from foodlens.evaluation.stats import (
    RunReport,
    bootstrap_group_ci,
    group_permutation_test,
    permute_within_groups,
)
from foodlens.models.baselines import baseline_specs

CFG = load_config()


def _df(n_groups: int = 5, n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        ts = pd.date_range("2026-09-01", periods=n, freq="3min") + pd.Timedelta(days=g)
        h = np.arange(n) * 0.05
        rows.append(pd.DataFrame({"session_id": f"s{g}", "piece_id": f"p{g // 2}", "ts": ts,
                                  "experiment": f"E{g % 2}", "y": h,
                                  "x": -h + rng.normal(0, 0.01, n)}))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def test_get_groups_prefers_piece_then_session():
    df = _df()
    assert get_groups(df, CFG)[0] == "p0"
    df.loc[0, "piece_id"] = None
    assert get_groups(df, CFG)[0] == "s0"
    with pytest.raises(SplitError):
        get_groups(df.drop(columns="session_id"), CFG)


def test_logo_and_inner_never_share_groups():
    df = _df()
    groups = get_groups(df, CFG, ["session_id"])
    splits = logo_splits(groups)
    assert len(splits) == 5
    assert_no_group_overlap(splits, groups)
    for tr, _ in splits:
        inner = nested_inner_splits(groups[tr], CFG)
        assert inner is not None
        assert_no_group_overlap(inner, groups[tr])
    assert nested_inner_splits(np.array(["a", "a", "b"]), CFG) is None  # < 3 groups -> fixed


def test_overlap_detector_and_single_group():
    with pytest.raises(SplitError):
        assert_no_group_overlap([(np.array([0]), np.array([1]))], np.array(["a", "a"]))
    with pytest.raises(SplitError):
        logo_splits(np.array(["a", "a"]))


def test_temporal_holdout_no_time_overlap():
    df = _df()
    ((tr, te),) = temporal_holdout_splits(df, CFG)
    for sid in df["session_id"].unique():
        m = (df["session_id"] == sid).to_numpy()
        assert df["ts"].to_numpy()[tr][m[tr]].max() < df["ts"].to_numpy()[te][m[te]].min()
    assert len(tr) / len(df) == pytest.approx(CFG.cv.temporal_train_frac, abs=0.05)


def test_leaky_splitter_is_named_and_mixes_groups():
    assert "LEAKY" in random_frame_splits_LEAKY.__name__
    df = _df()
    groups = get_groups(df, CFG, ["session_id"])
    with pytest.raises(SplitError):
        assert_no_group_overlap(random_frame_splits_LEAKY(len(df), CFG), groups)


def test_regression_and_classification_metrics():
    y, yhat = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 5.0])
    r = M.regression_metrics(y, yhat, np.array(["a", "a", "b", "b"]))
    assert r["mae"] == pytest.approx(0.5) and r["rmse"] == pytest.approx(1.0)
    assert r["per_group_mae"] == {"a": 0.0, "b": 1.0} and r["spearman"] == pytest.approx(1.0)
    classes = ["a", "b"]
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    yc = np.array(["a", "b", "b"])
    c = M.classification_metrics(yc, np.array(["a", "b", "a"]), proba, classes, 10)
    assert c["balanced_accuracy"] == pytest.approx(0.75)
    assert c["confusion_matrix"] == [[1, 0], [1, 1]]
    assert c["brier"] == pytest.approx(0.5 / 3)
    assert c["ece"] == pytest.approx(0.5 / 3)  # bin 0.5: acc 0 vs conf 0.5, weight 1/3


def test_bootstrap_ci_and_permutation():
    y = np.repeat([1.0, 2.0, 3.0], 4)
    groups = np.repeat(["a", "b", "c"], 4)
    lo, hi = bootstrap_group_ci(y, y + 1.0, groups, M.mae, 200, 0)
    assert lo == hi == pytest.approx(1.0)
    perm = permute_within_groups(y, groups, np.random.default_rng(0))
    assert all(sorted(perm[groups == g]) == sorted(y[groups == g]) for g in "abc")
    res = group_permutation_test(lambda v: float(v[0]), y, groups, 50, 0, greater_is_better=True)
    assert 0 < res["p_value"] <= 1 and res["n_permutations"] == 50


def test_baselines_beat_mean_on_learnable_signal(tmp_path):
    df = _df(n=40)
    df["freshness_class"] = np.where(df["y"] < 1.0, "fresh", "use_soon")
    groups = get_groups(df, CFG, ["session_id"])
    specs = baseline_specs(CFG)
    out = {}
    for name in ("baseline_mean", "baseline_red_median"):
        spec = specs[name]
        d = df.rename(columns={"x": spec.features[0]})
        res = evaluate(spec.make, "regression", d, spec.features, "y", groups,
                       logo_splits(groups), CFG, permute=False)  # fmt: skip
        out[name] = res["metrics"]["mae"]
    assert out["baseline_red_median"] < 0.2 * out["baseline_mean"]
    spec = specs["baseline_red_median"]
    d = df.rename(columns={"x": spec.features[0]})
    res = evaluate(spec.make, "classification", d, spec.features, "freshness_class", groups,
                   logo_splits(groups), CFG, permute=False)  # fmt: skip
    assert res["metrics"]["balanced_accuracy"] > 0.9
    rep = RunReport("r1", "m", "y", "regression", "LOGO", "session_id", 5, len(df), "B", ["x"],
                    "h", "c", res["metrics"])  # fmt: skip
    assert rep.save(tmp_path).is_file()
