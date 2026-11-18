"""Tests for the P4 feature code: frame features, schema, triplets, feature QC (synthetic only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foodlens.config import load_config
from foodlens.features.build import (
    blur_to_nan,
    drop_edge_glints,
    edge_inner,
    extract_frame_features,
    step_correct_frames,
)
from foodlens.features.qc import feature_qc, high_corr_pairs
from foodlens.features.run import frame_table, triplet_table
from foodlens.features.schema import (
    FRAME_META,
    GROUPS,
    export_feature_dictionary,
    frame_specs,
    group_features,
    triplet_meta,
    triplet_specs,
)
from foodlens.features.triplets import add_cross_light, add_relative, wide_triplets
from foodlens.vision.glints import detect_glints_relative
from foodlens.vision.texture import lbp_features

CFG = load_config()
LIGHTS = CFG.capture.light_order
T0 = pd.Timestamp("2026-09-01 10:00", tz="Asia/Kolkata")


def _masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lit ROI = inner ellipse, shadow = ring around it, reference = left border strip)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    e = ((yy - h / 2) / (h * 0.3)) ** 2 + ((xx - w / 2) / (w * 0.3)) ** 2
    ref = np.zeros((h, w), bool)
    ref[:, :10] = True
    return e <= 0.6, (e > 0.6) & (e <= 0.95), ref


def test_lbp_histogram_sums_to_one_and_ignores_brightness():
    rng = np.random.default_rng(0)
    ch = rng.integers(20, 60, size=(80, 80)).astype(np.uint8)
    mask = np.zeros_like(ch, bool)
    mask[10:70, 10:70] = True
    f = lbp_features(ch, mask, CFG.vision)
    bins = [f[f"lbp_{i}"] for i in range(CFG.vision.lbp_p + 2)]
    assert pytest.approx(sum(bins)) == 1.0 and f["lbp_entropy"] > 0
    g = lbp_features((ch.astype(int) * 3).astype(np.uint8), mask, CFG.vision)  # monotonic change
    assert all(pytest.approx(f[k]) == g[k] for k in f)
    with pytest.raises(ValueError):
        lbp_features(ch, np.zeros_like(mask), CFG.vision)


def test_extract_frame_features_matches_schema(make_meat_image):
    img = make_meat_image(light="GREEN", glints=12, seed=2)
    lit, shadow, ref = _masks(img)
    f = extract_frame_features(img, "GREEN", lit, CFG, shadow, ref)
    assert list(f) == [s.name for s in frame_specs(CFG)] + ["ref_mean"]
    keys = ["roi_mean", "shd_mean", "roi_mean_rel_ref", "glcm_energy", "lbp_entropy"]
    assert np.isfinite([f[k] for k in keys]).all()
    assert f["roi_mean"] > 100 and f["glint_density"] > 0 and f["glintabs_density"] > 0
    bare = extract_frame_features(img, "GREEN", lit, CFG)
    assert np.isnan(bare["shd_mean"]) and np.isnan(bare["roi_mean_rel_ref"])
    assert bare["full_mean"] == pytest.approx(bare["roi_mean"], rel=0.01)


def test_ratio_features_do_not_depend_on_gain(make_meat_image):
    img = make_meat_image(light="RED", glints=0, seed=4)
    lit, shadow, ref = _masks(img)
    a = extract_frame_features(img, "RED", lit, CFG, shadow, ref)
    b = extract_frame_features((img * 0.5).astype(np.uint8), "RED", lit, CFG, shadow, ref)
    assert b["roi_mean"] == pytest.approx(a["roi_mean"] / 2, rel=0.02)
    assert b["roi_cv"] == pytest.approx(a["roi_cv"], rel=0.1)


def test_edge_glints_are_dropped():
    ch = np.full((120, 120), 40, np.uint8)
    roi = np.zeros_like(ch, bool)
    roi[20:100, 20:100] = True
    for y, x in [(60, 60), (22, 60)]:  # interior glint, glint on the ROI edge
        ch[y - 1 : y + 2, x - 1 : x + 2] = 200
    res = detect_glints_relative(ch, roi, CFG.glint)
    assert len(res.props_df) == 2
    kept = drop_edge_glints(res, edge_inner(roi, 8))
    assert kept.props_df[["centroid_y", "centroid_x"]].round().values.tolist() == [[60.0, 60.0]]
    assert edge_inner(roi, 0) is roi


def _frame_rows(n: int, lights: list[str] | None = None) -> pd.DataFrame:
    lights = lights or [LIGHTS[i % 3] for i in range(n)]
    return pd.DataFrame(
        {
            "session_id": "s1",
            "light": lights,
            "ts": [T0 + pd.Timedelta(seconds=60 * i) for i in range(len(lights))],
            "scene_segment": 0,
            "qc_reasons": "",
            "roi_mean": 10.0,
        }
    )


def test_blur_to_nan_only_glint_and_texture():
    df = _frame_rows(3).assign(glint_density=1.0, glcm_energy=0.2, lbp_0=0.1)
    df.loc[1, "qc_reasons"] = "W_BLUR"
    out = blur_to_nan(df.assign(**{c: 1.0 for c in group_features(CFG, ["glint", "texture"])}), CFG)
    assert out["blur_nan"].tolist() == [False, True, False]
    assert out.loc[1, ["glint_density", "glcm_energy", "lbp_0"]].isna().all()
    assert out.loc[1, "roi_mean"] == 10.0 and out.loc[0, "glint_density"] == 1.0


def test_step_correct_frames_removes_gain_and_records_it():
    df = _frame_rows(24).assign(roi_cv=0.3)
    for c in group_features(CFG, ["intensity"]):
        df[c] = 10.0
    later = df.index >= 12
    df.loc[later, "scene_segment"] = 1
    df.loc[later, "roi_mean"] = 20.0
    out = step_correct_frames(df, CFG)
    assert out["roi_mean"].tolist() == pytest.approx([10.0] * 24)
    assert out.loc[later, "step_gain"].tolist() == pytest.approx([2.0] * 12)
    assert out["roi_cv"].eq(0.3).all()  # ratios are not corrected


def _trip_frames(lights: list[str], gaps_s: list[float]) -> pd.DataFrame:
    ts = T0 + pd.to_timedelta(np.concatenate([[0], np.cumsum(gaps_s)]), unit="s")
    return pd.DataFrame(
        {
            "frame_id": [f"f{i}" for i in range(len(lights))],
            "session_id": "s1", "experiment": "EXP-1", "piece_id": None, "light": lights,
            "ts": ts, "elapsed_s": (ts - T0).total_seconds(), "temp_c": 30.0,
            "scene_segment": 0, "blur_nan": False, "v": np.arange(1.0, len(lights) + 1),
        }
    )  # fmt: skip


def test_partial_triplet_is_flagged():
    d = _trip_frames(["RED", "GREEN", "RED", "GREEN", "BLUE"], [60] * 4)
    t = wide_triplets(d, ["v"], LIGHTS, CFG.triplet.max_span_s)
    assert len(t) == 2
    assert t["is_complete"].tolist() == [False, True]
    assert t["has_BLUE"].tolist() == [False, True] and np.isnan(t.loc[0, "v__BLUE"])
    assert t["n_lights"].tolist() == [2, 3]


def test_span_above_max_splits_and_missing_light_still_has_columns():
    d = _trip_frames(["RED", "GREEN", "RED", "GREEN"], [60, 60, 400])
    t = wide_triplets(d, ["v"], LIGHTS, CFG.triplet.max_span_s)
    assert len(t) == 3  # R G | R | G (span 400 s > 200 s)
    assert {"v__BLUE", "frame_id__BLUE", "has_BLUE"} <= set(t.columns)
    assert not t["has_BLUE"].any()


def test_cross_light_and_relative():
    d = _trip_frames(LIGHTS * 3, [60] * 8)
    t = wide_triplets(d, ["v"], LIGHTS, CFG.triplet.max_span_s)
    t = add_cross_light(t, ["v"], ["v"])
    assert t.loc[0, "v__ratio_RG"] == pytest.approx(1 / 2)
    assert t.loc[0, "v__logratio_RB"] == pytest.approx(np.log(1 / 3))
    t = add_relative(t, ["v__RED", "v__ratio_RG"])
    assert t["v__RED__rel"].tolist() == pytest.approx([1.0, 4.0, 7.0])
    assert t.loc[0, "v__ratio_RG__rel"] == pytest.approx(1.0)


def test_tables_have_exactly_the_schema_columns(make_meat_image):
    """Every column built by frame_table / triplet_table is in the schema and vice versa."""
    rows = []
    for i in range(6):
        light = LIGHTS[i % 3]
        img = make_meat_image(light=light, glints=8, seed=i)
        lit, shadow, ref = _masks(img)
        feats = extract_frame_features(img, light, lit, CFG, shadow, ref)
        rows.append({"frame_id": f"f{i}", "mask_reason": "", "roi_gap_s": 0.0} | feats)
    frames = _trip_frames(LIGHTS * 2, [60] * 5).drop(columns=["v", "blur_nan"])
    frames = frames.assign(qc_status="ok", qc_reasons="")
    frame = frame_table(frames, pd.DataFrame(rows), CFG)
    assert list(frame.columns) == FRAME_META + [s.name for s in frame_specs(CFG)]
    trip = triplet_table(frame, CFG)
    assert list(trip.columns) == triplet_meta(CFG) + [s.name for s in triplet_specs(CFG)]
    assert len(trip) == 2 and trip["is_complete"].all()


def test_feature_qc_flags_constant_nan_and_redundant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    trip = pd.DataFrame({"a__RED": x, "b__RED": 2 * x + 1e-6, "c__RED": 1.0, "d__RED": np.nan})
    trip["e__RED"] = rng.normal(size=50)
    frame = pd.DataFrame({"light": "RED", "a": x})
    qc, pairs = feature_qc(frame, ["a"], trip, list(trip.columns), 0.98)
    t = qc[qc["level"] == "triplet"].set_index("column")
    assert bool(t.loc["c__RED", "constant"]) and t.loc["d__RED", "pct_nan"] == 100
    assert t.loc["a__RED", "max_abs_r_partner"] == "b__RED" and t.loc["e__RED", "n_high_corr"] == 0
    assert pairs[["a", "b"]].values.tolist() == [["a__RED", "b__RED"]]
    assert high_corr_pairs(trip, ["e__RED"], 0.98).empty


def test_feature_dictionary(tmp_path):
    d = export_feature_dictionary(CFG, tmp_path / "fd.csv")
    assert (tmp_path / "fd.csv").is_file() and d["name"].is_unique
    assert set(d["group"]) <= set(GROUPS)
    assert not d.loc[d["group"] == "relative", "single_shot_ok"].any()
    assert d.loc[d["group"] != "relative", "single_shot_ok"].all()
    banned = ("ts", "session", "elapsed", "centroid", "bbox", "width", "height", "frame_id")
    assert not any(b in n for n in d["name"] for b in banned)


def test_neck_cut_removes_thin_tail_and_triplet_masks(make_meat_image):
    from foodlens.vision.triplet_roi import frame_reference, neck_cut, triplet_masks

    m = np.zeros((200, 300), bool)
    m[50:150, 150:260] = True  # meat block
    m[95:105, 20:150] = True  # 10 px wide tray strip joined to it
    cut = neck_cut(m, 8)
    assert not cut[:, :120].any() and cut[60:140, 160:250].all()
    assert neck_cut(m, 0) is m
    red = make_meat_image(light="RED", glints=5, seed=1)
    blue = make_meat_image(light="BLUE", glints=5, seed=1, level=45)[..., 2]
    tm = triplet_masks(red, blue, CFG)
    assert tm.ok and tm.lit.any() and not (tm.lit & tm.shadow).any()
    assert ((tm.lit | tm.shadow) == tm.full).all()
    assert not triplet_masks(np.zeros_like(red), None, CFG).ok
    ch = red[..., 0].copy()
    ch[:, :20] = 60  # lit tray strip far from the meat
    ref = frame_reference(ch, tm.full, CFG)
    assert ref is not None and ref[:, :15].all() and not (ref & tm.full).any()
