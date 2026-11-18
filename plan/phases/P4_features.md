# P4: Feature table (1 session)

**Goal:** one tidy, documented feature table at frame and triplet level, used by every model, the notebook and the API (the same code path for training and inference).
**Skills to load:** `session-protocol`, `image-pipeline`, `long-running-commands`.

## P4.T1: Feature schema (`features/schema.py`)
- A single list of `FeatureSpec(name, light, group, unit, description, physical_meaning, single_shot_ok: bool)`.
- Groups: `intensity`, `glint`, `texture`, `ratio`, `relative` (relative = `single_shot_ok=False`).
- `export_feature_dictionary()` → `reports/tables/feature_dictionary.csv` (report table T5).
- A test: every column produced by `build.py` exists in the schema and vice versa.

## P4.T2: Frame features (`features/build.py`)
- `extract_frame_features(img_rgb, light, roi_mask, cfg) -> dict`: reuse the `eda/frame_stats.py` + `vision/texture.py` functions (GLCM/LBP per the skill).
- **This exact function is what the Predictor calls at inference.** No training-only shortcuts.
- Parallel over frames, cached by config hash. Output `data/processed/features_frame.parquet` with id columns (`frame_id, session_id, piece_id, ts, light, elapsed_s, temp_c`) + features.

## P4.T3: Triplets (`features/triplets.py`)
- The assembly rule from the skill. Wide format: `<feature>__RED`, `__GREEN`, `__BLUE`, plus the cross-light ratios and `has_RED/GREEN/BLUE`.
- `relative` features: divide by the session's first complete triplet (for EDA/experiments only).
- Output `data/processed/features_triplet.parquet`. Expected ≈ 300 triplets (944 frames / 3, minus the D1A blue gaps, which give partial triplets).
- **Tests:** synthetic timestamps with a missing BLUE → partial triplet flagged; a span > max → split correctly.

## P4.T4: Sanity report
- `reports/tables/feature_qc.csv`: % NaN per column, constant columns, |corr| > 0.98 pairs (to drop later inside the Pipeline, **not** globally).
- **Accept:** no column 100 % NaN; row counts are logged; check.sh PASS.
