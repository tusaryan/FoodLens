# P5: Labels and the evaluation framework (1 session) — science-critical, use a strong model

**Goal:** define what we predict and build an evaluation harness that cannot leak, before training any serious model.
**Skills to load:** `session-protocol`, `ml-evaluation`.

## P5.T1: Labels (`labels/labels.py`, `labels/thermal_age.py`)
- `hours_since_start` = elapsed_s / 3600 (per session).
- `equiv_hours_25C` = Σ ((T−T_min)/(T_ref−T_min))² Δt over the frame times, T from `temp_c` + `indoor_offset_c`; `T ≤ T_min` contributes 0. Unit-tested against a hand calculation (constant 25 °C → equals hours; 33 °C for 1 h → (41/33)² ≈ 1.544).
- Verify the `T_min` default against a poultry-spoilage source (Ratkowsky-model papers listed in docs/01 refs 4–5). Record the source in config comments.
- If `pre_age_h` (hours at room temperature before filming) is known in samples.yaml, add it → `age_h_total`. Otherwise `age_h_total = NaN` and models use the relative labels.
- `storage_day` from samples.yaml (nullable).
- `freshness_class` (3 classes; **must be confirmed by the user/mentor** — write it to Open questions):
  - Default **A** (if `storage_day` is confirmed): D1 → `fresh`, D2 → `use_soon`, D3 → `spoiled_risk`.
  - Default **B** (otherwise): by `equiv_hours_25C` relative to the start: `< 2 h` fresh, `2–4 h` use_soon, `≥ 4 h` spoiled_risk (thresholds in config, chosen before seeing the model results; justify from the TVC growth literature, approximate).
  - Store which definition was used in every metrics JSON.

## P5.T2: Splitters (`evaluation/splits.py`)
- `get_groups(df, cfg)` → `piece_id` if fully non-null, else `session_id`.
- `logo_splits`, `nested_inner_splits`, `temporal_holdout_splits` (first 60 % → last 40 % within each session), `random_frame_splits` (leakage demo only; its name must contain `LEAKY`).
- **Tests:** no group appears in both train and test (for every splitter except LEAKY); the temporal holdout has no time overlap.

## P5.T3: Metrics and statistics (`evaluation/metrics.py`, `evaluation/stats.py`)
- The full metric list from the skill; `bootstrap_group_ci(y, yhat, groups, metric, n=2000)`; `group_permutation_test(...)`; `expected_calibration_error`.
- `RunReport` dataclass → JSON (`reports/metrics/<run_id>.json`) with metrics, CIs, config hash, git commit (`git rev-parse HEAD`), label definition, feature set, n groups, n samples.

## P5.T4: Baselines + leakage demo (`models/baselines.py`, `evaluation/leakage_demo.py`)
- Baselines: mean/majority; `roi_median__RED`-only linear model; `glint_count`-only (the old method); the mentor's "first-frame-normalised intensity" as a single feature.
- Leakage demo: RandomForest on all features under LOGO vs LEAKY random split → `reports/tables/leakage_demo.csv` + figure `F10b`. Plain-language caption: "same model, same data. Only the split changed."

## P5.T5: `foodlens evaluate --model baseline_*` end-to-end, check.sh PASS, results in PROGRESS key numbers.
