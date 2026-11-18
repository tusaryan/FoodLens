# P6: Classical models, light ablation, calibration (1–2 sessions)

**Goal:** find the best interpretable model under grouped CV and answer "which light (R, G, B or all) predicts best?".
**Skills to load:** `session-protocol`, `ml-evaluation`, `long-running-commands`.

## P6.T1: Model zoo (`models/classical.py`)
Each model is an sklearn `Pipeline` (impute median → drop near-constant → scale → [optional SelectKBest inside the fold] → estimator). Regression and classification variants:
- Ridge / LogisticRegression (L2, class_weight balanced)
- RandomForest (n_estimators 500, min_samples_leaf 3)
- HistGradientBoosting (small: max_depth 3, learning_rate 0.05, l2 1.0)
- SVR / SVC (RBF)
- Optional: LightGBM if installed (`uv add lightgbm`)
Small, fixed hyper-parameter grids (≤ 12 combos) tuned with nested inner grouped CV. **Never on the test fold.**

## P6.T2: Experiments (`foodlens train --experiment <name>`; configs in `configs/experiments/*.yaml`)
1. **Light comparison (the core question of the project, user S06):** `feature_set ∈ {RED, GREEN, BLUE, RG, RB, GB, RGB_all (+ cross-light ratios)}` × model → `reports/tables/light_ablation.csv` and a figure. Same folds, same targets, same feature recipe per light, so the only difference is the light. Report each set's grouped metric with its group-bootstrap CI and a **paired** comparison per held-out session (which light wins in how many of the 5 sessions). Start from the P3 evidence: `reports/tables/eda_light_comparison.csv`, `feature_signal_ranking.csv` (F8).
2. Feature groups: intensity only / + glint / + texture / + ratios → what each group adds.
3. Targets: `hours_since_start` (regression), `freshness_class` (classification), `storage_day` (if available).
4. Single-shot (absolute features) vs session-normalised (relative) features → shows why calibration (grey card) matters.
Each run writes a `RunReport`. A summary goes to `reports/tables/model_comparison.csv` sorted by the grouped metric, with CI columns.

## P6.T3: Calibration and abstention
- For the best classifier: `CalibratedClassifierCV` (sigmoid, inner grouped CV). Reliability diagram `F13`, ECE before/after.
- OOD detector (`inference/ood.py`): Mahalanobis distance on the scaled features fitted on the training fold; threshold = 99th percentile of the training distances. Report the abstention rate on the test folds and the accuracy on non-abstained samples (both numbers, always together).

## P6.T4: Interpretation
- Permutation importance on the held-out folds (grouped) → `F14`; partial dependence for the top 3 features.
- Link the top features to their physical meanings from the feature dictionary (simple language).

## P6.T5: Choose the model + record the decision
- Selection rule (decided **before** looking): the highest grouped balanced accuracy (class task) whose CI lower bound beats the best baseline; ties → the simpler model.
- `docs/08_MODEL_RESULTS.md`: a table quoting `model_comparison.csv`, light-ablation findings, an honest statement vs the 94 % target, failure analysis (which session fails and why, with images).
- Save the final pipeline trained on **all** groups to `models/foodlens_spoil_v1/model.joblib` + `model_card.md` (data, metrics with CIs, intended use, limits: same capture box/LEDs/camera settings only).
