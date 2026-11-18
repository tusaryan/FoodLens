---
name: ml-evaluation
description: Leakage-free evaluation rules for FoodLens - grouped CV, baselines, bootstrap CIs, permutation tests, calibration, and the random-vs-grouped leakage demo. Load for any task that trains or scores a model.
---
# Evaluation rules

## Units
- Observation = one **triplet** (R, G, B frames within one 3-min cycle) unless the task says frame-level.
- Group = `session_id` now; switch to `piece_id` when `configs/samples.yaml` has it confirmed. Splitters must read the group column name from config.

## Splits (`foodlens.evaluation.splits`)
- Primary: `LeaveOneGroupOut` over sessions (5 folds now).
- Hyper-parameters: nested, inner `GroupKFold` on the training groups only. If the training fold has fewer than 3 groups, use a fixed default config (document this).
- Temporal sanity check: within a session, also report "train on first 60 % of time → test on last 40 %" as a secondary extrapolation test (never as the headline number).
- Leakage demo (report only): the same model with a random `KFold(shuffle=True)` over frames. Show both numbers side by side.

## Everything fitted goes in a Pipeline
`Pipeline([("scale", StandardScaler()), ("select", ...), ("model", ...)])`. The OOD detector's mean/cov is fitted on the training fold only.

## Metrics
- Regression (hours / equivalent hours): MAE, RMSE, Spearman ρ, R²; per-group MAE.
- Classification (freshness class): balanced accuracy, macro-F1, per-class precision/recall, confusion matrix, Brier score, ECE (10 bins).
- Always include baselines: (a) predict training mean / majority class, (b) single-feature model (ROI median of red), (c) the "spot count" feature alone (the old method).

## Uncertainty and significance
- 95 % CI by bootstrapping **groups** (resample sessions with replacement, 2000 reps, seed from config). With 5 groups, CIs will be wide. Say so.
- Permutation test: shuffle labels **within the grouped structure** (permute group-level labels or time within a group, as appropriate), 1000 reps, report p-value.
- Mixed-effects or per-session trend tests for EDA claims (`statsmodels.mixedlm` with session random intercept).

## Red flags (stop and investigate)
- Score > 0.98, or the grouped score is close to the random-split score with only 5 groups.
- A feature that encodes the session: absolute timestamp, file index, image orientation, ROI position. These are banned as features.
- Different preprocessing for train and test.

## Output
`reports/metrics/<run_id>.json` (all metrics + CIs + config hash + git commit) and `reports/tables/model_comparison.csv`. Docs quote these files.
