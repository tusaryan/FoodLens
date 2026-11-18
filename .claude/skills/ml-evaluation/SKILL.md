---
name: ml-evaluation
description: Leakage-free evaluation rules for FoodLens - grouped CV, baselines, bootstrap CIs, permutation tests, calibration, and the random-vs-grouped leakage demo. Load for any task that trains or scores a model.
---
# Evaluation rules

## Units
- Observation = one **triplet** (R, G, B frames within one 3-min cycle) unless the task says frame-level.
- Group = `piece_id` (S08: every session is its own piece unless `continues:` in samples.yaml). Splitters read the group column name from config.

## Splits (`foodlens.evaluation.splits`)
- Primary: `LeaveOneGroupOut` over pieces (9 folds now); sensitivity: leave-one-experiment-out.
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
- Score > 0.98, or the grouped score is close to the random-split score with only 9 groups.
- A CNN whose Grad-CAM mass sits on the mask edge / outline, or whose score collapses on the EXP-2-only check: suspect a setup shortcut.
- A feature that encodes the session: absolute timestamp, file index, image orientation, ROI position. These are banned as features.
- Different preprocessing for train and test.

## CNN / deep-learning rules (P7A–P7D)
- **Same outer folds** as P6 (`evaluation.splits`); CNN results go through the same `score`, bootstrap, `paired.py` and `model_comparison.csv`, so CNN vs classical is a paired comparison per held-out piece.
- **No test-piece pixel ever in training:** datasets are built from the row ids the CV loop passes in; a spy test proves the held-out group never reaches `fit`. This includes composites, patches, caches, normalisation and the reference photo of the reference-pair ablation (a test piece may only be paired with its OWN first photo, never a training piece's).
- **Early stopping / epoch choice / learning-rate choice / temperature scaling only on INNER grouped folds** of the training pieces. Never look at the test piece's loss curve. Retrain on all training pieces for the chosen number of epochs.
- **Augmentation is fitted on nothing:** fixed geometric rules, mild blur, an equal gain on all channels. No hue/colour jitter (colour is the signal), no per-image standardisation, no mixup across pieces. Normalisation = the backbone's fixed ImageNet constants. Test time: no augmentation, a fixed seeded patch grid.
- **Shortcut checks:** crops are scale-normalised by meat area and training uses interior patches (no silhouette, no image size). Report the within-piece rank correlation (framing is constant inside a piece), an EXP-2-only LOGO, a "scale factor alone" baseline, and Grad-CAM saliency on the mask edge band.
- **Seeds and determinism:** `dl.device.seed_everything` (python, numpy, torch, MPS) + `torch.use_deterministic_algorithms(True, warn_only=True)`; MPS is still not bit-exact, so run seeds `[0, 1, 2]` and report the mean and spread; record torch/timm versions, device and the weight file hash in the RunReport.
- **Class / piece balance** (class weights, 1/n-per-piece sample weights) is computed from the training pieces only.
- **Permutation test for a fine-tuned CNN** needs retraining per shuffle: state the number of reps next to p (≥ 19 → p ≥ 0.05 resolvable).
- **Pre-register** the main architecture, hyper-parameters and the selection rule in `configs/default.yaml` (`dl:`) before the first CNN result. Anything tried afterwards is labelled exploratory (winner's curse).
- A CNN random-split score is only shown as the leakage demo (expect ≈ 0.98).

## Output
`reports/metrics/<run_id>.json` (all metrics + CIs + config hash + git commit) and `reports/tables/model_comparison.csv`. Docs quote these files.
