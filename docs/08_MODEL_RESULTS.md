# 08 · Model results (Phase P6, session S09)

**For:** the mentor and the project team. **Data:** 1258 complete R/G/B triplets from 9 chicken
pieces (exp3_D1A excluded, open question 9). **Labels:** 4 freshness levels by temperature-
equivalent hours at 25 °C (`fresh` < 2 h, `use_soon` 2–6 h, `quality_loss` 6–12 h, `spoiled_risk`
≥ 12 h; chance balanced accuracy = 0.25). Every number below comes from the file named next to
it (written by `foodlens evaluate`, `foodlens train --experiment <name>`, `foodlens finalize`).

## How the models were tested (one paragraph)
Every score is **leave-one-piece-out**: the model never sees any photo of the piece it is tested
on (9 folds; leave-one-experiment-out, 4 folds, as a check). Everything that learns from data
(imputation, scaling, feature selection, the hyper-parameter grid of ≤ 12 settings, calibration,
the OOD detector) is fitted **inside the training pieces only**. Hyper-parameters are tuned with an
inner GroupKFold over the training pieces (nested CV). 95 % CIs come from resampling **pieces**,
not photos. The model zoo is: L2 logistic / ridge regression ("linear"), random forest (500 trees),
small gradient boosting (HGB, depth 3) and an RBF support-vector machine. The final-model
selection rule was written into `configs/default.yaml` (`train.selection`) **before** any P6 result
was seen.

## 1. Headline (pre-registered rule) — `reports/tables/model_comparison.csv`, `model_selection.json`
| Single-shot model (all 153 features) | Balanced acc. [95 % CI] | MAE equiv. h [95 % CI] |
|---|---|---|
| linear (selected) | 0.313 [0.227, 0.503] | 4.16 [2.18, 5.83] |
| random forest | 0.329 [0.192, 0.519] | 4.32 [1.55, 6.73] |
| gradient boosting | 0.354 [0.207, 0.597] | 4.14 [1.51, 6.44] |
| SVM | 0.271 [0.230, 0.396] | 3.94 [1.68, 5.14] |
| baseline: RED median only | 0.207 [0.096, 0.391] | 5.66 [3.93, 7.11] |
| baseline: training mean / majority | 0.077 [0.000, 0.154] | 6.61 [5.58, 7.65] |
| (not single-shot) mentor first-frame baseline | 0.489 [0.406, 0.646] | 5.00 [1.67, 6.50] |

- The rule picks **linear**: its CI lower bound (0.227) beats the best single-shot baseline
  (0.207). HGB has the highest value but its lower bound (0.2071) misses the baseline (0.2074).
- **Honest reading:** the selected classifier's grouped permutation test gives **p = 0.23**
  (100 within-piece label shuffles, `model_selection.json`). So on an unseen piece the 4-level
  classifier is **not** reliably better than chance yet. The age estimate is better: MAE 4.2 h
  versus 5.7 h (RED baseline) and 6.6 h (mean), but the CIs overlap.
- **Versus the 94 % target:** far from it. With 9 pieces, of which only 2 reach `spoiled_risk`,
  a single photo of a NEW piece cannot yet be classified reliably. Within-piece time
  extrapolation (temporal 60/40 split, same pieces in train and test) reaches 0.60–0.70, which
  shows the signal exists but that **between-piece differences** dominate.

## 2. Which light predicts best? (core question) — `light_ablation.csv`, `light_ablation_paired.csv`, F12
Same pieces, folds, targets and feature recipe; only the LED differs.
- **BLUE is the most informative single light, RED the least.** Linear on BLUE: balanced acc.
  0.460 [0.376, 0.630], MAE 3.24 h [2.32, 4.02]; SVM on BLUE: 0.522 [0.450, 0.626]. RED alone:
  0.233 (linear), MAE 7.97 h.
- Paired per held-out piece (linear): BLUE beats all-lights on 7 of 9 pieces for the level
  (mean +0.085 balanced acc., CI of the difference [0.014, 0.157]); RED is worse than all-lights
  by +2.3 h MAE (CI [0.5, 4.5]).
- **Is the BLUE signal the meat?** Yes, as far as we can check: under BLUE the meat darkens
  fastest (e.g. −11.3 %/h exp1_D1B, −8.6 %/h exp2_D2B, −4.6 %/h exp5) while the lit tray /
  holder reference stays flat (−0.8 to +1.6 %/h except exp2_D2A, the known camera step)
  (`light_meat_vs_ref_trends.csv`). A plausible (untested) physical reason: blue light is absorbed
  strongly by the myoglobin forms whose balance changes as the surface dries and oxidises.
- **Caution:** these best numbers were found by trying ~100 model × feature-set combinations, so
  they are optimistic (winner's curse). They are a **hypothesis to confirm on new sessions**,
  not the reported result. The pre-registered model stays all-lights linear.

## 3. What each feature group adds — `feature_group_ablation.csv`
- Brightness ("intensity") alone already gives balanced acc. 0.38–0.41 (HGB/RF/linear) — as good as
  or better than all features (0.31–0.35). Extra groups add more noise than signal with 9 pieces.
- Glint features make the linear age model unstable (MAE 13.1 h with intensity + glint): glint
  counts differ strongly between pieces/setups, so a linear model extrapolates wildly. Texture
  and ratios bring it back to 4.0–4.2 h.

## 4. Single-shot vs session-normalised (why a grey card matters) — `relative_vs_single_shot.csv`
Dividing each feature by the same piece's first photo (`__rel`, needs the start of filming):
balanced acc. 0.42–0.51 (linear 0.511 [0.384, 0.587]) versus 0.27–0.35 single-shot, matching the
mentor's first-frame baseline (0.489). The piece-to-piece offset is the main obstacle. A grey /
white reference card in every photo would give single photos a similar reference
(`docs/04_DATA_COLLECTION_CHANGES.md`).

## 5. Calibration and abstention — `calibration.csv`, `reliability.csv`, `ood_abstention.csv`, F13
- Uncalibrated, the linear classifier is badly over-confident: ECE 0.567, Brier 1.216.
  Grouped sigmoid calibration lowers ECE to 0.296 (Brier 0.957), **but balanced accuracy falls
  from 0.313 to 0.224**. The S09 rule ("keep calibration if it lowers ECE") kept it in the saved
  model. This trade-off is an open question (PROGRESS #12).
- OOD (Mahalanobis, threshold = 99th percentile of leave-one-piece-out distances inside the
  training pieces): abstains on 3.6 % of held-out triplets; accuracy on kept 0.202 vs all 0.218.
  The phase's plain rule (in-sample distances) abstains on **100 %** of new pieces and is useless.
  Sanity check: when RED and BLUE features are swapped (a wrong-LED photo), 51 % are flagged, so
  the detector catches gross setup errors only about half the time.

## 6. What the model looks at — `importance_*.csv`, `partial_dependence.csv`, F14, F15
Shuffling a feature on the held-out piece and measuring the loss:
- Level classifier: the BLUE family (+0.051 balanced acc.) and GREEN family (+0.046) matter
  most, then cross-light ratios (+0.044); the RED family adds nothing (−0.002).
- Age regressor: the RED family (+0.84 h) and the RED/BLUE brightness ratio (`roi_mean__ratio_RB`,
  +0.83 h) matter most. The partial dependence (F15) shows the predicted age rising as the
  RED/BLUE ratio rises, i.e. **the meat loses blue reflectance faster than red as it ages**
  (consistent with section 2), and falling as GREEN texture entropy (`lbp_entropy__GREEN`) rises.
- Importances vary strongly between held-out pieces (std larger than the mean): no single
  feature is a reliable marker on its own.

## 7. Failure analysis — `per_piece_selected.csv`
| Held-out piece | MAE equiv. h | Balanced acc. | Why |
|---|---|---|---|
| exp4_D1A (25 h overnight) | 6.42 | 0.25 | when it is held out only ONE other piece (exp5) shows `spoiled_risk` |
| exp2_D4A | 6.38 | 0.33 | not yet explained (largest age error among short sessions; check its photos in P7) |
| exp1_D1A | 4.63 | 0.75 | only 19 complete triplets (blue LED off until 13:34) |
| exp1_D1B, exp2_D2B | 0.87, 1.58 | ≈ 0.52 | short sessions close to the training average |
The two overnight pieces are the only source of the `quality_loss` / `spoiled_risk` levels, so
every model is weakest exactly where freshness matters most. Representative photos:
`reports/figures/F6_start_mid_end_<session>.png`, crop review `crops_sample_S09.png`.

## Saved model — `models/foodlens_spoil_v1/` (model.joblib not in git; model_card.md is)
Linear (C = 0.1, top-20 features) calibrated classifier + ridge equivalent-hour regressor + OOD
detector, trained on all 9 pieces. Intended use: research prototype for the same box, LEDs and
camera settings only; indicative, never a food-safety decision.

## What would help most (recommendations)
1. **More pieces**, especially long sessions with an unchanged setup (user plans 3: open
   question 11) — the CIs are wide because there are only 9 groups and 2 spoiled pieces.
2. **A grey/white card in every photo** (section 4): the session-relative result suggests this is
   the largest single gain available.
3. Pre-register a **BLUE-only** model and test it on the new sessions (section 2).
