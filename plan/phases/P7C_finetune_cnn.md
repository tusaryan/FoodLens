# P7C: Fine-tuned CNN (main model) + ablations, nested grouped early stopping (2 sessions)

**Goal:** train and evaluate the pre-registered main model (docs/09 §4: composite triplet,
EfficientNet-B0, last 2 stages trainable, ordinal level + equivalent-hours heads) under exactly the
P6 folds, plus the two ablations. **Session split:** S-a = T1–T3 (code + smoke + main model
running), S-b = T4–T5 (ablations, results).
**Skills to load:** `session-protocol`, `ml-evaluation` (incl. its CNN section),
`long-running-commands`.

## P7C.T1: Model and losses (`dl/model.py`, `dl/losses.py`)
- `FreshnessNet(backbone, trainable_stages, n_levels)`: timm backbone (pretrained, BN frozen in
  eval mode), global pooling, dropout, two heads: ordinal logits (n_levels − 1 cumulative
  thresholds) and equivalent hours. `predict_proba` converts cumulative probabilities to the 4
  level probabilities (monotone, sums to 1).
- Loss = ordinal BCE (class weights) + `dl.huber_weight` × Huber(hours / scale); sample weights =
  class weight × 1/(n triplets of the piece), both computed from the TRAINING pieces only.
- **Tests:** shapes on a tiny model (CPU); only the configured stages require grad; ordinal
  probabilities are valid; a loss step on a synthetic batch reduces the loss.

## P7C.T2: Training loop and nested early stopping (`dl/train.py`, `dl/cv.py`)
- `fit(model, train_rows, epochs, cfg, seed)`: AdamW (lr head / backbone from config), cosine
  schedule, one random patch per training triplet per epoch (near-duplicates: an epoch is one pass
  over triplets, not over all patches), seeded DataLoader (`num_workers` 0 or seeded workers).
- `cnn_oof_predict(df, groups, cfg)`: for every outer split from `evaluation.splits`
  (LOGO piece, leave-one-experiment-out): inner `GroupKFold(dl.inner_folds)` over the training
  pieces → per inner fold the epoch with the best inner balanced accuracy → E* = median →
  retrain on all training pieces for E* epochs → predict the outer test piece on the fixed
  patch grid (mean of patch probabilities / hours). **The outer test piece is never used for
  epoch choice, weights, calibration or any threshold.** `assert_no_group_overlap` on every split.
- Output: the same OOF table and RunReport format as P6 (family `cnn_main`), so `score`,
  bootstrap CIs, `paired.py` and `model_comparison.csv` work unchanged. Checkpoint per outer fold
  to `models/cnn_runs/<run_id>/` (gitignored); resumable if a run is killed.
- **Tests:** a spy test proves the outer test group never enters `fit` or the inner splits;
  E* comes only from inner folds; a 2-piece synthetic run completes on CPU in < 10 s.

## P7C.T3: Main model runs (`foodlens train-cnn --experiment cnn_main`)
- Smoke: `--limit 20` (few triplets, 1 epoch, 2 folds) → check the OOF table and RunReport.
- Full: LOGO piece + leave-one-experiment-out, seeds `[0, 1, 2]`, in the background with a log
  (`long-running-commands`; estimate the time from the smoke run first; expect hours on MPS).
  Report the mean over seeds and the seed spread; the headline row is the seed-mean prediction.
- Grouped permutation test: shuffle levels within pieces and retrain; as many reps as the time
  allows (≥ 19 → p resolution 0.05), run overnight; state the number of reps next to p.
- Extra honesty checks (docs/09 §4): within-piece rank correlation per piece; EXP-2-only LOGO
  (7 pieces); the normalised scale factor must not predict the level (logistic on the scale
  factor alone, LOGO, reported as a row).

## P7C.T4: Ablations (exactly two; configs in `configs/experiments/cnn_*.yaml`)
1. **Light:** the same model on single-light inputs (RED, GREEN, BLUE) vs the composite; paired
   per piece → `reports/tables/dl_light_ablation.csv`. Exploratory (state it).
2. **Reference-pair (Siamese):** a shared backbone embeds the current composite and the piece's
   FIRST complete triplet (the reference); the head sees [current, current − reference]. Not
   single-shot: reported next to the mentor first-frame baseline (0.489) and P6 `__rel` (0.511),
   never as the single-shot headline. Leakage note: the reference of a test piece is that test
   piece's own first photo (allowed: it is part of the input, like the mentor's method), never a
   training piece's photo, and its label is never used.
- Optional, only if time remains and labelled exploratory: the best P7B backbone fine-tuned.

## P7C.T5: Results write-up
`docs/10_CNN_RESULTS.md` "Fine-tuned CNN" section: headline table (CNN vs P6 linear vs baselines,
same folds, CIs, permutation p), paired per-piece wins, ablations, honesty checks, per-piece
failure table. Numbers only from the CSV/JSON files.
