# P7B: Frozen pretrained embeddings + small heads (1 session)

**Goal:** the cheapest, safest deep-learning baseline. Pretrained networks are run once (no
weights trained); a small logistic / ridge head is fitted inside each training fold. Answers
"do pretrained image features carry a between-piece age signal at all?" and gives the CNN light
comparison cheaply. Uses the P7A dataset (eval transform only) and the existing CV code.
**Skills to load:** `session-protocol`, `ml-evaluation`, `long-running-commands`.

## P7B.T1: Embedding extraction (`dl/embed.py`, replaces the `models/embeddings.py` stub)
- Backbones (from `dl.frozen_backbones`, all timm, pretrained): `vit_small_patch14_dinov2`,
  `convnext_tiny`, `efficientnet_b0`, `mobilenetv3_large_100`; optional
  `vit_base_patch16_clip_224.openai` if time allows. Global-pooled features, `eval()` mode,
  `torch.no_grad()`.
- Per triplet: embedding of the composite (mean over the fixed patch grid) AND one embedding per
  light (single-light mode) for the light comparison.
- Cache `data/processed/emb_<backbone>_<hash>.parquet` keyed by `triplet_id` (+ model name, weight
  file hash, torch/timm versions). `foodlens embed --backbone <name> [--limit 20]`, `STAGE DONE`.
- **Tests:** shape and determinism on a synthetic composite with a tiny randomly-initialised timm
  model on CPU (no download in tests); cache key changes when the backbone changes.
- **Accept:** `--limit 20` then full run per backbone (log the time; expected minutes on MPS).

## P7B.T2: Heads under the P6 evaluation
- Models = sklearn `Pipeline` (scale → PCA fitted in-fold, n_components from config ≤ 64 →
  L2 logistic / ridge), registered so `evaluation.cv.oof_predict` and `evaluate` run unchanged.
  Optional: embedding + the 153 hand-crafted features concatenated.
- Same targets, folds (LOGO piece headline, leave-one-experiment-out), metrics, group-bootstrap
  CIs and RunReports as P6; rows appended to `reports/tables/model_comparison.csv`
  (family `frozen_<backbone>`). Grouped permutation test (≥ 200 reps, cheap here).
- Add the **within-piece rank correlation** metric (Spearman ρ of predicted vs true hours inside
  each held-out piece, mean and per piece) to `evaluation/metrics.py`; report it for P6 models too.
- **Accept:** one row per backbone × task × scheme, all with CIs; nothing selected on the test fold.

## P7B.T3: Light comparison with frozen features
- Feature sets: RED-only, GREEN-only, BLUE-only embeddings, R+G+B late fusion (concat), and the
  composite. Paired per-piece comparison vs the composite (`evaluation/paired.py`) →
  `reports/tables/dl_light_ablation_frozen.csv` and a figure. Compare with the P6 finding
  (BLUE best, RED worst) in two sentences.

## P7B.T4: Short write-up
- Append a "Frozen embeddings" section to `docs/10_CNN_RESULTS.md` (create it): table quoted
  from the CSVs, the light comparison, one honest paragraph. Record which frozen backbone was
  best, but **the P7C backbone stays the pre-registered one** (docs/09 §4); a different backbone
  may only be added in P7C as a clearly labelled exploratory run.
