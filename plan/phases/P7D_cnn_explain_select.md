# P7D: What the CNN looks at, calibration/OOD, CNN vs classical, final selection (1 session)

**Goal:** show WHAT the network uses (glints, fibre texture, colour patches, edges), make its
confidence honest, compare it with P6 on the same folds, apply the pre-registered selection rule
and save the final model with a card and an export for the optional apps.
**Skills to load:** `session-protocol`, `ml-evaluation`, `long-running-commands`, `image-pipeline`.

## P7D.T1: Grad-CAM on held-out pieces (`dl/saliency.py`)
- Grad-CAM (own ~40-line implementation with hooks on the last conv stage; no new dependency)
  using each outer fold's model on ITS held-out piece only.
- Figure `reports/figures/F16_gradcam_<level>.png`: for each level, held-out composites with the
  map overlaid (varied sample, new frames; LESSONS).
- Numbers `reports/tables/dl_saliency_zones.csv`: share of saliency mass on glint pixels (P2
  relative glint detector), the mask edge band, and the interior, per level and piece. A map that
  sits mostly on the edge band = shape/setup shortcut → say so in docs/10.
- Light importance: zero one channel of the composite on the held-out piece → drop in balanced
  accuracy / rise in MAE (`dl_channel_occlusion.csv`). Compare with the P6 permutation importance.
- **Tests:** Grad-CAM on a tiny model returns a map of the input size in [0, 1].

## P7D.T2: Calibration and abstention for the CNN
- Temperature scaling of the ordinal logits fitted on the INNER-fold predictions only, scored
  out-of-fold (ECE, Brier, reliability figure), keep it only if the grouped ECE falls (S09 rule);
  answer open question 12 with the user before choosing for the app.
- OOD: `inference/ood.MahalanobisOOD(mode="grouped")` on the pooled CNN embeddings, fitted inside
  each training fold; report abstention rate and accuracy on kept triplets, plus the RB-swap check.

## P7D.T3: CNN vs classical (same folds)
- One table `reports/tables/cnn_vs_classical.csv`: P6 linear, best frozen (P7B), CNN main
  (seed mean), baselines; balanced accuracy, MAE, within-piece ρ, CIs, permutation p; paired
  per-piece differences with CIs (`paired.py`). Figure F17 (per-piece paired dots).
- Leakage demo for the CNN (report only): the same CNN with a RANDOM triplet split
  (`random_frame_splits_LEAKY`), 1 seed, next to its grouped score (expected ≈ 0.98 vs grouped).

## P7D.T4: Final selection (rule pre-registered in `dl.selection` in P7A, BEFORE any result)
- The **headline model is the pre-registered CNN main model**, reported whatever its score.
- It becomes the saved final model unless the paired per-piece comparison shows it significantly
  WORSE than P6 linear on balanced accuracy (CI of the difference entirely below 0) or its CI lower
  bound does not beat the best single-shot baseline; then P6 linear stays the saved model and
  docs/10 says why. Ablations never become the final model (exploratory).
- Final fit on all 9 pieces with E* = median of the outer-fold E*, 3 seeds (ensemble of 3 or the
  seed-0 model: fixed in config). Save `models/foodlens_cnn_v1/` (weights gitignored; `model_card.md`
  tracked: data, folds, metrics with CIs, Grad-CAM summary, limits: same box/LEDs/camera only).
- Export: TorchScript and ONNX (`torch.onnx.export`, opset from config) of the preprocessing-free
  network; a parity test (ONNX Runtime vs torch on a synthetic batch, max |Δ| < 1e-4) if
  `onnxruntime` is added (`uv add --group dev onnxruntime`, note why).
- docs/10 final section + update docs/08 "Saved model" to point to the CNN (or explain why not).
