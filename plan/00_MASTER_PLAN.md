# FoodLens: Master Plan

**Goal (graded):** a reproducible, leakage-free ML model plus a notebook that estimates the room-temperature age and freshness class of raw chicken from R/G/B-LED photos, with an honest evaluation and a report.
**Optional:** FastAPI backend → web app → Android app → voice assistant.

Read `AGENTS.md` first. Each phase file is self-contained: goal, inputs, tasks with exact specs, acceptance checks, skills to load, and the session split.

| Phase | File | Sessions | Depends on | Output |
|---|---|---|---|---|
| P0 | `phases/P0_foundation.md` | 1 | — | uv env, package skeleton, config, check.sh, CI-like tests |
| P1 | `phases/P1_ingest_qc.md` | 1–2 | P0 | `data/processed/manifest.parquet`, QC rejects, temperature cache |
| P2 | `phases/P2_roi_glints.md` | 1–2 | P1 | ROI masks, crops, glint detector, QA contact sheets |
| P3 | `phases/P3_eda.md` | 1–2 | P2 | EDA notebook + figures F5–F8, written findings |
| P4 | `phases/P4_features.md` | 1 | P2 (P3 informs) | `features.parquet` (frame + triplet), feature dictionary |
| P5 | `phases/P5_labels_eval.md` | 1 | P4 | labels, splitters, metrics, baselines, leakage demo |
| P6 | `phases/P6_classical_models.md` | 1–2 | P5 | model comparison, light ablation, calibration, chosen model |
| P7A | `phases/P7A_cnn_data.md` | 1 | P5 (crops) | torch/timm, composite-triplet dataset, scale normalisation, interior patches, leak-free augmentation, pre-registered CNN config |
| P7B | `phases/P7B_frozen_embeddings.md` | 1 | P7A | frozen pretrained embeddings + in-fold heads, CNN light comparison |
| P7C | `phases/P7C_finetune_cnn.md` | 2 | P7A | **main model**: fine-tuned CNN (nested grouped early stopping), light + reference-pair ablations |
| P7D | `phases/P7D_cnn_explain_select.md` | 1 | P7B, P7C | Grad-CAM, calibration/OOD, CNN vs classical on the same folds, final model + card + ONNX |
| P8 | `phases/P8_inference_notebook_report.md` | 2 | P7D | `Predictor` API, end-to-end notebook, README, report draft |
| P9* | `phases/P9_P11_optional_apps.md` | 1 | P8 | FastAPI service |
| P10* | `phases/P9_P11_optional_apps.md` | 2–3 | P9 | Next.js web app |
| P11* | `phases/P9_P11_optional_apps.md` | 3+ | P9 | Android capture app + voice agent |

\* optional, only after P8 is complete and the user confirms.

**Critical path for grading: P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7A → P7B → P7C → P7D → P8.** About 15–17 sessions.
**S10 restructure (user decision):** the deep-learning / CNN track (P7A–P7D) is the **main model**; the P6 classical models are its baselines on the same leave-one-piece-out folds. Rationale, options and honest expectations: `docs/09_DEEP_LEARNING_PLAN.md`.

## Definition of Done (every task)
- The code exists in `src/foodlens/`, is typed and documented, and is under the size limits.
- Tests exist and pass. `scripts/check.sh` passes.
- The stage runs from the CLI (`--limit 20` and full) and writes its output plus a `STAGE DONE` line.
- PROGRESS.md is updated and the work is committed.

## Decisions already made (do not re-open without the user)
1. Keep the mentor's method (R/G/B LEDs, smartphone, intensity, normalisation). Refinements only.
2. Grouped evaluation by session/piece. The leakage demo is shown, never used as the headline.
3. Bright spots = glints (moisture/gloss) features, not bacteria counts.
4. Labels: `hours_since_start` (primary regression), `equiv_hours_25C` (temperature-adjusted), `freshness_class` (bins defined in P5; confirm with the user), `storage_day` (if confirmed by the data team).
5. Temperature: configurable per sample (`weather` | `manual` | `log`); PEC Chandigarh default coordinates 30.765, 76.786.
6. Single-shot prediction uses absolute features only. Session-relative features are for EDA/science.
7. uv + `.venv`, Python 3.12, jupytext notebooks. **S10 (user):** PyTorch + timm pretrained CNN (fine-tuned) is the headline model; scikit-learn classical models are the baselines. The evaluation scheme (grouped LOGO piece, nested tuning, group-bootstrap CIs, permutation test) and the labels do NOT change.
8. The reference-pair (Siamese) CNN needs a reference photo, so like the `__rel` features it is reported as a separate, non-single-shot track (decision 6 stands unless the user changes it).
