# P8: Predictor, end-to-end notebook, README and report (2 sessions). Graded deliverable.

**Skills to load:** `session-protocol`, `notebook-authoring`, `ml-evaluation`.
**Model:** the final model chosen in P7D.T4: normally the CNN (`models/foodlens_cnn_v1/`), else
P6 linear (`models/foodlens_spoil_v1/`). The Predictor supports both behind one interface; the
model card's `family` decides.
**Session split:** S-a = T1–T3, S-b = T4–T6.

## P8.T1: Predictor (`inference/predictor.py`, `inference/result_schema.py`)
- `Predictor.load(model_dir)`; `predict(images: list[PathOrArray], light_tags: list[str] | None, temp_c: float | None) -> PredictionResult`.
- Steps: validate → light check (tags vs pixels) → triplet ROI → meat-only crops →
  - CNN: `dl.composite.build_composite` + `scale_normalise` + `grid_patches` + the eval transform (the **same** functions as training) → torch model (TorchScript; CPU fallback when MPS is absent) → embedding OOD → temperature-scaled probabilities.
  - Classical fallback: `extract_frame_features` (the **same** function as training) → triplet → OOD → model → calibrated probabilities.
- `PredictionResult` (pydantic, `schema_version="1"`): `status` (ok | abstain | error), `age_hours` {point, low, high} (interval from the CV residual quantiles), `freshness_class`, `class_probs`, `confidence`, `explanation` (CNN: Grad-CAM overlay path + channel importances; classical: `top_features` [{name, value, effect}]), `warnings` [codes], `error` {code, message, hint}, `model_version`, `timings_ms`.
- Every failure path returns `status=error|abstain` with a code. **No exceptions escape to callers except programming errors.**
- Optional apps use the ONNX export from P7D.T4; a parity test checks ONNX vs torch on one triplet.
- **Tests:** a synthetic triplet → ok-shaped result (tiny untrained CNN saved to a tmp model dir, CPU); a dark image → error E_DARK_FRAME; a random-noise image → abstain/E_NO_MEAT_FOUND; schema round-trip JSON.
- CLI: `foodlens predict img1.jpg img2.jpg img3.jpg [--temp 30]` prints the JSON.

## P8.T2: End-to-end notebook `notebooks/src/00_FoodLens_end_to_end.py`
Sections (markdown in simple language + short code cells calling the package): 1 Problem & physics → 2 Data & QC → 3 ROI & glints → 4 EDA highlights → 5 Features (classical) → 6 Labels & thermal age → 7 Fair evaluation (leakage demo) → 8 CNN: composite triplets, patches, training without peeking → 9 Results: CNN vs classical on the same folds, light ablation, Grad-CAM → 10 Predict on a held-out piece → 11 Limitations & future work (data: more long same-setup sessions, grey card). It runs in < 10 min using cached stage outputs and saved CNN fold predictions (it never retrains the CNN) (and tells the user which CLI to run if the cache is missing).

## P8.T3: Reproducibility script
`scripts/reproduce_all.sh`: ingest → qc → roi → crops → features → labels → evaluate → train (P6 experiments) → finalize → dl-data → embed → train-cnn → explain/finalize CNN → notebooks, each with `tee` logs and a final `ALL STAGES DONE`. Run it fully once (classical ≈ 2 h, CNN several hours on MPS: flag `--skip-cnn` reuses the saved CNN outputs; follow `long-running-commands`).

## P8.T4: README final (see the current README skeleton)
Setup (uv; torch/timm size and weight caches), quick start, the full pipeline, how to add a new session (samples.yaml + temperature mode), how to predict, results table (quoted from the CSV), theory summary (link docs/01), tech stack, cleanup/uninstall, references.

## P8.T5: Report draft `reports/report/FoodLens_report.md`
Follow `docs/05_REPORT_OUTLINE.md`. Every number is pulled by a small script (`scripts/fill_report_numbers.py`) from the metrics/tables into `{{placeholders}}`. Export to DOCX/PDF with pandoc if available (`brew install pandoc` → list in uninstall script).

## P8.T6: Final audit (fresh session recommended, strong model)
Use the checklist in `plan/phases/P8_audit_checklist.md` (create it): leakage, banned features, CI presence, figure-number consistency, notebook executes, README commands work in a fresh clone (`git clone . /tmp/fl && cd /tmp/fl && uv sync && bash scripts/check.sh`).
