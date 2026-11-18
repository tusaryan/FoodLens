# P8: Predictor, end-to-end notebook, README and report (2 sessions). Graded deliverable.

**Skills to load:** `session-protocol`, `notebook-authoring`, `ml-evaluation`.
**Session split:** S-a = T1–T3, S-b = T4–T6.

## P8.T1: Predictor (`inference/predictor.py`, `inference/result_schema.py`)
- `Predictor.load(model_dir)`; `predict(images: list[PathOrArray], light_tags: list[str] | None, temp_c: float | None) -> PredictionResult`.
- Steps: validate → light check (tags vs pixels) → ROI → `extract_frame_features` (the **same** function as training) → triplet → OOD → model → calibrated probabilities.
- `PredictionResult` (pydantic, `schema_version="1"`): `status` (ok | abstain | error), `age_hours` {point, low, high} (interval from the CV residual quantiles), `freshness_class`, `class_probs`, `confidence`, `top_features` [{name, value, effect}], `warnings` [codes], `error` {code, message, hint}, `model_version`, `timings_ms`.
- Every failure path returns `status=error|abstain` with a code. **No exceptions escape to callers except programming errors.**
- **Tests:** a synthetic triplet → ok-shaped result; a dark image → error E_DARK_FRAME; a random-noise image → abstain/E_NO_MEAT_FOUND; schema round-trip JSON.
- CLI: `foodlens predict img1.jpg img2.jpg img3.jpg [--temp 30]` prints the JSON.

## P8.T2: End-to-end notebook `notebooks/src/00_FoodLens_end_to_end.py`
Sections (markdown in simple language + short code cells calling the package): 1 Problem & physics → 2 Data & QC → 3 ROI & glints → 4 EDA highlights → 5 Features → 6 Labels & thermal age → 7 Fair evaluation (leakage demo) → 8 Results & light ablation → 9 Predict on a held-out session example → 10 Limitations & future work. It runs in < 10 min using cached stage outputs (and tells the user which CLI to run if the cache is missing).

## P8.T3: Reproducibility script
`scripts/reproduce_all.sh`: ingest → qc → roi → features → labels → train (final experiments) → evaluate → notebooks, each with `tee` logs and a final `ALL STAGES DONE`. Run it fully once (≈ 30–60 min; follow `long-running-commands`).

## P8.T4: README final (see the current README skeleton)
Setup (uv), quick start, the full pipeline, how to add a new session (samples.yaml + temperature mode), how to predict, results table (quoted from the CSV), theory summary (link docs/01), tech stack, cleanup/uninstall, references.

## P8.T5: Report draft `reports/report/FoodLens_report.md`
Follow `docs/05_REPORT_OUTLINE.md`. Every number is pulled by a small script (`scripts/fill_report_numbers.py`) from the metrics/tables into `{{placeholders}}`. Export to DOCX/PDF with pandoc if available (`brew install pandoc` → list in uninstall script).

## P8.T6: Final audit (fresh session recommended, strong model)
Use the checklist in `plan/phases/P8_audit_checklist.md` (create it): leakage, banned features, CI presence, figure-number consistency, notebook executes, README commands work in a fresh clone (`git clone . /tmp/fl && cd /tmp/fl && uv sync && bash scripts/check.sh`).
