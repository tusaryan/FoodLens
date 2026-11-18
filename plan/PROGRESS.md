# PROGRESS: single source of truth for "where are we"

## Current state (rewrite every session, 3–6 lines)
- S05 (2026-11-18) finished **P2** (T4–T5).
  - `qc/scene_qc.py`: local robust z for transient scene changes, `drift_px`, persistent-step `scene_segment`.
  - ROI-based `W_BLUR` (`lap_var_roi`) and `qc_pre_roi`; `ingest → qc → roi` now gives the final manifest and is idempotent.
  - `foodlens eda-sheets` → `reports/figures/qa_roi_<session>.png`. check.sh PASS (97 tests).
- Real data: 919 usable frames, 0 scene-change rejects (max |z| 4.0), 1 W_BLUR (D3A 13:41). One persistent scene step: exp2_D2A 12:59:32, +28/+45/+65 % R/G/B brightness, 54 frames after it.
- QA sheets: the holder is excluded (D3A). The ROI covers 75–90 % of the meat (the dim shadowed side is excluded). The glint threshold is always the floor of 40, so BLUE glint counts collapse on dim sessions (open questions 6–7).
- Next: **P3.T1–T3** (frame stats, start/middle/end, time series).
- Blockers: data-team answers (docs/04 §D) still open; not needed until P5.

## Phase checklist
- [x] Planning & research (S01)
- [x] P0 Foundation: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S02)
- [x] P1 Ingest & QC: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S03)
- [x] P2 ROI & glints: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] (S04, S05)
- [ ] P3 EDA: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ] T6 [ ]
- [ ] P4 Features: T1 [ ] T2 [ ] T3 [ ] T4 [ ]
- [ ] P5 Labels & evaluation: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ]
- [ ] P6 Classical models: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ]
- [ ] P7 Deep features (optional-but-recommended): T1 [ ] T2 [ ] T3 [ ] T4 [ ]
- [ ] P8 Inference, notebook, README, report: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ] T6 [ ]
- [ ] P9* API · [ ] P10* Web · [ ] P11* Android/voice

## Key numbers (filled by code outputs; never by hand)
| Metric | Value | Source file |
|---|---|---|
| Frames total / usable | 944 / 919 | reports/tables/qc_summary.csv (TOTAL row, usable = ok + warn) |
| Frames with ROI ok | 919 / 919 | reports/tables/roi_summary.csv (sum of roi_ok) |
| Best grouped balanced accuracy (class) | ? | reports/metrics/… |
| Best grouped MAE hours | ? | reports/metrics/… |

## Open questions (for user / data team)
1. Meaning of D1/D2/D3 and A/B; is any piece filmed in two sessions? (blocks `piece_id`, `storage_day`)
2. Storage between EXP-2 sessions (fridge?) and purchase times (blocks absolute age).
3. Freshness class boundaries: default proposal in P5 (confirm with the mentor).
4. (S04, for the user) ROI = one fixed session ROI from RED frames applied to all lights, holder/tray removed by BLUE/RED > 1.0 at the ROI edge. Confirm this is acceptable (alternative: per-frame masks per light, which leak into the tray under BLUE).
5. (S04, resolved) exp2_D3A IMG_20260902_134133 (W_BLUR) is kept: its ROI is normal (RED IoU 0.94, normal area). Defocus mainly affects glint/texture features; decide in P4 whether to NaN those for W_BLUR frames.
6. (S05, for the user/mentor) The session ROI covers 75–90 % of the meat. It excludes the dim, shadowed side (RED median 11–18 vs 36–77 inside; clearest in D2B, where the right ~20 % of the piece is excluded). Keep "well-lit meat only" (current; low-SNR shadow pixels are out) or include shadowed meat? See reports/figures/qa_roi_exp2_D2B.png.
7. (S05, decide in P4) The glint threshold is always the absolute floor `glint.abs_min` = 40 (75/75 QA panels). On dim BLUE sessions the counts fall to 0–40 while specks remain visible, so glint counts mix up illumination level and moisture. Proposal for P4: scale the floor by the ROI brightness (e.g. top-hat > k × `sroi_mean`) or use per-light floors, validated on the QA sheets.
8. (S05, decide in P4/P5) exp2_D2A has a persistent +30–65 % brightness step at 12:59:32 (the 92 s capture gap; the meat did not move). Frames carry `scene_segment`. Should absolute-brightness features be corrected per segment, or should D2A be reported as a sensitivity case?

## Recent sessions (newest first; keep 5 in full)
### S05: 2026-11-18: P2 ROI & glints, T4–T5 (Claude Opus)
- Did:
  - `qc/scene_qc.py`: rolling-median robust z with MAD floors; mask geometry scored for RED only; log global mean and `sroi_mean` for all lights; `drift_px`; step detector → `scene_segment` + scene_steps.csv.
  - `apply_qc` ROI rules (`E_NO_MEAT_FOUND`, `E_SCENE_CHANGE`, ROI blur) and `qc_pre_roi`; roi stage re-runs qc.
  - `qc/qa_sheets.py` + `foodlens eda-sheets`; config `qc.scene_*` / `qc.step_*` / `qa`; skill updated.
- Files: src/foodlens/qc/{scene_qc,qa_sheets,frame_qc}.py, vision/roi_batch.py, stages.py, cli.py, config.py, viz.py, configs/default.yaml, tests/test_{scene_qc,qa_sheets,qc,roi_stage,skeleton}.py, reports/figures/qa_roi_*.png, reports/tables/{scene_steps,qa_sheet_frames,qc_summary}.csv, plan/*.
- Verified:
  - Smoke runs then full roi (919 ok) and eda-sheets (75 panels).
  - Acceptance met: 14:30:51 still MIXED; max loss 1/108 frames.
  - 3 sheets inspected (D3A, D1A, D2B); check.sh PASS.
- Next: P3.T1–T3.
### S04: 2026-11-18: P2 ROI & glints, T1–T3 (Claude Opus)
- Did: `vision/roi.py` (segment_roi with multi-Otsu lowest threshold + floor, session_reference_mask, exclude_foreign, mask_iou), `vision/glints.py` (top-hat with outside-ROI fill, detect_glints, glint_features), `vision/glint_dynamics.py` (match_glints: Hungarian + merge/split rules), `vision/roi_batch.py` + `stages.run_roi`, CLI `foodlens roi`; config keys roi.{otsu_classes,min_level,ref_lights,foreign_*}, glint.{merge_area_ratio,sat_level}; warning W_ROI_LOW_IOU; skill image-pipeline updated.
- Files: src/foodlens/vision/{roi,roi_batch,glints,glint_dynamics}.py, stages.py, cli.py, config.py, errors.py, configs/default.yaml, tests/test_{roi,glints,roi_stage}.py (+conftest level param), reports/tables/roi_{summary,sessions}.csv, plan/*.
- Verified: roi --limit 20 then full (919 frames, 1 min 37 s, 919 ok, no NaN); visual overlays checked (holder excluded in D1B/D2A/D2B/D3A); check.sh PASS.
- Next: P2.T4–T5.
### S03: 2026-11-18: P1 Ingest & QC (Claude Opus)
- Did: `io/exif.py`, `io/manifest.py` (sha1 cache, samples.yaml folder → session, rejects kept with codes), `io/images.py` (draft-mode loader), `io/inventory.py`, `qc/light_check.py`, `qc/frame_qc.py` (DARK/MIXED/W_LIGHT_MISMATCH/W_BLUR, stats cache), `io/temperature.py` (weather/fixed/log sources, JSON cache, offline error), `stages.py`, CLI ingest/qc, notebook 01, config sections `capture` + `temperature`, error codes E_NO_SESSION/E_MIXED_LIGHT/W_*.
- Files: src/foodlens/{io,qc}/*, stages.py, cli.py, viz.py, errors.py, config.py, configs/*, tests/test_{manifest,qc,temperature,cli_stages}.py, notebooks/src/01_data_inventory.py, reports/{tables,figures}, plan/*.
- Verified: full ingest (944 rows, ~5 s) + qc (~12 s); acceptance met (24 DARK in D1A only; mismatches = 24 DARK + 1 MIXED, explained); temp for all frames; check.sh PASS.
- Next: P2.
### S02: 2026-11-18: P0 Foundation (Claude Opus)
- Did: uv project (Python 3.12, deps from P0.T1, no torch), package layout per docs/03 §6 with docstring-only stage modules, `errors.py` (codes + typed errors), `logging_utils.py` (rich, `FOODLENS_LOG`), `viz.py`, pydantic `config.py` (`load_config`, `load_samples`, `config_hash`, `cfg.path()`), `configs/default.yaml` + `samples.yaml` (5 sessions, piece_id/storage_day/pre_age_h null), typer CLI stubs, check.sh + check_sizes.py + build_notebooks.sh + uninstall_global_tools.sh, synthetic-image fixtures and 40 tests.
- Files: pyproject.toml, uv.lock, .python-version, src/foodlens/**, configs/*, scripts/*, tests/*, README.md, plan/*.
- Verified: check.sh PASS (ruff, format, 40 tests in ~2 s, size limits); `foodlens --help` lists 9 commands; build_notebooks.sh --check executed a throwaway notebook successfully (removed).
- Next: P1.
### S01: 2026-11-18: Planning (Claude Opus)
- Did: dataset inventory (944 jpg, 5 sessions), EXIF check (manual ISO100 1/60 f1.7, auto WB), first-pass whole-frame EDA, found the blue-LED failure in D1A (24 black frames) and a scene-change frame, glint trend inconsistency; read Papers 23/24; wrote docs/01–06, AGENTS.md, skills, plan.
- Files: AGENTS.md, CLAUDE.md, GEMINI.md, docs/*, plan/*, .claude/skills/*, reports/eda_initial/*, .gitignore.
- Verified: EDA script ran on all 944 images (24 s). No package code yet, so no check.sh.
- Next: P0.

## Older sessions (one sentence each)
- (none)
