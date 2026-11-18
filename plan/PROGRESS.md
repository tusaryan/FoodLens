# PROGRESS: single source of truth for "where are we"

## Current state (rewrite every session, 3–6 lines)
- S04 (2026-11-18) finished **P2.T1–T3**: `vision/roi.py` (multi-Otsu ROI, session ROI, holder exclusion), `vision/glints.py` + `vision/glint_dynamics.py`, `foodlens roi` → `data/processed/roi_index.parquet` (919 rows, all roi_ok), session masks + crops under `data/interim/roi/<hash>/`. check.sh PASS (86 tests).
- ROI design (changed vs the skill draft, skill updated): ONE session ROI per session = median of the RED frame masks minus bluish holder/tray regions touching its edge; used for all lights. Per-frame masks are QC only (RED IoU vs session ROI 0.87–0.99).
- Glint detector works on real frames (5–456 glints per frame); the threshold is always the `glint.abs_min` floor (40). Check visually in P2.T5.
- Next: **P2.T4–T5** (scene-change/drift QC using roi_index, QA contact sheets with glints).
- Blockers: data-team answers (docs/04 §D) still open; not needed until P5.

## Phase checklist
- [x] Planning & research (S01)
- [x] P0 Foundation: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S02)
- [x] P1 Ingest & QC: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S03)
- [ ] P2 ROI & glints: T1 [x] T2 [x] T3 [x] T4 [ ] T5 [ ] (S04)
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

## Recent sessions (newest first; keep 5 in full)
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
