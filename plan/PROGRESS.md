# PROGRESS: single source of truth for "where are we"

## Current state (rewrite every session, 3–6 lines)
- S06 (2026-11-18) finished **P3** (T1–T6): `foodlens eda-stats` → `frame_stats.parquet` (919 × 50) + `glint_props.parquet`; `foodlens eda` → F5–F8 + `eda_*` tables; `foodlens eda-sample` = new varied visual sample per session; notebook 03; `docs/07_EDA_FINDINGS.md`. check.sh PASS (109 tests).
- Main result: meat brightness falls in 5/5 sessions under every light (MixedLM −4.0/−4.3/−5.4 %/h R/G/B); the lit tray/holder does not, so it is not LED drift. Absolute levels differ ~2× between pieces, so models need relative/ratio features.
- Light comparison (`eda_light_comparison.csv`): RED most features significant in 5/5 (20); GREEN best brightness SNR; BLUE fastest relative change but dim/noisy. P6 now runs the full R/G/B/pairs/all ablation.
- Relative glint rule (≥ 1.75 × surrounding meat) gives a consistent RED glint decline (5/5 vs 2/5 with the abs floor). Recommendations for open questions 4–8 are written under each answer below.
- Next: **P4.T1–T4** (feature specs, frame + triplet features). Blockers: data-team answers (docs/04 §D) still open; needed from P5.

## Phase checklist
- [x] Planning & research (S01)
- [x] P0 Foundation: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S02)
- [x] P1 Ingest & QC: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S03)
- [x] P2 ROI & glints: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] (S04, S05)
- [x] P3 EDA: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S06)
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
| Pooled meat-brightness slope R / G / B (%/h, MixedLM) | −4.0 / −4.3 / −5.4 | reports/tables/eda_key_numbers.csv |
| Promising feature × light combinations (sign agrees ≥ 4/5) | 100 / 123 (mostly redundant brightness) | reports/tables/eda_key_numbers.csv |
| Best grouped balanced accuracy (class) | ? | reports/metrics/… |
| Best grouped MAE hours | ? | reports/metrics/… |

## Open questions (for user / data team)
1. Meaning of D1/D2/D3 and A/B; is any piece filmed in two sessions? (blocks `piece_id`, `storage_day`): I don't know about it but mot probably they are seperate test for that experiment with D1 to Dx representing the days like day 1, day 2, etc. and A, B, C suggest number of experiment done in the same day, they might be simultaeuos or one after another.
2. Storage between EXP-2 sessions (fridge?) and purchase times (blocks absolute age).: I am not sure about it but most of the meat sample are taken freshly and then it travels to college department in room temp by prof. and then handover to student to collect the data/capture image in room temp condition with this lighting condition of red, blue, green and camera, angle and related settings fixed.
3. Freshness class boundaries: default proposal in P5 (confirm with the mentor).: Instead of saying the exact age we can have level of freshness like fresh, med eatable, stale, stale and not recommended to be consumable. I have just suggested a hint, it should not be named like this exactly but we can have similar multiple logical level of distinction so that we can classify them clearly(along with a i button when UI is made so that it can also depict a range of exact age belonging in this category to have some idea of how old it can be, but this depends on the storage condition, temp, etc. So, that should be dynamic in nature as something can be stale within a day and same meat can remain fresh when stored in refrigerated environment/with some chemicals on it to let it remain fresh. So don't hard code it rather decide it dynamically including a factor like depending on training data, testing and real world testing with one image of meat and its condition in which it is captured to predict this freshness and age. While saying that this is only indicative and not exact age when anyone click that i button).
4. (S04, for the user) ROI = one fixed session ROI from RED frames applied to all lights, holder/tray removed by BLUE/RED > 1.0 at the ROI edge. Confirm this is acceptable (alternative: per-frame masks per light, which leak into the tray under BLUE).: choose the best logical and most correct alternative for this yourself, so think deeply for this that should work in any an every condition dynamically and correctly.
   - **S06 recommendation (adopted):** keep ONE session ROI built from RED for all lights. RED masks are stable (IoU 0.90–0.99), while per-frame BLUE masks leak into the lit tray (IoU 0.33–0.87). One mask also means R, G and B are measured on exactly the same meat, which the cross-light ratios need. It works for any new session because it is rebuilt from that session's own RED frames. Resolved unless the mentor objects.
5. (S04, resolved) exp2_D3A IMG_20260902_134133 (W_BLUR) is kept: its ROI is normal (RED IoU 0.94, normal area). Defocus mainly affects glint/texture features; decide in P4 whether to NaN those for W_BLUR frames: please explain this in simple language easy to understand language, also what affects this and how it affect others. By this way i can suggest correctly what to do.
   - **S06 explanation:** a blurred photo smears each tiny bright spot into a soft glow, so it no longer stands out from the meat. The glint count collapses (4 vs ≈ 100 in the neighbouring photos) and texture looks smoother, but the average brightness stays correct. **Proposal for P4:** keep brightness features of W_BLUR frames, set glint and texture features to missing (NaN) for them. Only 1 of 919 frames is affected.
6. (S05, for the user/mentor) The session ROI covers 75–90 % of the meat. It excludes the dim, shadowed side (RED median 11–18 vs 36–77 inside; clearest in D2B, where the right ~20 % of the piece is excluded). Keep "well-lit meat only" (current; low-SNR shadow pixels are out) or include shadowed meat? See reports/figures/qa_roi_exp2_D2B.png.: I think we should include the shadow part as it will help us better understand how is the bright and dark spot are changing wrt to the complete meat sample and would help in doing eda and making correct decisions. What's your thought on this, what is ou recommendation, what should we do, think scientifically, logically and deeply.
   - **S06 recommendation:** keep the well-lit ROI as the main feature area AND measure the shadowed side separately (done: `shd_*`, `full_*` features, `figures/qa_zones.png`). Evidence: the shadow side darkens in 5/5 sessions too, but it is only 4–18 grey levels (near noise / JPEG steps), its size depends on how the piece sits, and it is much less dark under BLUE (the LEDs shine from different angles). So it is kept as extra features; P6 decides with the ablation whether it adds accuracy. Nothing is thrown away.
7. (S05, decide in P4) The glint threshold is always the absolute floor `glint.abs_min` = 40 (75/75 QA panels). On dim BLUE sessions the counts fall to 0–40 while specks remain visible, so glint counts mix up illumination level and moisture. Proposal for P4: scale the floor by the ROI brightness (e.g. top-hat > k × `sroi_mean`) or use per-light floors, validated on the QA sheets.: I don't know, and I was not able to understand what is the issue too, so please help me understand it and if you need my input afterward then ask me else do the best possible thing we should do for this.
   - **S06 explanation + decision:** the old rule calls a spot a glint only if it is at least 40 grey levels brighter than its surroundings. Under the dim BLUE light the whole meat is only ~15 grey levels, so almost nothing can be 40 brighter: the count measures the lamp brightness, not the moisture. The new rule asks "is the spot at least 1.75 × as bright as the meat around it?", which is the same question under every light. Evidence: RED glint count falls in 5/5 sessions with the new rule vs 2/5 with the old (`eda_key_numbers.csv`). **P4:** use the relative rule as the main glint features, keep the old rule as the "old method" baseline, and drop glints on the ROI edge band (false hits next to the bright holder under BLUE). No input needed unless the mentor objects.
8. (S05, decide in P4/P5) exp2_D2A has a persistent +30–65 % brightness step at 12:59:32 (the 92 s capture gap; the meat did not move). Frames carry `scene_segment`. Should absolute-brightness features be corrected per segment, or should D2A be reported as a sensitivity case? i don't know about it should do what is logically and scientifically suitable for our model to accurately and correctly detect features so that in future it can adapt and works accurately too for any other messy data image.
   - **S06 recommendation:** the jump is a camera colour-balance change (+33/+59/+87 % R/G/B inside the ROI), not the meat. (a) Brightness: divide by the per-segment gain (implemented: `eda.trends.step_correct`); (b) prefer features that cancel it: meat ÷ surroundings (`roi_mean_rel_ref`, removes it under G and B) and cross-light ratios; (c) glint features after the jump: report D2A as a sensitivity case. For the future: lock white balance on the phone and put a grey card in the box (docs/04).

## Recent sessions (newest first; keep 5 in full)
### S06: 2026-11-18: P3 EDA, T1–T6 (Claude Opus)
- Did:
  - T1: `eda/frame_stats.py`, `eda/meat_zones.py` (shadow zone + lit non-meat reference region), `vision/texture.py` (masked GLCM), `detect_glints_relative`; `foodlens eda-stats`.
  - Varied visual review (user request): `eda/visual_sample.py` + `foodlens eda-sample --tag S06` (27 new frames: random + trend outliers + W_BLUR / after-step), log `tables/visual_review_log.csv`.
  - T2–T5: `eda/{compare,trends,plots,dynamics,signal,run}.py`, `features/triplets.py`; `foodlens eda` → F5–F8, eda_* tables, ranking, light comparison, key numbers.
  - T6: notebook 03, docs/07; P6 light ablation widened to R/G/B/pairs/all.
- Files: src/foodlens/{eda/*,features/triplets.py,vision/{glints,texture}.py,cli.py,config.py}, configs/default.yaml, tests/test_eda.py, notebooks/src/03_eda.py, docs/07, reports/{figures,tables}, plan/*.
- Verified: eda-stats --limit 20 then full (919 frames, ~1.5 min); eda full (~27 s); figures and samples inspected; check.sh PASS (109 tests).
- Next: P4.T1–T4.
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
## Older sessions (one sentence each)
- S01 (2026-11-18, planning): dataset inventory (944 jpg, 5 sessions), first-pass EDA (D1A blue-LED failure, glint trend flip), Papers 23/24, docs/01–06, AGENTS.md, skills and the phased plan.
