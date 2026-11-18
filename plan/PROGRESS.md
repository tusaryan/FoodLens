# PROGRESS: single source of truth for "where are we"

## Current state (rewrite every session, 3–6 lines)
- S08 (2026-11-18) finished **P5** (T1–T5): labels (`foodlens labels`, 9 sessions / 1305 triplets; exp3_D1A excluded by user decision), grouped splitters, metrics + group bootstrap + grouped permutation test, RunReport JSONs, 4 baselines, leakage demo. check.sh PASS (141 tests).
- Labels: 4 freshness levels by equivalent hours at 25 °C (2 / 6 / 12 h, config); every session = own piece; `continues:` in samples.yaml re-groups a piece with no code change. Headline CV = leave-one-piece-out (9 groups), sensitivity = leave-one-experiment-out (4 groups).
- Results (`reports/tables/model_comparison.csv`, `leakage_demo.csv`): best single-shot baseline = RED median, MAE 5.7 h vs 6.6 h for the mean; all-feature RF (grouped) MAE 4.4 h, bal. acc 0.35 (chance 0.25). LEAKY random split: 0.17 h / 0.98 = the leakage demo.
- New `foodlens crops`: meat-only crops (lit + shadow, background 0) of all 4817 frames for P7 + visual review (12 new frames, all clean).
- Next: **P6** classical models on the P5 harness (`foodlens evaluate`). Carry-over: refresh P3 outputs on 10 sessions with triplet ROIs.

## Phase checklist
- [x] Planning & research (S01)
- [x] P0 Foundation: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S02)
- [x] P1 Ingest & QC: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S03)
- [x] P2 ROI & glints: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] (S04, S05)
- [x] P3 EDA: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] T6 [x] (S06)
- [x] P4 Features: T1 [x] T2 [x] T3 [x] T4 [x] (S07)
- [x] P5 Labels & evaluation: T1 [x] T2 [x] T3 [x] T4 [x] T5 [x] (S08)
- [ ] P6 Classical models: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ]
- [ ] P7 Deep features (optional-but-recommended): T1 [ ] T2 [ ] T3 [ ] T4 [ ]
- [ ] P8 Inference, notebook, README, report: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ] T6 [ ]
- [ ] P9* API · [ ] P10* Web · [ ] P11* Android/voice

## Key numbers (filled by code outputs; never by hand)
| Metric | Value | Source file |
|---|---|---|
| Frames total / usable (S07, 10 sessions) | 4860 / 4817 | reports/tables/qc_summary.csv (TOTAL row, usable = ok + warn) |
| Frames with ROI ok | 4829 / 4829 | reports/tables/roi_summary.csv (sum of roi_ok) |
| Triplets total / complete | 1624 / 1574 | reports/tables/feature_counts.csv (TOTAL row) |
| Pooled meat-brightness slope R / G / B (%/h, MixedLM; S06, OLD 5 sessions) | −4.0 / −4.3 / −5.4 | reports/tables/eda_key_numbers.csv |
| Promising feature × light combinations (sign agrees ≥ 4/5; S06, OLD 5 sessions) | 100 / 123 (mostly redundant brightness) | reports/tables/eda_key_numbers.csv |
| Labelled triplets / sessions (exp3_D1A excluded; S08) | 1305 / 9 | reports/tables/label_summary.csv |
| Mean baseline, LOGO piece: MAE equiv h [95 % CI] (S08) | 6.61 [5.58, 7.65] | reports/tables/model_comparison.csv |
| RED-median baseline, LOGO piece: MAE equiv h [CI], perm. p (S08) | 5.66 [3.93, 7.11], p = 0.001 | reports/tables/model_comparison.csv |
| Glint-count (old method), LOGO piece: MAE equiv h (S08) | 10.53 (worse than the mean) | reports/tables/model_comparison.csv |
| Mentor first-frame baseline (not single-shot): MAE / bal. acc (S08) | 4.99 / 0.49 | reports/tables/model_comparison.csv |
| RF all single-shot features: grouped vs LEAKY MAE equiv h (S08) | 4.35 [1.60, 6.51] vs 0.17 | reports/tables/leakage_demo.csv |
| RF all features: grouped vs LEAKY balanced accuracy, 4 levels (S08) | 0.35 [0.21, 0.53] vs 0.98 | reports/tables/leakage_demo.csv |
| Best grouped balanced accuracy (class, single-shot model) | ? (P6) | reports/metrics/… |
| Best grouped MAE hours (single-shot model) | ? (P6) | reports/metrics/… |

## Open questions (for user / data team)
1. Meaning of D1/D2/D3 and A/B; is any piece filmed in two sessions? (blocks `piece_id`, `storage_day`): I don't know about it but mot probably they are seperate test for that experiment with D1 to Dx representing the days like day 1, day 2, etc. and A, B, C suggest number of experiment done in the same day, they might be simultaeuos or one after another.
2. Storage between EXP-2 sessions (fridge?) and purchase times (blocks absolute age).: I am not sure about it but most of the meat sample are taken freshly and then it travels to college department in room temp by prof. and then handover to student to collect the data/capture image in room temp condition with this lighting condition of red, blue, green and camera, angle and related settings fixed.
3. Freshness class boundaries: default proposal in P5 (confirm with the mentor).: Instead of saying the exact age we can have level of freshness like fresh, med eatable, stale, stale and not recommended to be consumable. I have just suggested a hint, it should not be named like this exactly but we can have similar multiple logical level of distinction so that we can classify them clearly(along with a i button when UI is made so that it can also depict a range of exact age belonging in this category to have some idea of how old it can be, but this depends on the storage condition, temp, etc. So, that should be dynamic in nature as something can be stale within a day and same meat can remain fresh when stored in refrigerated environment/with some chemicals on it to let it remain fresh. So don't hard code it rather decide it dynamically including a factor like depending on training data, testing and real world testing with one image of meat and its condition in which it is captured to predict this freshness and age. While saying that this is only indicative and not exact age when anyone click that i button).
   - **S08 decision (implemented, mentor to confirm):** 4 levels by **temperature-equivalent hours at 25 °C** since the start of filming: `fresh` < 2 h, `use_soon` 2–6 h, `quality_loss` 6–12 h, `spoiled_risk` ≥ 12 h (`configs/default.yaml` labels; names and thresholds are config, not code). This is the "dynamic" part you asked for: the level depends on the temperature history, so at 35 °C `spoiled_risk` starts after ~7 clock hours, at 20 °C after ~17 h, and in a fridge almost never (`reports/tables/level_clock_ranges.csv` = the future "i" button text). Thresholds were fixed before any model result and are approximate (2 h = USDA "2-hour rule"; 12 h ≈ spoilage flora reaching ~10^7 CFU/g at 25 °C). Learning the boundaries from the photos instead would be circular (the model would define the classes it is graded on), so they stay pre-registered; later real-world data can refine them. Always shown as indicative, never exact.
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

9. (S07, for the data team) exp3_D1A at 19:55 (+7 h): the meat brightens +59/+111/+27 % R/G/B in one step while the tray changes only +7–16 %, so the piece was probably moved or turned over. Was it? Brightness is step-corrected; glint/texture after the step should be reported as a sensitivity case (like D2A).: I think exp3_D1A does not have reliable data, so i think we should drop it. For the brightness issue i think there might be some hinderence/setup might be mistakely moved that lead to refocus and brightness increase or similar isue might have persisted due to huma interference. So i think we should ignore that dataset exp3_D1A. But one important lesson you missed that i also prompted in my main prompt while writing the plan that we have to first process, clean, crop and do all standard data preprocessing steps before doing this kind of task for image processing/deep learning. Here the learning was we have to crop the image before doing any analysis for bateria detection. The cropping should only include the meat piece region - both the shadow/dark and bright region, we only have to crop and remove the background other than the meat piece so that we can analyze the dark and bright region behaviour of only meat correctly for bacterial growth/changes and moisture changes in meat piece.
   - **S08 done:** exp3_D1A is excluded from labels/evaluation via `exclude: true` + reason in `configs/samples.yaml` (raw data and features kept; one line flips it back). On cropping: every feature since P2 is computed **only inside the meat mask**, so background pixels never entered a feature; the "full" mask already includes the shadowed side (`full_*`, `shd_*`). S08 makes this explicit as a preprocessing step: `foodlens crops` writes a meat-only crop per frame (full meat = lit + shadow, background set to 0) for P7 and the visual review (`reports/figures/crops_sample_S08.png`).
10. (S07, for the data team) EXP-3/4/5 used a different framing (meat 4–7 % of the frame vs 12–25 %, fully lit, so no shadow zone: `shd_*` NaN) and BLUE becomes very dim after ~5 h. Was the setup (camera distance, LED) changed? Also: EXP-2 D1A/D2A/D2B/D3A/D4A are on 31-08, 01-09, 02-09, 03-09: same piece(s) stored between days (fridge?) or a new piece each day? This decides `piece_id` / `storage_day` and the CV grouping in P5 (if the same piece, group by experiment).: i am not sure about it but lets assume it as seperate new piece most probably, so for now treat them as new piece for exp. But ensure that if i later inform you that if any of the exp is continuation of any bef then ensure that our pipeline can adapt to it immediately without much codebase change, for now treat them as new unless i specifically say that it is continuation.
   - **S08 done:** every session is its own piece (`piece_id` = session id). If you later learn that a session continues an earlier one, add `continues: <earlier session>` (optionally `storage_temp_c`) to its samples.yaml entry and rerun `foodlens labels && foodlens evaluate`. That needs no code change: the piece_id (CV group), storage_day and the cumulative age then follow automatically (tested). As a safety check every model is also scored with **leave-one-experiment-out** (in case the EXP-2 days are one piece).
11. (S08, for the user/mentor) The `shd_*` (shadow-zone) features are missing exactly in the EXP-3/4/5 framing, so "is the value missing?" reveals the long overnight sessions: a setup proxy. They are excluded from the models (`evaluation.drop_feature_prefixes`) but kept in the table. Confirm, or ask the data team to keep the framing fixed in future sessions.

## Recent sessions (newest first; keep 5 in full)
### S08: 2026-11-18: P5 Labels & evaluation, T1–T5 (Claude Opus)
- Did:
  - User answers applied: exp3_D1A `exclude: true` (samples.yaml); every session its own piece + `continues:`/`storage_temp_c` chain support (`samples_config.py`); 4 freshness levels by equivalent hours + clock-hour ranges per temperature ("i" button); meat-only crop stage `foodlens crops` (`vision/crop.py`).
  - T1 `labels/{thermal_age,labels}.py` (+ T_min −10 °C sensitivity: < 2.4 % change); T2 `evaluation/splits.py`; T3 `evaluation/{metrics,stats,cv}.py` (RunReport JSON); T4 `models/baselines.py`, `evaluation/leakage_demo.py` (F10b); T5 `evaluation/run.py`, CLI `labels` / `evaluate --model` / `crops`.
  - `shd_*` excluded from models (missingness = setup proxy); post-step sensitivity only for exp2_D2A (exp5 steps are tray-only).
- Files: configs/{default,samples}.yaml, src/foodlens/{samples_config,config,errors,cli}.py, labels/*, evaluation/*, models/baselines.py, vision/crop.py, tests/test_{labels,evaluation,errors,features,skeleton}.py, reports/{metrics,tables,figures}, plan/*.
- Verified: labels full (1305 rows); evaluate --limit 20 then full (48 runs, ~9 min); crops --limit 20 then full (4817 ok, 18.5 min); F10b + 12-frame crop sheet inspected; check.sh PASS (141 tests).
- Next: P6.

### S07: 2026-11-18: P4 Features, T1–T4 + new data (Claude Opus)
- Did:
  - Data: samples.yaml fixed (`DATE_…` renames) + 5 new sessions; session folder fallback accepts `exp-5_D1A`; `E_DUPLICATE` for byte-identical copies; ingest/qc/roi/eda-sheets rerun on 4860 images.
  - QA sheets showed the meat moving in the overnight sessions → `vision/triplet_roi.py` (per-triplet RED/BLUE masks, neck cut r=25, per-frame reference).
  - T1 `features/{specs,schema}.py` + dictionary; T2 `features/build.py` (relative glints with edge band, abs baseline, ratios, LBP, W_BLUR NaN, step correction), `features/run.py` (cache, stage); T3 `features/triplets.py` (wide `__LIGHT`, cross-light ratios/log-ratios, `__rel`); T4 `features/qc.py`.
- Files: configs/{default,samples}.yaml, src/foodlens/{features/*,vision/{texture,triplet_roi}.py,io/manifest.py,errors.py,config.py,cli.py,eda/{run,frame_stats}.py}, tests/test_{features,manifest,config,errors,skeleton}.py, reports/{tables,figures}, .claude/skills/image-pipeline, plan/*.
- Verified: features --limit 20 then full (4817 frames, 1624 triplets, 0 all-NaN columns, 23 min); 34-frame visual sample; check.sh PASS (123 tests).
- Next: P5.

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
## Older sessions (one sentence each)
- S03 (2026-11-18, P1): ingest + QC (manifest with EXIF/sequence/temperature, pixel light verification, DARK/MIXED/BLUR flags, inventory notebook).
- S02 (2026-11-18, P0): uv project, package skeleton, typed errors, logging, validated pydantic config, CLI stubs, check.sh and 40 synthetic-image tests.
- S01 (2026-11-18, planning): dataset inventory (944 jpg, 5 sessions), first-pass EDA (D1A blue-LED failure, glint trend flip), Papers 23/24, docs/01–06, AGENTS.md, skills and the phased plan.
