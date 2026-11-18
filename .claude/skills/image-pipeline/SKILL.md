---
name: image-pipeline
description: Conventions and exact algorithms for FoodLens image handling - manifest, light verification, frame QC, ROI segmentation, glint detection, texture features, triplets.
---
# Image pipeline conventions

## Reading
- Open with Pillow, apply `ImageOps.exif_transpose`, convert to RGB uint8. Never write back to `experiment/`.
- Working resolution: `config.vision.work_long_side` (default 1024 px) for ROI and colour features; full resolution only for glint statistics (`config.vision.glint_long_side`, default 2048).
- Cache derived arrays/PNGs under `data/interim/<stage>/<config_hash>/`.

## Light verification (`qc/light_check.py`)
- p95 of the max channel < `qc.dark_p95` (default 3) → `DARK`.
- Otherwise the light is the dominant channel by mean over pixels whose max channel > `qc.min_signal` (default 8). Require dominance ratio ≥ `qc.dominance_ratio` (default 0.6), else `MIXED` (reject).
- Compare with the schedule label (`sequence_index % 3` → R,G,B). A mismatch is a warning; the pixel label wins.

## Scene change / outliers (`qc/scene_qc.py`, applied by `foodlens qc` once `roi_index.parquet` exists; `roi` re-runs `qc`)
- Per session and light, time-sorted: robust z = (x − centred rolling median over `qc.scene_window` frames) / max(1.4826·MAD of the residuals, floor). A ROLLING median (S05), because a session median flags slow trends (D2B meat settling) and the persistent D2A step.
- Features: log lit-channel global mean and log mean of the lit channel inside the session ROI (`sroi_mean`) for all lights; log per-frame mask area and mask centroid (px) for `roi.ref_lights` (RED) only, because BLUE/GREEN per-frame masks join lit tray/box strips (S05).
- MAD floors (`qc.scene_floor_*`) dominate on real data (raw MADs ≈ 0.5 px / 0.5 %), so |z| > `qc.scene_z` means: >20 % mask area, >12 px centroid, >13 % brightness vs the neighbouring frames → `E_SCENE_CHANGE` (reject).
- `drift_px` = per-frame mask centroid distance from the session×light median (descriptive; BLUE is noisy).
- Persistent steps: median log global mean of the `qc.step_k` frames after vs before differs ≥ `qc.step_min_log` → session step (lights merged within `qc.step_merge_s`), `scene_segment` per frame, `reports/tables/scene_steps.csv`. Kept, not rejected.
- `qc_pre_roi` = status from the non-ROI rules; the ROI stage selects on it so re-runs are idempotent.
- Blur: variance of the Laplacian inside the session ROI (`lap_var_roi`, Laplacian of the full plane, variance over ROI pixels) below the session-light 5th percentile / `qc.blur_div` → `W_BLUR` (warning). Falls back to the whole frame until every valid frame has an ROI.

## Visual QA (`foodlens eda-sheets`, `qc/qa_sheets.py`)
- Per session: rows = lights, columns = `qa.positions` of the usable frames; session ROI outline yellow, glint centroids cyan, crop = ROI bbox + `qa.crop_margin`. `reports/figures/qa_roi_<session>.png` + `tables/qa_sheet_frames.csv` (count, threshold per panel).

## ROI segmentation (`vision/roi.py`)
1. Take the verified light's channel, Gaussian blur σ=2 px (at work resolution).
2. Multi-Otsu with `roi.otsu_classes` (3) classes; use the LOWEST threshold, floored at `roi.min_level` (S04: plain Otsu kept only the brightly lit half of the meat). Morphological close (disk r=7) then open (r=5). Fill holes.
3. Keep the largest connected component. Reject if its area fraction is outside [`roi.min_frac`, `roi.max_frac`] (defaults 0.03, 0.8).
4. Erode by `roi.erode_px` (default 6) to drop the edges/shadow rim (Paper 24 circular crop analogue).
5. Session ROI = per-pixel median of the masks of the `roi.ref_lights` frames (RED only; S04: GREEN/BLUE masks leak into the lit tray/holder), largest component. This ONE mask is the ROI for every frame of the session (all lights), so a triplet's R/G/B values come from the same pixels. Per-frame masks are kept for QC: a ref-light frame with IoU vs the session ROI < `roi.min_iou` → `W_ROI_LOW_IOU`.
6. Save the mask PNG and the bbox. Unit test with a synthetic ellipse on a dark background plus noise.

## Glints (`vision/glints.py`), i.e. "bright spots"
- Inside the ROI: white top-hat (disk r=`glint.tophat_r`, default 9 px at glint resolution) on the verified channel.
- Threshold = ROI median + `glint.k` × MAD (default k=6), and also ≥ `glint.abs_min` (default 40).
- Connected components with area ≥ `glint.min_area_px` (default 3).
- Features: count per cm² of ROI (ROI pixel area as proxy, reported as count per 10⁴ px), total glint area fraction, size quantiles (p50, p90), max size, mean eccentricity, fraction of saturated pixels, and the glint brightness median.
- Dynamics (session-level EDA only, not single-shot features): match glints between consecutive same-light frames (centroid distance < `glint.match_px`); count births, deaths, merges, splits. This quantifies the droplet merge/split/evaporation idea.

## Intensity and colour features (per light, inside ROI excluding glint pixels)
- median, p10, p25, p75, p90, IQR, mean, std (of the verified channel, 0-255).
- Cross-light ratios on a triplet: medR/medG, medG/medB, medR/medB, and log-ratios. These carry spectral (absorption) information.
- Relative-to-first-frame versions (`*_rel`), used only for EDA and the "session-normalised" experiment, never as default single-shot inputs.

## Texture
- GLCM (skimage `graycomatrix`, distances [1,3], 4 angles, 32 levels) contrast, homogeneity, energy, correlation; LBP (P=8, R=1, uniform) 10-bin histogram. Computed on the ROI bbox with mask.

## Triplets (`features/triplets.py`)
- Within a session, sorted by time: group consecutive R, G, B frames whose timestamps span ≤ `triplet.max_span_s` (default 200). A triplet's time = the mean of its frames. Missing lights → partial triplet with NaN plus a `has_<light>` flag.

## Banned as model features
Timestamp, file index, session id, orientation, image width/height, ROI centroid/bbox position.

## EDA measurements (P3, S06: `foodlens eda-stats`, `foodlens eda`, `foodlens eda-sample`)
- `eda/frame_stats.py` (P4 reuses it): per usable frame, at glint resolution on the verified channel inside the SESSION ROI: `roi_*` intensity stats with glint pixels (either rule) excluded, `glint_*` (abs rule), `rglint_*` (relative rule), `glcm_*`, `shd_*` / `full_*` (shadow zone), `ref_mean` + `roi_mean_rel_ref`.
- Relative glint rule (`detect_glints_relative`): top-hat > max(median + k·MAD, `glint.rel_floor`) AND top-hat ≥ `glint.rel_k` × the opening (local surface). Light-level independent; RED count falls in 5/5 sessions vs 2/5 for the abs rule. Known artefact: false glints on the ROI edge next to the bright holder under BLUE (drop an edge band in P4).
- GLCM: ROI quantised to `vision.glcm_levels` over its own p1–p99 (`vision.glcm_clip_pct`), outside = level 0 and dropped from the matrix: brightness-independent texture.
- Zones (`eda/meat_zones.py`): full meat = smoothed RED session median > `eda.shadow_min_level`, opened by `eda.shadow_open_r`, touching the lit ROI, holder removed, eroded; shadow = full − lit. Reference = lit non-meat (tray/holder/walls) ≥ `eda.ref_gap_px` from the meat and > `eda.ref_min_level` per light: the LED/camera-drift control.
- Scene steps: brightness features are divided by the per-segment gain (`eda.trends.step_correct`); glint/texture are not.
- Visual review: never re-inspect the same fixed frames. `foodlens eda-sample --tag S<NN>` renders random (per session × time third) + trend-outlier + special (W_BLUR, after-step) frames not yet in `tables/visual_review_log.csv` / `qa_sheet_frames.csv`.
