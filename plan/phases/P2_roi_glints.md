# P2: ROI segmentation, glint detection, scene-change QC (1–2 sessions)

**Goal:** a reliable meat mask for every usable frame plus a validated glint (bright-spot) detector. This is the foundation of every feature, so quality checks matter more than speed.
**Skills to load:** `session-protocol`, `image-pipeline`, `long-running-commands`.
**Session split:** S-a = T1–T3, S-b = T4–T5.

## P2.T1: ROI segmentation (`vision/roi.py`)
- `segment_roi(img_rgb: np.ndarray, light: str, cfg) -> RoiResult(mask, bbox, area_frac, ok, reason)`. Algorithm: exactly the `image-pipeline` skill, steps 1–4.
- `session_reference_mask(masks) -> np.ndarray`: per-pixel median of the masks (same shape after resizing to work resolution).
- `mask_iou(a, b)`.
- **Tests:** an ellipse on noise → IoU with the ground truth > 0.9; a dark image → `ok=False, reason=E_NO_MEAT_FOUND`; two blobs → keeps the larger.

## P2.T2: Glint detection (`vision/glints.py`)
- `detect_glints(channel: np.ndarray, roi_mask, cfg) -> GlintResult(labels, props_df)` (props: area, centroid, eccentricity, mean/max intensity, saturated_frac).
- `glint_features(result, roi_area_px) -> dict` (the feature list in the skill).
- `match_glints(prev, curr, max_dist_px) -> dict(births, deaths, merges, splits, persist)`: Hungarian matching on centroids (scipy `linear_sum_assignment`). A merge = 2+ previous glints whose nearest current glint is the same one and whose area grew ≥ 1.5×; a split is the reverse.
- **Tests:** a synthetic image with 10 known dots → count 10 ± 0, areas within 20 %; dots outside the ROI are ignored; merge/split detected on crafted frame pairs.

## P2.T3: CLI `foodlens roi`
- For QC-ok frames: compute the mask at work resolution and save the PNG mask + crop (bbox, verified channel only, 8-bit) to `data/interim/roi/<hash>/`. Store `roi_area_frac, roi_cx, roi_cy, roi_ok, roi_reason, roi_iou_session` in `data/processed/roi_index.parquet`.
- Run `--limit 20` first, then everything (expected 5–15 min). Follow `long-running-commands`.

## P2.T4: Scene-change and drift QC (finish the TODO from P1.T2)
- Robust z-scores per session×light (skill rules). Mark `SCENE_CHANGE` rejects. `drift_px` = centroid distance to the session median.
- Re-write `manifest.parquet` QC columns (stage idempotent).
- **Accept:** the D1A frame `IMG_20260828_143051.jpg` (14:30 blue) is rejected as SCENE_CHANGE or DARK/MIXED; total rejects listed in `qc_summary.csv`; no session loses > 10 % of its non-dark frames (if it does, inspect and tune the thresholds with a written justification in LESSONS).

## P2.T5: Visual QA (mandatory, a human check is possible)
- `foodlens eda-sheets`: per session, a contact sheet of 12 frames (start/25 %/50 %/75 %/end × lights) with the mask outline in yellow and glint centroids as small cyan circles → `reports/figures/qa_roi_<session>.png`.
- The agent opens at least 2 sheets (image viewer/read tool) and writes 3–5 bullet observations in the session log. If masks visibly miss the meat or include the holder/background, fix the algorithm (e.g. adjust `roi.erode_px`, add a border-touching exclusion) before finishing.
- Note: the D3A blue frames show a bright blue object at the bottom (holder). Verify that it is excluded from the ROI.
