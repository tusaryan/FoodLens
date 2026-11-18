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

## Scene change / outliers
- Per session and light: robust z-score (median/MAD) of the ROI area, ROI centroid and the global mean. |z| > `qc.scene_z` (default 6) → `SCENE_CHANGE`.
- Blur: variance of the Laplacian on the ROI below the session-light 5th percentile / 3 → `BLUR` (warning only unless extreme).

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
