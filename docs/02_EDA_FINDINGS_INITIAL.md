# First-pass EDA findings (2026-11-18)

Script: `reports/eda_initial/firstpass_eda_script.py`. Per-frame output: `reports/eda_initial/frame_stats_firstpass.csv`.
Method: every photo decoded at 1/8 resolution (JPEG draft mode), **whole frame, no ROI yet**, so the numbers are indicative only. Phase 3 redoes this properly on ROI crops.

## 1. Inventory
| Session folder | Photos | Start | End | Duration | Orientation |
|---|---|---|---|---|---|
| EXP-1 / exp1_D1A | 132 | 28-08 12:19:50 | 14:30:51 | 2.2 h | landscape 3468×4624 stored |
| EXP-1 / exp1_D1B | 118 | 28-08 14:32:24 | 16:29:25 | 1.95 h | landscape |
| EXP-2 / DATE-01-09 / exp2_D2A | 169 | 01-09 11:03:59 | 13:52:32 | 2.8 h | landscape |
| EXP-2 / DATE-01-09 / exp2_D2B | 181 | 01-09 13:59:39 | 16:59:41 | 3.0 h | **rotated** (4624 wide) |
| EXP-2 / DATE-02-09 / exp2_D3A | 344 | 02-09 08:17:30 | 14:00:33 | 5.7 h | **rotated** |
| **Total** | **944** (~472 MB) | | | | |

- Interval: 60 s (one 92 s gap in D2A). Light cycle R→G→B is strict in all sessions.
- Camera: OnePlus CPH2381, **manual exposure** (ExposureMode=1), ISO 100, 1/60 s, f/1.7, 4.71 mm, flash off, **auto white balance** (WB=0).
- Pieces: the shapes in `figures/sheet_all.jpg` suggest D1A ≠ D1B, D2A ≠ D2B, and D3A is a different orientation. **Needs confirmation from the data team** (see the open questions in `04_DATA_COLLECTION_CHANGES.md`).

## 2. Brightness
- Photos are **very dark**: 73–96 % of pixels have max-channel < 15/255; whole-frame dominant-channel mean is ~5–13/255. The meat ROI is brighter (~70/255 red per the team's cropped table). Little of the 8-bit range is used, so expose brighter or shoot RAW in future (see `04_…`).
- LEDs are narrow-band: under red light, mean G and B are ≈ 0.03–0.5.
- No clipped (≥250) pixels at 1/8 scale. Specular glints may still clip at full resolution; Phase 3 checks this.

## 3. Data-quality defects found
1. **Blue LED failure, EXP-1 D1A:** 24 of 44 blue-slot photos are **completely black** (R=G=B=0) up to `IMG_20260828_133351.jpg`. These produced the `0 / 1.59e-05` rows in the team's cropped Excel and break "divide by first image".
2. **Scene-change frame:** the last D1A blue frame (`14:30`) shows the room with lights on and paper in view. It must be removed automatically.
3. **Camera drift:** frame-1 vs last-frame shift ≈ (−4, +7) px at 1/8 scale (~56 px full-res) in D1A. Other sessions ≤ 2–4 px, apart from one unreliable estimate in D2B. Per-frame ROI detection is needed.
4. Schedule-based labels (`shorting_code.py`) agree with the pixel colour in all sessions **except the dark frames**. The `floor(elapsed/60)` rule is still fragile: one 59 s interval at a boundary would mislabel everything after it (see `04_…`).

## 4. Trend with time (whole frame, Spearman ρ vs elapsed time)
| Session | Light | n | hours | ρ(brightness) | ρ(spot count) | last/first brightness |
|---|---|---|---|---|---|---|
| D1A | R / G / B | 44/44/20 | 2.2 | −0.90 / −0.93 / −0.45 | +0.94 / +0.90 / +0.06 | 0.99 / 0.99 / 1.46* |
| D1B | R / G / B | 40/39/39 | 1.9 | −0.99 / −1.00 / −0.99 | −0.80 / −0.88 / −0.94 | 0.93 / 0.92 / 0.90 |
| D2A | R / G / B | 57/56/56 | 2.8 | +0.31 / +0.31 / +0.38 | +0.39 / +0.35 / +0.32 | 1.14 / 1.37 / 1.52 |
| D2B | R / G / B | 61/60/60 | 3.0 | −0.96 / −0.99 / −0.66 | −0.68 / −0.78 / −0.82 | 0.90 / 0.86 / 0.92 |
| D3A | R / G / B | 115/115/114 | 5.7 | −0.96 / −0.99 / −0.42 | −0.37 / +0.72 / −0.80 | 0.91 / 0.93 / 1.02 |
\* inflated by the scene-change frame.

**Reading:** brightness mostly falls within a session, which fits surface drying. The spot-count direction is **inconsistent** across sessions. Visual check (`figures/sheet_exp2_D3A.jpg`): an even, wet sheen at 08:17 turns into **fewer, concentrated glints along the fibres** by 11:08–13:59. That looks like evaporation and the film breaking into droplets, the "split/merge/evaporate" behaviour the team predicted. This supports using spot **size distribution and total glint area** as features, not raw counts.

## 5. Consequences for the plan
- ROI segmentation, light verification and frame QC come **before** any feature or model work (Phases 1–2).
- The unit of evaluation is the **piece/session**. Only 5 sessions exist, so we use leave-one-session-out and report confidence intervals.
- A between-day signal (D1 vs D2 vs D3, if those are storage days) is likely stronger than within-session hours. Both tasks are defined in Phase 5.

## Figures
- `figures/sheet_exp1_D1A.jpg`: start/mid/end × R/G/B, contrast-stretched (note the blue scene-change frame).
- `figures/sheet_exp2_D3A.jpg`: start/mid/end × R/G/B.
- `figures/sheet_all.jpg`: first vs last red frame of each session.
