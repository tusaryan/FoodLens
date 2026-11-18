# Notes for the data-collection team

The team's scripts in `data-collection-code/` are **left unchanged**. Corrected logic is implemented inside `src/foodlens/` (Phase 1–2), so the ML pipeline does not depend on them. Share this list with the team.

## A. `shorting_code.py` (sorting into RED/GREEN/BLUE)
| # | Issue | Effect | Recommended change |
|---|---|---|---|
| A1 | Light chosen by `int(elapsed // 60) % 3` | A photo taken 59.9 s after a boundary (timer jitter: we see 59–61 s gaps) can get the **previous** light. One shifted photo makes every later label wrong | Use the photo **sequence index** (`i % 3`) when the capture interval equals the light period, **and** verify with the dominant pixel channel |
| A2 | No check that the LED was actually on | EXP-1 D1A has 24 black "BLUE" photos, which went into the analysis as zeros | Flag frames whose 95th-percentile brightness < 3/255 as `DARK` and leave them out of RED/GREEN/BLUE |
| A3 | Time read from the **filename** only | Renamed or copied files lose time | Read `DateTimeOriginal` (+ `SubsecTimeOriginal`) from EXIF; fall back to the filename |
| A4 | Hard-coded Windows path `MAIN_FOLDER` | Must edit code to run | Command-line argument: `python shorting_code.py <folder>` |
| A5 | CSV dates as `dd-mm-YYYY` | Ambiguous in Excel | ISO 8601 `2026-08-28T12:19:50` |
| A6 | No record of camera settings | Cannot prove settings were fixed | Add ISO, exposure, WB, focal length columns from EXIF |
| A7 | Scene-change frames (lights on, hand in view) kept | Outliers | Flag frames whose brightness jumps > 5× the session median |

## B. `intensity.py` (mean RGB per image)
| # | Issue | Effect | Recommended change |
|---|---|---|---|
| B1 | Mean over the **whole photo** | 73–96 % of pixels are black background, so the meat signal is diluted ~5–10× and camera drift changes the value | Use the meat ROI mask; report the **median** (Paper 24) plus percentiles |
| B2 | Mean, not median | Glints (very bright) pull the mean up | Median plus p10/p90; glints measured separately |
| B3 | GUI folder picker (tkinter) | Not scriptable, fails on servers | CLI argument |
| B4 | Sorted by name | OK only if names are zero-padded | Sort by EXIF time |
| B5 | Output has no time column | Analysis must re-join with the CSV log | Include the timestamp and elapsed seconds |

## C. Capture protocol suggestions for NEW data (low cost)
1. **Put a grey/white reference card** (a cheap 18 % grey card or a white PTFE/paper strip) in every frame next to the meat. This enables calibrated, single-photo prediction. *Most important.*
2. Set **white balance to manual** (fixed Kelvin); it is currently auto.
3. Raise exposure so the meat's median is ~100–150/255 without clipping, or shoot **RAW (DNG)** if the phone allows.
4. Log temperature and humidity with a ₹300–500 digital thermo-hygrometer: a photo of its display every 30 min, or a CSV.
5. Record in `samples.yaml`: purchase date/time, storage before the experiment (fridge or room, hours), piece id, cut (breast/thigh), skin on/off.
6. Keep the camera and box fixed. Mark the meat position with tape.
7. **More independent pieces matter more than more hours per piece.** 10 pieces × 3 h is far more useful for ML than 2 pieces × 15 h.
8. Save one photo **with all LEDs off** at the start (dark frame) and one with a **plain white card** under each LED (flat field).
9. **(S10, for the CNN)** Keep the **framing** (camera height, zoom, meat position) identical across sessions. Today the only two pieces that reach `spoiled_risk` (EXP-4, EXP-5) were filmed with a smaller framing than EXP-2, so a CNN could learn "small picture = old meat". At least **3 long (≥ 16 h) sessions with the same setup**, and ideally short sessions with that setup too, so every freshness level appears under one framing.
10. **(S10)** Note any time the piece is touched, moved or turned over (exp3_D1A was excluded for this).

## D. Open questions for the team (answers go into `configs/samples.yaml`)
1. What do `D1`, `D2`, `D3` mean: storage day since purchase? And `A`/`B`: different pieces or the same piece?
2. EXP-2 runs 31-08 to 03-09, but folders exist only for 01-09 and 02-09. Where was the chicken stored between sessions (fridge or room) and for how long?
3. When was each piece bought, and was it chilled before?
4. Were the LEDs the same model/brightness for all experiments? Why was the blue LED off in D1A until ~13:34?
5. Were the D2B/D3A photos rotated (phone orientation changed)?
