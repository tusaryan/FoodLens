# P1: Ingest, temperature and frame QC (1–2 sessions)

**Goal:** one trustworthy table (`manifest.parquet`) describing every raw frame, with verified light and QC status. Every later stage reads only this table.
**Skills to load:** `session-protocol`, `image-pipeline`, `long-running-commands`.
**Session split:** S-a = T1–T3, S-b = T4–T6.

## P1.T1: Manifest builder (`io/manifest.py`, `io/exif.py`)
For each `*.jpg|*.jpeg|*.png` under `paths.raw_dir` (recursive, sorted, ignoring folders containing `_sorted`, case-insensitive):
| column | how |
|---|---|
| `frame_id` | `<session_id>__<stem>` |
| `path` | relative to the repo root |
| `sha1` | of the file bytes (for change detection) |
| `experiment` | the first path part matching `EXP-\d+` → `EXP-1` |
| `session_id` | the folder name matching `exp\d+_D\d+[A-Z]` (e.g. `exp1_D1A`) |
| `day_code`, `piece_code` | parsed from the session id: `D1`, `A` (raw codes; meaning set in samples.yaml) |
| `ts` | EXIF DateTimeOriginal + SubsecTimeOriginal (tz Asia/Kolkata); fallback to the filename `IMG_YYYYMMDD_HHMMSS`; neither → reject `E_NO_TIME` |
| `ts_source` | `exif` / `filename` |
| `exposure_s, iso, fnumber, wb_mode, exposure_mode, focal_mm, make, model` | EXIF (None if missing) |
| `width, height, orientation` | after `exif_transpose` |
| `seq_idx` | 0-based order within the session by `ts` |
| `elapsed_s` | `ts − session start` |
| `schedule_light` | `["RED","GREEN","BLUE"][seq_idx % 3]` (config: `capture.light_order`) |
- Session metadata from `configs/samples.yaml` is merged in (piece_id, storage_day, temp mode). An unknown session → warning plus defaults.
- Use joblib parallel for the EXIF reads. Cache: skip unchanged sha1 unless `--force`.
- **Tests:** a synthetic folder tree in tmp with 6 fake JPEGs (write EXIF with Pillow), including one without EXIF and one with a bad name → correct columns and rejects.

## P1.T2: Light verification and frame QC (`qc/light_check.py`, `qc/frame_qc.py`)
Implement exactly the rules in the `image-pipeline` skill (dark, dominance, mismatch). Also compute and store per frame (at work resolution): `p95_max`, `dom_channel`, `dom_ratio`, `global_mean_<c>`, `lap_var`.
- Columns added: `pixel_light` ∈ {RED, GREEN, BLUE, DARK, MIXED}, `light` (final = pixel_light if valid), `qc_status` ∈ {ok, reject, warn}, `qc_reasons` (semicolon list).
- Scene-change detection needs the ROI → implement it in P2.T4 and re-run QC there. Leave a TODO marker and a test placeholder (skipped) here.
- **Tests:** synthetic red → RED; black → DARK; mixed equal channels → MIXED.

## P1.T3: CLI `foodlens ingest` + `foodlens qc`
- Write `data/processed/manifest.parquet` and `reports/tables/inventory.csv` (per session: n frames, start, end, duration h, n per light, camera settings unique values).
- `reports/tables/qc_summary.csv` (per session × reason counts) and `reports/qc_rejects.csv` (frame_id, reasons).
- **Accept (real data):** total 944 frames; exp1_D1A has 24 DARK frames (blue slot); the other sessions have 0 DARK; schedule vs pixel mismatches = only those DARK frames. If the numbers differ, investigate before continuing and record the explanation in LESSONS.md.

## P1.T4: Temperature module (`io/temperature.py`)
- `TemperatureSource` protocol with `get_series(session) -> pd.Series[°C] indexed by hourly ts`.
  - `WeatherSource`: Open-Meteo archive `https://archive-api.open-meteo.com/v1/archive` (`hourly=temperature_2m,relative_humidity_2m`, lat/lon/tz from config). **Cache** the JSON in `data/cache/weather/<lat>_<lon>_<start>_<end>.json`. Offline + no cache → `DataError(E_TEMP_UNAVAILABLE)` with a clear message suggesting `manual` mode. Network timeout 20 s, 2 retries.
  - `ManualSource`: a constant from samples.yaml `temp.value_c`.
  - `LogSource`: CSV `timestamp,temp_c[,rh]`, interpolated.
- Add `indoor_offset_c` (default 0.0) to the config: outdoor reanalysis ≠ room temperature, so the user can adjust.
- `temp_c` and `rh` are added to the manifest per frame (time interpolation).
- **Tests:** a mocked HTTP response (no network in tests), manual mode, log mode, the offline error.
- **Accept:** the real run fills `temp_c` for all frames (expected 25–35 °C range); figure `reports/figures/temperature_sessions.png`.

## P1.T5: Inventory notebook source `notebooks/src/01_data_inventory.py`
Markdown explains the dataset, capture protocol and QC findings in simple words; it shows the inventory table, QC summary, temperature plot and 1 example per rejection reason (thumbnails, contrast-stretched).

## P1.T6: Wrap up
check.sh PASS, full real-data run logged, PROGRESS key numbers (frames total/usable) filled from `qc_summary.csv`.
