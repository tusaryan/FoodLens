# P3: Exploratory data analysis (1–2 sessions)

**Goal:** understand how the meat's appearance changes with time under each light, **before** any model. Every claim in the report's EDA chapter comes from here.
**Skills to load:** `session-protocol`, `image-pipeline`, `notebook-authoring`, `ml-evaluation` (stats section).
**Inputs:** `manifest.parquet`, `roi_index.parquet`, ROI crops. **Session split:** S-a = T1–T3, S-b = T4–T6.
Put the computation in `src/foodlens/eda/` (new sub-package; modules ≤ 300 lines). The notebook only calls it and plots.

## P3.T1: Per-frame ROI statistics (prerequisite)
`eda/frame_stats.py`: for each QC-ok frame, the ROI intensity stats (median, p10, p90, mean, std on the verified channel, excluding glints) + `glint_features`. Saved to `data/processed/frame_stats.parquet`. (P4 reuses these functions. Do not duplicate them.)

## P3.T2: Start / middle / end comparison
- For each session × light: the frames at 0 %, 50 %, 100 % of the duration. Show them side by side (contrast-stretched with the **same** stretch for all three so the brightness change is real), plus difference images (end − start) with a diverging colormap. → `F6`.
- A table of the median ROI intensity, glint count, glint area fraction at start/mid/end + % change.

## P3.T3: Time series
- Per session, 3 rows (R/G/B) × metrics [ROI median, glint area fraction, glint count/10⁴px, GLCM contrast]: raw value vs elapsed hours, plus a rolling median (window 5) and the relative-to-first version (the mentor's normalisation: first = 1.0). → `F5`.
- Trend statistics per session×light×metric: Spearman ρ with p-value, Theil–Sen slope per hour. A table in `reports/tables/eda_trends.csv`.
- Pooled test: `statsmodels` MixedLM `metric ~ hours` with a random intercept per session; report the slope ± CI. **This is the honest "does it change consistently across pieces?" answer.**

## P3.T4: Glint dynamics (the team's droplet hypothesis)
- Using `match_glints` between consecutive same-light frames: births, deaths, merges, splits per hour; glint size distribution (histograms at start/mid/end, log-x) → `F7`.
- Interpret in simple language: evaporation → fewer/smaller glints; film breaking → more, smaller; merging → fewer, bigger. State which pattern each session shows. Avoid causal bacterial claims.

## P3.T5: Which light carries the most information?
- For each feature × light: |Spearman ρ| with hours, averaged over sessions and the **sign consistency** (fraction of sessions with the same sign). A heat-map → `F8`.
- Cross-light ratio features (medR/medG etc.) on triplets: the same analysis.
- Output: a ranked table `reports/tables/feature_signal_ranking.csv`. **A feature is "promising" only if its sign is consistent in ≥ 4/5 sessions.**

## P3.T6: Notebook + written findings
- `notebooks/src/03_eda.py` built and executed.
- `docs/07_EDA_FINDINGS.md`: 8–15 bullet findings, each citing a figure/table file, written for the mentor in plain language; include "what this means for modelling" and "limitations".
- Update PROGRESS key numbers.
