# 07 · EDA findings (Phase P3, session S06)

**For:** the mentor and the project team. **Data:** 919 usable photos from 5 sessions (one
chicken piece each, 2–5.7 h at room temperature), each lit in turn by a red, green and blue LED.
Every number below comes from a file written by `foodlens eda-stats` / `foodlens eda`; the file is
named next to it. Notebook: `notebooks/03_eda.ipynb`. *Glints* = tiny bright reflections of the
wet surface (moisture). They are **not** bacteria.

## How we analysed it (in one paragraph)
For every photo we measured the **same meat pixels** (the lit "ROI") under its own light: the
brightness (median, mean, percentiles), glints (two rules, see finding 5), texture (GLCM), the
shadowed side of the meat separately, and, as a control, the brightness of the lit tray and holder
around the meat. Then we asked, per session and per light, whether each value rises or falls with
time (Spearman rho, Theil–Sen slope), whether it does so in the same direction in every session
(sign consistency), and fitted one pooled model over all sessions (mixed model with a separate
starting level per session).

## Findings
1. **The meat gets darker over time, in every session and under every light.** The pooled model
   gives −4.0 %/h (RED), −4.3 %/h (GREEN) and −5.4 %/h (BLUE) of the starting brightness, each
   with a narrow 95 % CI (`tables/eda_key_numbers.csv`, `tables/eda_mixed_models.csv`;
   figure `F5_relative_all_sessions.png`). The start / middle / end photos show a uniform darkening
   (`figures/F6_start_mid_end_<session>.png`, blue = darker).
2. **It is the meat, not the lamp.** The lit tray / holder / walls around the meat (`ref_mean`) stay
   flat or brighten slightly (median +0.9 to +2.4 %/h) while the meat darkens (median −3.3 to
   −8.5 %/h) (`tables/eda_key_numbers.csv`). So LED warm-up or camera drift does not explain the
   trend. Exception: in `exp2_D2B` the surroundings brighten by about 21–26 %/h (liquid on the tray
   or the meat settling; see the bottom edge in `F6_start_mid_end_exp2_D2B.png`).
3. **Which light is most promising?** All three carry the darkening signal, with different
   strengths (`tables/eda_light_comparison.csv`, `figures/F8_signal_heatmap.png`):
   - RED has the most features that are significant in all 5 sessions (20).
   - GREEN has the best brightness signal-to-noise (trend ≈ 30× the frame-to-frame noise per hour).
   - BLUE changes fastest in relative terms but is the dimmest (ROI ≈ 16/255) and noisiest; its
     photos are visibly quantised (`figures/visual_sample_S06_1.png`).
   - The red/green/blue **ratios** of one 3-minute cycle also drift consistently (`roi_mean`
     red/green ratio rises in 5/5 sessions): green and blue fall faster than red.
   Which light *predicts* best is decided in P6 by the light-ablation experiment under grouped CV.
4. **Absolute brightness differs more between pieces than it changes within a piece.** The starting
   RED ROI mean ranges from 38 to 75 grey levels across sessions, while one session loses only a
   few % per hour (`tables/eda_key_numbers.csv`). A single photo's absolute brightness therefore
   cannot tell the age of an unknown piece. The model needs normalised or relative features (ratios
   between lights, meat ÷ surroundings), and P6 must test this with leave-one-session-out.
5. **Glint counts depend on the detection rule (open question 7).** With the P2 rule (fixed floor of
   40 grey levels) the glint count goes up in some sessions and down in others (falls in only 2–3
   of 5 sessions per light; this is the old D1A/D1B sign flip). With the light-independent rule (a
   glint must be ≥ 1.75 × the surrounding meat) the RED glint count falls in **5/5** sessions
   (`tables/eda_key_numbers.csv`, `tables/eda_glint_dynamics.csv`). In the varied visual sample the
   fixed-floor rule counts bright surface grain in the brightest band as glints and misses specks
   in dimmer parts (`figures/visual_sample_S06_2.png`, `_3.png`).
6. **Glints get smaller over time** in 15 of 15 session × light series with the relative rule (13/15
   with the fixed rule). With the relative rule, RED shows "evaporation" (fewer, smaller) in 5/5
   sessions, GREEN mostly "film breaking" (more, smaller; 4/5) and BLUE is mixed
   (`tables/eda_glint_dynamics.csv`, `figures/F7_glint_dynamics_rel.png`). This fits a drying
   surface film breaking into smaller patches. About 53–78 % of glints
   persist from one photo to the next, so glints are mostly fixed by surface geometry, with
   detection noise near the threshold.
7. **Texture (GLCM) changes less consistently** (contrast falls in 4/5 sessions per light) than
   brightness (`tables/feature_signal_ranking.csv`). It is still "promising" by the 4/5 rule.
8. **The shadowed side of the meat (open question 6)** is 3–21 % of the meat area
   (`tables/eda_zones.csv`, `figures/qa_zones.png`). It also darkens in 5/5 sessions (`shd_mean`), but
   its level is only about 4–18 grey levels (close to the noise and JPEG steps), and the RED-defined
   shadow is much less dark under BLUE (the LEDs light the meat from different angles).
9. **The D2A colour jump (open question 8)** at 12:59:32 raised the ROI brightness by about 1.3×
   (RED), 1.6× (GREEN) and 1.9× (BLUE) (`tables/eda_step_gains.csv`). After correcting this step,
   D2A darkens like the others. Meat ÷ surroundings removes the jump without any correction under
   GREEN and BLUE, but not fully under RED (`tables/eda_trends.csv`, variant `raw`). Glint values jump
   at the step too and are not corrected.
10. **One defocused photo** (`exp2_D3A`, 13:41:33, the W_BLUR warning) loses almost all glints
   (4 vs ≈ 100 in its neighbours) while its brightness is normal (`figures/visual_sample_S06_2.png`).

## What this means for modelling
- Use brightness features (mean, percentiles) of all three lights. Prefer the **mean** over the
  median: the median of 8-bit pixels moves in whole grey levels (steps in F5).
- Add light-independent / relative features: cross-light ratios, meat ÷ surroundings, relative
  glints. These are what can transfer from one piece to another.
- Run the full R / G / B / pairs / all light ablation in P6 (now in the P6 phase file).
- For W_BLUR frames, set glint and texture features to "missing" (brightness stays valid).
- Treat D2A after 12:59:32 as a sensitivity case for glint features.

## Limitations
- **Only 5 sessions (pieces).** "5/5 sessions agree" is encouraging, not proof. Confidence
  intervals in the models will be wide.
- Photos minutes apart are not independent, so per-session p-values are optimistic. The pooled
  model and the cross-session agreement are the honest summaries.
- Time since the start of filming is not the meat's true age (purchase and storage history
  unknown, open questions 1–2).
- White balance was automatic. Ratios within one photo can drift; ratios between separately lit
  photos are safer.
- Correlation with time is not causation: darkening and glint shrinkage fit surface drying and
  ageing, but this EDA does not measure chemistry or microbes.
- 100 of 123 feature × light combinations pass the 4/5 rule, but most are near-duplicates of the
  same brightness signal (`tables/feature_signal_ranking.csv`).
