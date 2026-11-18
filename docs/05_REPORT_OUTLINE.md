# Final report: outline and figure/table plan

Target: 25–40 pages, IEEE-style references. Written in Phase 8 from artefacts the pipeline produces. **No number may be typed by hand: every number comes from `reports/tables/*.csv` or `reports/metrics/*.json`.**

| § | Section | Content | Figures / tables (file produced by) |
|---|---|---|---|
| 1 | Abstract | Problem, method, grouped result with CI, limitation | — |
| 2 | Introduction | Food safety, need for cheap screening, smartphone sensing | — |
| 3 | Literature review | Papers 15–24; scatter physics; ML on meat images; leakage | **T1** related-work comparison (manual, cited) |
| 4 | Theory | Bacteria size vs pixel size; Mie vs Rayleigh scattering; specular vs diffuse; myoglobin; drying; Ratkowsky thermal age | **F1** size-scale diagram; **T2** refractive indices (Paper 24) |
| 5 | Materials & data | Box, LEDs, phone, settings, sessions, schedule | **F2** setup schematic; **T3** dataset inventory (`reports/tables/inventory.csv`, P1) |
| 6 | Data QC | Dark frames, scene change, drift, light verification | **F3** rejected examples; **T4** QC counts (P1) |
| 7 | Pre-processing | ROI segmentation, glint detection | **F4** raw → mask → ROI → glints panel (P2) |
| 8 | EDA | Trends per light, start/mid/end, glint statistics, droplet merge/split/evaporation | **F5** time series per session × light; **F6** start/mid/end grid; **F7** glint size histograms over time; **F8** correlation heat-map (P3) |
| 9 | Features & labels | Feature list with physical meaning; labels; thermal age | **T5** feature dictionary (`features/schema.py` export, P4); **F9** temperature curves (P1) |
| 10 | Evaluation protocol | Grouped CV, baselines, CI, permutation test, leakage demo | **F10** split diagram; **T6** random vs grouped (P5) |
| 11 | Models & results | Model comparison, light ablation (R/G/B/all), calibration | **T7** model table with CI; **F11** confusion matrix; **F12** predicted vs true age; **F13** reliability curve; **F14** feature importance (P6–P7) |
| 12 | Discussion | What works, failure cases, why spot count alone fails, limits (n pieces) | **F15** failure gallery |
| 13 | Deployment (if done) | API/app architecture, OOD abstain | Diagrams from `03_ARCHITECTURE.md` |
| 14 | Conclusion & future work | More pieces, grey card, RAW, TVC lab counts for ground truth | — |
| A | Appendix | Config, reproducibility steps, model card | — |

Figure rules: 300 dpi PNG plus SVG, colour-blind safe palette, axis labels with units, one message per figure, the same light colours everywhere (R `#d62728`, G `#2ca02c`, B `#1f77b4`).
