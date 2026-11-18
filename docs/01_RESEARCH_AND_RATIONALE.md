# FoodLens: Research Basis, Approach and Justification

> Audience: the mentor, examiners and any new teammate. Plain language first, technical details after.
> Status: written 2026-11-18 after the first-pass data check (see `02_EDA_FINDINGS_INITIAL.md`).

---

## 1. What we are building (one paragraph)

We photograph a chicken piece inside a dark box. It is lit by **one coloured LED at a time: red, then green, then blue**, and the light changes every 60 s. A phone with **fixed (manual) camera settings** takes one photo per minute for several hours. From these photos, a machine-learning (ML) model learns how the **appearance of the meat surface changes as it ages at room temperature**. Given new photos taken the same way, it estimates **how long the chicken has been at room temperature** (an "age" in hours) and gives a **freshness class** (Fresh / Use-soon / Spoiled-risk) with a confidence score.

We keep the mentor's method (Paper 24 style: smartphone camera + controlled LED light + image intensity + normalisation). We do **not** change what the experiment is. We only make the analysis more careful so the result can be trusted.

---

## 2. What the two key papers did, and what we take from each

### Paper 24: Liang, Park & Yoon (2014), *Sci. Rep.* 4:5953, smartphone biosensor for meat
- **Setup:** an 880 nm near-infrared LED shines straight down onto ground beef. An iPhone camera photographs the **scattered light** at 15°, 30°, 45° and 60°. No chemicals or antibodies are used ("reagentless").
- **Signal:** the photo is cropped to a circle and converted to 8-bit grey. The **median** pixel intensity is taken, then **divided by the median of a clean control sample** ("normalised intensity", where control = 1).
- **Physics:** bacteria (refractive index n≈1.388) and bacterial fragments stick to fat particles (n≈1.40–1.46). This forms "pseudo-colonies" of about 1–5 µm. Particles of that size scatter light strongly (**Mie scattering**: particle size is close to the light's wavelength), and the **angle of strongest scatter changes as the clusters grow**. The bacteria are not seen directly. Their effect on light scattering is measured instead.
- **Result:** the *pattern* of intensity across angles revealed E. coli concentration from 10¹ to 10⁸ CFU/mL.

**What we adopt:** (1) the median of a cropped region instead of the whole-photo mean; (2) normalisation to a reference; (3) dark-room controlled lighting; (4) the idea that we measure **bulk optical change**, not individual bacteria; (5) the smartphone as the detector.
**Our analogue:** Paper 24 varied the **angle** at one wavelength. Our setup varies the **wavelength** (R ≈ 620–630 nm, G ≈ 520–530 nm, B ≈ 460–470 nm) at one fixed angle. Scattering and absorption both depend on wavelength, so the *pattern across R/G/B* plays the same role as their *pattern across angles*.

### Paper 23: Sweeney & Yoon (2017), *IEEE Sensors J.* 17(21):6911, angular photodiode array, wound model
- An array of photodiodes at several angles measured Mie-scatter "spectra" from porcine skin wound models infected with *S. aureus*, *E. coli* or *Salmonella*. The group's companion open-access paper (Sci. Rep. 2017, PMC5500527) used a **650 nm (red) LED**. The system detected and told infections apart in seconds, without contact.
- **What we adopt:** use the **whole multi-point pattern as a feature vector** and let a **classifier** learn it. Do not threshold a single number. That is exactly what an ML model on R/G/B features does.

### Other papers in the list (15–22): how they support us
| Ref | Idea | Relevance |
|---|---|---|
| 15 Adams 2006, 16 Hayden 2016, 18 Alsulami 2018 | Laser/forward light scatter detects bacteria by their scatter pattern | Scatter patterns carry bacterial information; justifies optical (not chemical) sensing |
| 17 You 2011, 19 Nicolini 2015, 21 Zhu 2012 | Handheld/smartphone optical detection of pathogens | Smartphone as a valid optical detector |
| 20 Min 2021 | Smartphone imaging **+ ML classifiers** for Salmonella lateral-flow | Precedent for smartphone images → ML classifier |
| 22 Dönmez 2020 | Label-free smartphone **dark-field** scattering quantifies bacteria | Dark background + scatter = our dark-box design |

---

## 3. A very important physical fact: we cannot see individual bacteria

| Object | Real size | Size in our photos |
|---|---|---|
| One bacterium (E. coli / Pseudomonas) | ~0.5–1 µm wide, 1–3 µm long | **~1/40 of one pixel** |
| One camera pixel on the meat (phone ~20 cm away, 3468 px across ~15 cm) | — | **~40 µm** |
| Micro-colony of 1000 cells | ~10–30 µm | under 1 pixel, blurred by the lens |
| Visible colony on agar (10⁷–10⁹ cells) | 1–3 mm | 25–75 pixels |
| Water droplet / wet fibre glint | 0.1–3 mm | 3–75 pixels |

**Conclusion:** most **bright spots in our photos are specular reflections** ("glints") from the wet surface, fat and muscle-fibre ridges. They are not individual bacteria or colonies. Bacteria change the image **indirectly**:
1. **Surface moisture / slime (biofilm)** changes gloss, so the number, size and shape of glints change.
2. **Myoglobin oxidation** (red → brown, metmyoglobin) changes how much **green and blue** light is absorbed. That is why separate R/G/B lights are useful.
3. **Protein breakdown and drying** change texture and scattering, so the median intensity and texture statistics change.

So we **keep counting bright spots**, as the team did originally, but we use them correctly: as **moisture/gloss features**, together with colour and texture features. A spot count alone is not a "bacteria counter". Our data confirms this (Section 5).

---

## 4. Problems in the current analysis, and the fix for each (plain language)

| # | Current practice | Why it is a problem | Our fix | Proof / reference |
|---|---|---|---|---|
| 1 | Mean of the **whole photo** | ~75–95 % of each photo is black background, so the number is mostly background and camera noise | Automatically find the meat region (ROI) and use its **median** | Paper 24 crops and uses the median; our EDA: 73–96 % of pixels are dark |
| 2 | Light label from the **time schedule only** | In EXP-1 D1A the **blue LED was off** for 24 of 44 "blue" photos, which are pure black. These gave "intensity 0" rows in the Excel. One frame shows the room with the lights on | Check the light **from the pixels** (dominant channel), drop dark and scene-change frames automatically | `02_EDA_FINDINGS_INITIAL.md` §3 |
| 3 | Normalise by the **first photo of the sample** | Works for one experiment. A real user has **one photo of unknown history**, so there is no "first photo" | Two feature sets: (a) **relative** features (vs first frame) for science/EDA, as in Paper 24; (b) **absolute, calibrated** features for prediction (a grey reference card in frame from now on) | Paper 24 normalised to a *control sample*, not to the same sample over time |
| 4 | Spot count = bacteria | Physically impossible at ~40 µm/pixel (Section 3); the trend direction flips between samples | Spots kept as gloss features plus colour/texture features; the model decides how much each matters | EDA: spot trend +0.94 in D1A, −0.94 in D1B |
| 5 | **Random** train/test split of photos | Photos 1 min apart are almost identical. A random split puts near-copies in both train and test, so the model "remembers" the piece and scores 99 % that is **fake** | Split by **chicken piece / session** (GroupKFold, leave-one-session-out). Test pieces are never seen in training | Kaufman et al. 2012 "Leakage in data mining", ACM TKDD; scikit-learn `GroupKFold` docs |
| 6 | "Elapsed time" = age | The piece was already some hours/days old when filming started, and room temperature varies | Label = **hours since purchase/start** (from metadata), plus a **temperature-adjusted age** (see §6) | Ratkowsky square-root model for poultry spoilage (Ratkowsky 1982; Gill et al. 1984 tested on 14 poultry studies) |
| 7 | Whole-frame crop coordinates copied from image 1 | The camera shifted by up to ~60 px in one session | Per-frame automatic ROI plus a drift check | EDA: drift of −4/+7 px at 1/8 scale in D1A |

---

## 5. Evidence from our own data (first pass, 944 photos)

- **5 sessions**, 3 pieces visible per shape (EXP-1: D1A, D1B; EXP-2: D2A, D2B, D3A). Sessions last 1.9–5.7 h with 1 photo/min and a strict R→G→B cycle.
- **The camera was in manual mode:** ISO 100, 1/60 s, f/1.7, identical in every photo. White balance was on **auto**, which should be fixed to manual.
- Whole-frame brightness **decreased** over time in 4 of 5 sessions (Spearman ρ ≈ −0.9 to −1.0), consistent with surface drying. D2A went **up**.
- Bright-spot count trend **changed sign** between sessions (D1A ρ=+0.94, D1B ρ=−0.94).
→ No single number tracks age reliably. We need a **multi-feature model** and **honest, piece-grouped evaluation**.

---

## 6. Handling "age" and temperature (the label problem)

- **What we can know:** time since the experiment started (EXIF timestamp), the storage day (D1/D2/D3, *to be confirmed by the data team*), and the air temperature.
- **Temperature source (configurable per sample):**
  1. `weather`: hourly outdoor temperature for PEC Chandigarh (30.765 N, 76.786 E) from the free **Open-Meteo archive API**, cached to disk. It is an approximation; indoor temperature is usually 1–4 °C different.
  2. `manual`: one temperature typed in for the whole sample.
  3. `log`: a CSV from a thermometer/logger (best for new data).
- **Temperature-adjusted age** (an "equivalent hours at 25 °C"):
  `equiv_hours = Σ ((T_i − T_min) / (25 − T_min))² · Δt_i`, with `T_min ≈ −8 °C` (configurable; verify in Phase 5).
  This is the Ratkowsky square-root growth model used in predictive microbiology for poultry. One hour at 33 °C counts as about 1.5 equivalent hours at 25 °C; one hour at 4 °C counts as about 0.13 h. This lets us compare pieces kept at different temperatures fairly.
- **Unknown age before filming:** we predict **age since the reference event recorded in metadata** (purchase or start). We say this clearly in the report and the app. We cannot honestly predict "time since slaughter" without that information.

---

## 7. About the ">94 % accuracy" target (please read)

- With **3–5 independent chicken pieces**, any score is uncertain. A model can hit 99 % on a random split and still fail on a new piece.
- **Our promise:** we report **piece-grouped** results with **95 % confidence intervals**, compared against a "dumb baseline", together with a **permutation test** (is the score better than chance?).
- We aim for **≥94 % balanced accuracy on the coarse freshness class under grouped evaluation**. If the data does not support it, we report the true number and **what data would fix it** (more pieces, a grey card, a thermometer log). An honest 80 % beats a fake 99 %, and examiners in ML check for exactly this.
- The report includes a side-by-side "random split vs grouped split" table. It **shows** the mentor why the grouped number is the real one.

---

## 8. What changes for the mentor's plan? (Nothing major)

| Mentor's plan | Kept? | Our refinement |
|---|---|---|
| Smartphone camera + R/G/B LED illumination | ✅ | Add pixel-based light verification |
| Intensity of R/G/B images over time | ✅ | ROI median instead of whole-frame mean |
| Normalise to the first reading | ✅ (for EDA/science) | Add calibrated absolute features for prediction |
| Count bright spots | ✅ | Interpreted as moisture/gloss features; specular vs diffuse separated |
| ML model to predict age | ✅ | Grouped evaluation, temperature-aware labels, confidence and "cannot judge" output |

---

## 9. References
1. Liang P-S, Park TS, Yoon J-Y (2014) Rapid and reagentless detection of microbial contamination within meat utilizing a smartphone-based biosensor. *Sci Rep* 4:5953. doi:10.1038/srep05953
2. Sweeney RE, Yoon J-Y (2017) Angular photodiode array-based device to detect bacterial pathogens in a wound model. *IEEE Sens J* 17(21):6911–6917. doi:10.1109/JSEN.2017.2752155 — https://experts.arizona.edu/en/publications/angular-photodiode-array-based-device-to-detect-bacterial-pathoge/
3. Companion open-access paper (650 nm LED, skin models): https://pmc.ncbi.nlm.nih.gov/articles/PMC5500527
4. Ratkowsky DA et al. (1982) Relationship between temperature and growth rate of bacterial cultures. *J Bacteriol* 149:1–5.
5. Gill/Olley-group evaluation of the square-root model on 28 poultry spoilage datasets (−2 to 25 °C), *Food Microbiology* 1984 — https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_proquest_miscellaneous_13901465
6. Chicken spoilage threshold ≈ 7 log CFU/g, staging (≤3, 3–6.9, >7 log): see the review list in https://pmc.ncbi.nlm.nih.gov/articles/PMC9043016 and https://uhra.herts.ac.uk/id/eprint/10387/1/sensors-23-04233.pdf (CNN meat quality on embedded hardware, *Sensors* 2023)
7. Low-cost multi-waveband LED imaging of meat quality: https://agris.fao.org/search/zh/records/64747d4b2d3f560f80afa255
8. Smartphone multispectral imager for packaged meat (USDA project): https://portal.nifa.usda.gov/web/crisprojectpages/1029880-smartphone-integrated-multispectral-image-sensor-for-packaged-meat-safety.html
9. Kaufman S, Rosset S, Perlich C (2012) Leakage in data mining. *ACM TKDD* 6(4).
10. scikit-learn: `GroupKFold`, `LeaveOneGroupOut`: https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
11. Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
12. References 15–22 as listed by the mentor (see `README.md` §References).

> Agents: references 4, 5 and 9 are well-known publications quoted from memory. Verify the exact page numbers before the final report (Phase 8).
