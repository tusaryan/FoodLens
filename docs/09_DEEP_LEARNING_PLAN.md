# 09 · Deep-learning (CNN) track: options, choice and honest expectations (S10)

**For:** the user, the mentor and the next agent sessions. **Decision (user, S10):** a
pretrained vision network fine-tuned on our photos (PyTorch + `timm`) becomes the **main model
track**. The P6 classical models stay as the baselines it must beat on the **same folds**.

## 1. Why a CNN is the right thing to try
- This is a computer-vision problem. A CNN learns its own image patterns (glint shape and spread,
  fibre texture, colour patches, where on the meat the change happens) instead of the 153 numbers
  we designed by hand. A pretrained network already knows edges, textures and colours from
  millions of photos, so it needs far fewer of our photos than a network trained from zero.
- Published meat / food freshness work mostly uses pretrained CNNs (VGG, ResNet, EfficientNet,
  MobileNet) with transfer learning ([JWPR 2026, chicken: healthy / defrosted / rotten][s1];
  [arXiv 2603.00368, smartphone + OOD-aware classifier][s2]).

## 2. What a CNN cannot fix: the data (please read)
| Fact (from our data) | What it means for any model, CNN included |
|---|---|
| **9 independent pieces** (exp3_D1A excluded) | The test is "a piece the model never saw". 9 pieces = only 9 test cases for "new piece" behaviour; CIs stay wide (±0.15 balanced accuracy). |
| Only **2 pieces** (exp4, exp5) reach `spoiled_risk`; `quality_loss` is ~90 % from the same 2 | When exp4 is the test piece, the model learns "spoiled" from ONE piece. |
| exp4/exp5 were filmed with a **different framing** (meat crop ~190 px vs ~450 px, fully lit) | A CNN can learn "small, softer picture = old meat" (shortcut learning). Section 4 lists the defences. |
| 1258 triplets are **near-duplicates 3 min apart** | Effective sample size is closer to the number of pieces than to 1258. A CNN can memorise each piece. |
| S08 leakage demo: random photo split **0.98** balanced acc. vs **0.35** grouped | With a random photo split any model looks ~98 % accurate because near-copies of the test photo are in training (`reports/tables/leakage_demo.csv`). |

**Question for the user:** in your earlier test where "deep learning beat ML", were the train and
test photos split **randomly** (or by folder of all photos mixed together)? If yes, that score was
inflated by the same leakage, and it does not tell us which model generalises to a new piece. We
will answer the question properly: CNN vs classical, **leave-one-piece-out, same 9 folds, paired
per piece**.

Paper 24 (Liang, Park & Yoon 2014), which the mentor's method follows, uses **no machine learning
and no CNN**: the median grey level of a cropped region, divided by a clean control, with t-tests.
So the CNN is our extension, not part of the reference method.

**What helps most** (more than any architecture): (1) at least 3 more long sessions with the SAME
setup (user is asking the mentor); more independent pieces beat more hours per piece; (2) a
grey / white reference card in every photo (session-relative features raised P6 balanced accuracy from 0.31 to
0.51: the card gives a single photo the same kind of reference); (3) manual white balance.

## 3. Options considered (simple language)
| Option | What it is | Fit for 9 pieces | Verdict |
|---|---|---|---|
| A. Frozen embeddings + small head | Run a pretrained net once, keep its 384–1024 numbers per photo, fit logistic/ridge on them | Very safe: nothing in the net is trained, cheap, fits our existing CV code | **Baseline phase P7B** (DINOv2 ViT-S/14, ConvNeXt-T, EfficientNet-B0, MobileNetV3-L, optional CLIP ViT-B/16) |
| B. Partial fine-tuning (last blocks) + heavy augmentation | Retrain only the top of the net on our photos | Adapts to meat/LED images while most weights stay fixed; the usual best choice for small data | **Main model (P7C)** |
| C. Full fine-tuning | Retrain every layer | ~5–28 M weights for ~9 independent pieces: memorises pieces | Rejected (can be a one-line config ablation later) |
| D. Siamese / reference-pair | Same net sees the current photo AND a reference (first photo of the piece, later a grey card); the head uses the difference | Matches our best finding (relative features 0.51) but needs a reference photo, so it is **not single-shot** | **Ablation 2** (reported next to the mentor first-frame baseline) |
| E. 3-light input | (i) composite: R channel of the RED photo, G of GREEN, B of BLUE stacked into one 3-channel image; (ii) one shared backbone per light + fusion | (i) maps exactly onto the pretrained RGB filters (each LED photo is ~monochrome in its own channel, checked S10), 1 forward pass, keeps pixel-level colour ratios; (ii) 3× compute | **(i) in the main model**; (ii) used for the per-light comparison (frozen, P7B) |
| F. Multi-task / ordinal | One head for equivalent hours, one ordinal head for the 4 levels (fresh < use_soon < quality_loss < spoiled_risk) | The levels are ordered; ordinal loss penalises "fresh vs spoiled" more than neighbour mistakes | **In the main model** |
| G. Sequence model (several photos in a row) | LSTM / temporal CNN over consecutive photos | Photos 3 min apart barely change; needs the history | Not planned |

## 4. Recommended main model (pre-registered in S10, copied to config in P7A before any training)
**"Composite-triplet EfficientNet-B0, last stages fine-tuned, multi-task ordinal + hours."**
1. **Input:** the three meat-only crops of one triplet (`foodlens crops`, background = 0, same
   bbox for R/G/B) → composite image (R from RED, G from GREEN, B from BLUE).
2. **Scale normalisation:** resize each composite so the meat area is the same number of pixels
   (√meat_px = `dl.meat_side_px`), with antialiasing. This removes the EXP-2 vs EXP-3/4/5
   framing difference as far as possible (pieces are of similar physical size).
3. **Interior patches:** train on random square patches lying ≥ 95 % inside the meat mask; test on
   a fixed, seeded grid of patches averaged. The network then cannot use the meat outline
   (silhouette = piece identity) or the image size (= setup).
4. **Backbone:** `efficientnet_b0` (ImageNet, timm). Small (5.3 M), fast on the M1 GPU (MPS),
   well proven on food images, exports to ONNX / TFLite for the optional Android app. BatchNorm
   statistics frozen (small batches, few pieces). Only the last 2 stages + head train.
5. **Heads and loss:** ordinal (cumulative logits, 3 thresholds) for the level + Huber on equivalent
   hours; class and piece weights computed **on the training pieces only**, so the 450-triplet
   overnight piece does not dominate.
6. **Augmentation (fixed rules, fitted on nothing, training only):** flips, 90° rotations, patch
   position, small rotation, mild blur (blunts sharpness/scale cues), a small equal gain on all
   three channels. **No hue/colour jitter** (colour change IS the signal).
7. **Early stopping without peeking:** for each outer test piece, inner grouped folds over the 8
   training pieces pick the number of epochs; the model is then retrained on all 8 pieces for that
   many epochs. The test piece's loss is never looked at.
8. **Reported like P6:** LOGO piece (headline) and leave-one-experiment-out, group-bootstrap CI,
   grouped permutation test, paired per-piece comparison vs P6 linear and the baselines, 3 seeds.

**Ablations (only two):** (1) **light:** single-light inputs (RED, GREEN, BLUE alone) vs the
composite, the project's core question asked again for the CNN; (2) **reference-pair**
(Siamese, option D) with the first photo of the piece as the reference.

**Extra honesty checks (CNN specific):** (a) **within-piece rank correlation** (Spearman ρ between
predicted and true hours inside each held-out piece): framing is constant within a piece, so this
cannot come from the setup shortcut; (b) EXP-2-only LOGO (7 pieces, one framing); (c) Grad-CAM
maps on held-out pieces: does the net look at glints, fibre texture, colour patches, or edges?

## 5. What we expect (said before any result)
Realistic: the CNN may reach roughly the P6 range (0.3–0.5 single-shot) and should beat the
baselines on age error; a large jump (> 0.7 on new pieces) is unlikely with 9 pieces and would be
investigated for leakage first (AGENTS.md red flags). The reference-pair model may do better
because it removes the piece-to-piece offset. Whatever the number, the report shows CNN vs
classical on identical folds, and the random-split number only as the leakage demo.

## 6. Sources
- [s1]: https://jwpr.science-line.com/attachments/article/88/JWPR16(1)31-41,2026.pdf (CNN transfer learning, sliced chicken)
- [s2]: https://arxiv.org/abs/2603.00368 (meat freshness, segmentation + OOD-aware classifier, smartphone)
- DINOv2 frozen features (linear probe as the main evaluation): https://arxiv.org/abs/2304.07193
- Natural-image foundation models on small medical sets (frozen probe vs fine-tuning): https://arxiv.org/abs/2310.19522
- Extra input channels in pretrained CNNs (initialising/adapting the first conv): https://arxiv.org/abs/2309.12463
- MeatScan dataset paper (controlled-setting meat images do not represent other conditions): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12546412/
- Quoted from memory, verify in P8: Tan & Le 2019 (EfficientNet, ICML); Liu et al. 2022 (ConvNeXt, CVPR);
  Cao, Mirjalili & Raschka 2020 (CORAL ordinal regression, Pattern Recognit. Lett.); Selvaraju et al. 2017
  (Grad-CAM, ICCV); Geirhos et al. 2020 (shortcut learning, Nat. Mach. Intell.); Kornblith et al. 2019
  (do better ImageNet models transfer better?, CVPR); Kaufman et al. 2012 (leakage, ACM TKDD).

[s1]: https://jwpr.science-line.com/attachments/article/88/JWPR16(1)31-41,2026.pdf
[s2]: https://arxiv.org/abs/2603.00368
