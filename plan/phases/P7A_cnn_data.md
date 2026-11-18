# P7A: CNN setup, triplet-composite dataset and leak-free augmentation (1 session)

**Goal:** everything a CNN needs before any training: PyTorch + timm in `.venv`, a deterministic
device/seed helper, a dataset that turns the existing meat-only crops into scale-normalised
composite triplets and interior patches, augmentation that learns nothing from data, the
pre-registered CNN settings in config, and tests that prove no test-piece pixel can reach training.
Background and the choice of design: `docs/09_DEEP_LEARNING_PLAN.md` (read §2 and §4).
**Skills to load:** `session-protocol`, `ml-evaluation`, `long-running-commands`, `image-pipeline`.

## P7A.T1: Dependencies and device
- `uv add torch torchvision timm` (note why in the session log; torch is ~1 GB in `.venv`). A
  global torch 2.12 exists in pyenv (LESSONS): never use it, always `uv run`.
- Weights download once (timm → `~/.cache/huggingface/hub`, torchvision → `~/.cache/torch/hub`).
  Add both folders to `scripts/uninstall_global_tools.sh` and the README cleanup section.
- `dl/device.py`: `pick_device(cfg)` (`mps` if available else `cpu`, overridable in config) and
  `seed_everything(seed)` (python, numpy, torch, MPS; `torch.use_deterministic_algorithms(True,
  warn_only=True)`); record torch/timm versions + device in every RunReport.
- **Accept:** `uv run python -c "import torch, timm; print(torch.backends.mps.is_available())"`;
  check.sh still < 60 s for pytest (tests use CPU and tiny models).

## P7A.T2: Pre-registered CNN config (`configs/default.yaml` → `dl:`, pydantic `extra="forbid"`)
Copy the main-model spec from docs/09 §4 into config NOW, before any CNN result exists:
backbone `efficientnet_b0`, trainable stages 2, frozen BN, `meat_side_px` (scale normalisation),
`patch_px`, `input_px`, `min_patch_meat_frac` 0.95, test-grid size, batch size, epochs max,
optimiser (AdamW, lr head / backbone, weight decay), Huber weight, inner folds for epoch choice,
seeds `[0, 1, 2]`, augmentation switches and ranges, and the **selection rule** (P7D.T4). Comment
every value. The frozen-backbone list for P7B goes here too.
- **Accept:** a config test that loads `dl:` and rejects an unknown key.

## P7A.T3: Composite triplet dataset (`dl/composite.py`, `dl/dataset.py`)
- `build_composite(red, green, blue, masks) -> (img HxWx3 uint8, mask)`: R channel of the RED
  crop, G of GREEN, B of BLUE (use the **verified** light from the manifest, never the schedule).
  Triplets missing a light are skipped and counted (log the reason).
- `scale_normalise(img, mask, meat_side_px)`: resize so √(mask area) = `meat_side_px`, antialiased;
  log the scale factor per frame (a later check shows it no longer separates EXP-2 from EXP-3/4/5).
- `sample_patches(img, mask, n, rng)` (training: random, ≥ `min_patch_meat_frac` meat) and
  `grid_patches(img, mask)` (evaluation: fixed seeded grid). Pixels outside the mask stay 0.
- `TripletPatchDataset(rows, cfg, train: bool)` (torch Dataset) built from a list of
  `triplet_id`s **passed in by the CV loop**; it never reads labels of rows it was not given.
  Optional cache of composites at `data/interim/dl/<hash>/` (`foodlens dl-data`, `STAGE DONE`).
- Per-light mode (for the light ablation): one LED's channel replicated to 3 channels.
- **Tests (synthetic images):** channel mapping is correct; scale normalisation gives the target
  meat area ±2 %; every training patch meets the meat fraction; grid patches are deterministic;
  a missing light raises/skips with a code; a dataset built from fold rows contains no row of the
  held-out group.

## P7A.T4: Augmentation that cannot leak (`dl/augment.py`)
Fixed rules, fitted on nothing, applied to **training patches only**: h/v flips, 90° rotations,
small rotation, mild Gaussian blur (σ range from config), one gain factor applied equally to all
three channels (range from config). **No** hue/saturation/colour jitter, no per-image
standardisation (absolute colour change is the signal), no mixup across pieces. Normalisation =
the backbone's fixed ImageNet mean/std (constants, not fitted).
- **Tests:** the eval transform is deterministic; train transforms keep channel order; the gain
  multiplies R, G and B equally; nothing in the module has a `fit`.

## P7A.T5: Visual check and setup probe
- `foodlens dl-data --limit 20`, then full (expect ~1258 composites). Render a varied sample
  (`eda-sample`-style, NEW random + outlier frames; LESSONS): composite + mask + 6 training
  patches for ≥ 2 EXP-2 and ≥ 2 EXP-3/4/5 pieces → `reports/figures/dl_composites_S<NN>.png`.
  Inspect it (colours plausible, no tray/holder in patches, similar apparent scale across setups).
- Table `reports/tables/dl_scale_check.csv`: per session meat_px before/after normalisation.
- **Accept:** composites for all complete triplets or a logged reason; figure inspected and
  described in the session log; check.sh PASS.
