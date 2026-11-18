# P7: Deep (CNN) features vs classical (1–2 sessions). Recommended, not strictly required.

**Goal:** test whether pretrained image embeddings beat hand-crafted features under the **same** grouped evaluation. With ~5 groups, full fine-tuning would overfit, so use **frozen embeddings + a small head**.
**Skills to load:** `session-protocol`, `ml-evaluation`, `long-running-commands`.

## P7.T1: Dependencies
`uv add torch torchvision timm`. The uv cache hard-links, but torch is still ~1 GB in `.venv`; note it in the README cleanup section. Use `device = "mps" if torch.backends.mps.is_available() else "cpu"`. Weights download once to `~/.cache/torch/hub` or `~/.cache/huggingface`: list them in `scripts/uninstall_global_tools.sh`.

## P7.T2: Embeddings (`models/embeddings.py`)
- Input: the ROI crop of the verified channel replicated to 3 channels (or the original RGB crop: test both), resized to 224, ImageNet normalisation.
- Backbones: `mobilenetv3_large_100` (default), `efficientnet_b0`; optional `vit_small_patch14_dinov2` if it fits in time.
- **Tiling:** 5 random-but-seeded 224 crops inside the ROI + 1 whole-ROI resize → mean-pooled embedding (more robust to glint location).
- Cache to `data/processed/emb_<backbone>_<hash>.parquet` (frame level) → triplet concat.
- **Tests:** a deterministic output shape on a synthetic image (CPU, tiny model, or mock).

## P7.T3: Heads and comparison
- PCA (fit in-fold, 32 comps) → Logistic/Ridge; and the embedding + hand-crafted concat.
- The exact same splits, metrics and RunReport as P6. Add rows to `model_comparison.csv`.
- Augmentation is not needed for frozen features. If the user explicitly asks for fine-tuning: last block only, ≤ 10 epochs, inner grouped early stopping. Document the risk.

## P7.T4: Final selection
Re-apply the P6.T5 rule over all models. If a CNN-based model wins, export it (joblib for the head + backbone name/hash; optional ONNX for apps) and update the model card. Otherwise record that classical won (a valid, explainable result).
