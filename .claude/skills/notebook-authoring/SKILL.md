---
name: notebook-authoring
description: How FoodLens notebooks are written (jupytext percent .py sources) and generated/executed into .ipynb with explanations.
---
# Notebook authoring

- Sources: `notebooks/src/NN_<name>.py` in jupytext percent format. Markdown cells `# %% [markdown]` explain WHY in simple language (audience: a mentor new to ML). Code cells `# %%` call functions from `foodlens`; no long logic inside notebooks (max ~25 lines per cell).
- Each notebook starts with: title, purpose, inputs (files), outputs (files), how long it takes.
- Each notebook must run top to bottom standalone: `bash scripts/build_notebooks.sh` converts (jupytext --to ipynb) and executes (`jupyter nbconvert --execute` or nbclient) with a timeout from config. A failure = check fails.
- The master notebook `notebooks/src/00_FoodLens_end_to_end.py` stitches the stages with `--limit`-style small settings so it runs in < 10 min; full runs use the CLI.
- Figures are saved via `foodlens.viz.save_fig(name)` to `reports/figures/` (PNG 300 dpi + SVG) AND shown inline.
- Never hand-edit `.ipynb`; never commit executed outputs larger than 5 MB (strip with `nbstripout` if needed; config decides).
