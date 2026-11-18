# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 · Data inventory and frame quality control
#
# **Purpose.** Show what raw data we have, how it was captured, which frames are unusable and
# why, and the room temperature during each session.
#
# **Inputs.** `data/processed/manifest.parquet` (from `foodlens ingest` then `foodlens qc`),
# `reports/tables/inventory.csv`, `reports/tables/qc_summary.csv`, `reports/qc_rejects.csv`.
#
# **Outputs.** `reports/figures/qc_examples.png`, `reports/figures/temperature_sessions.png`.
#
# **Run time.** About 10 s (it only reads the tables and a few photos). If the raw data or the
# manifest is missing (e.g. a fresh clone), the notebook says so and stops showing results.

# %% [markdown]
# ## How the photos were taken
# A piece of raw chicken sits in a dark box. A phone takes one photo every ~60 s while the box
# cycles through a **red, green and blue LED** (R, G, B, R, G, B, ...). Camera settings are fixed
# (manual ISO and shutter) but white balance is automatic. Each folder (e.g. `exp1_D1A`) is one
# **session**: one piece filmed continuously for 2-6 hours.
#
# Every photo becomes one row of the **manifest** table. We never edit the original photos.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from IPython.display import Image, display  # noqa: E402

from foodlens.config import load_config  # noqa: E402
from foodlens.stages import manifest_path  # noqa: E402

cfg = load_config()
reports = cfg.path("reports_dir")
figures = reports / "figures"
HAVE_DATA = manifest_path(cfg).is_file() and cfg.path("raw_dir").is_dir()
if HAVE_DATA:
    man = pd.read_parquet(manifest_path(cfg))
    print(f"manifest: {len(man)} frames, {man['session_id'].nunique()} sessions")
else:
    print("No manifest/raw data: run `uv run foodlens ingest && uv run foodlens qc` first.")

# %% [markdown]
# ## Inventory per session
# Start/end time, duration, number of frames per **scheduled** light and per light **measured
# from the pixels**, the longest gap between photos, and the camera settings (one unique value
# each means the settings never changed).

# %%
if HAVE_DATA:
    display(pd.read_csv(reports / "tables" / "inventory.csv").T)

# %% [markdown]
# ## Quality control (QC)
# We do not trust the schedule: each frame's light is measured from its pixels.
# - **E_DARK_FRAME**: the photo is black (the LED was off). In `exp1_D1A` the blue LED did not
#   work until about 13:34, so 24 blue-slot photos are black.
# - **E_MIXED_LIGHT**: no single colour dominates, e.g. the room light was switched on
#   (the last `exp1_D1A` photo at 14:30).
# - **W_BLUR** (warning only): far less sharp than the other photos of the same session and
#   light, e.g. the camera refocused.
#
# Rejected frames are kept in the manifest (with the reason) but not used for training.

# %%
if HAVE_DATA:
    display(pd.read_csv(reports / "tables" / "qc_summary.csv"))

# %% [markdown]
# ### One example per reason (contrast-stretched so the dark photos are visible)

# %%
if HAVE_DATA:
    from foodlens.viz import show_examples

    flagged = man[man["qc_reasons"].fillna("") != ""]
    first = flagged.assign(reason=flagged["qc_reasons"].str.split(";").str[0])
    picks = first.groupby("reason").head(1).sort_values("reason")
    picks = picks[picks["path"].map(lambda p: (cfg.root / p).is_file())]
    fig = show_examples(
        [cfg.root / p for p in picks["path"]],
        [f"{r}\n{f}" for r, f in zip(picks["reason"], picks["frame_id"], strict=True)],
        cfg.vision.work_long_side // 2,
        "qc_examples",
        figures,
    )
    plt.close(fig)
    display(Image(filename=str(figures / "qc_examples.png"), width=900))

# %% [markdown]
# ## Room temperature
# Bacteria grow faster when it is warmer, so temperature matters for the age label later (P5).
# We did not log temperature, so we use the hourly **weather archive** (Open-Meteo) for the
# capture location, interpolated to each photo's time. This is outdoor air temperature; the
# config value `thermal.indoor_offset_c` can correct it if indoor readings become available.

# %%
if HAVE_DATA:
    print(man.groupby("session_id")["temp_c"].agg(["min", "max", "mean"]).round(1))
    display(Image(filename=str(figures / "temperature_sessions.png"), width=700))

# %% [markdown]
# ## Take-aways
# - Use only frames with `qc_status` in {ok, warn}; the light column is the measured light.
# - Scene-change detection (moved sample, lights on) is finished in P2 once the meat region is
#   segmented; QC is re-run then.
