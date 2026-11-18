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
# # 03 · Exploratory data analysis: how does the meat change under red, green and blue light?
#
# **Purpose.** Before any model, look at how each photo measurement changes over time in every
# session and under each light, and which light carries the most consistent signal.
#
# **Inputs.** `data/processed/frame_stats.parquet` (`foodlens eda-stats`, one row per usable
# photo) and the tables / figures written by `foodlens eda` (`reports/tables/eda_*.csv`,
# `feature_signal_ranking.csv`, `reports/figures/F5-F8*`).
#
# **Outputs.** None by default (it reads the stage outputs). Set `RECOMPUTE = True` to rerun
# `foodlens eda` from here (~30 s).
#
# **Run time.** About 10 s.
#
# **Words used.** *ROI* = the well-lit meat area measured in every photo of a session (same
# pixels for R, G and B). *Glints* = tiny bright specular reflections of the wet surface
# (moisture), **not bacteria**. *Spearman rho* = how steadily a value rises (+1) or falls (-1)
# with time. *Session* = one piece filmed for 2-6 h.

# %%
import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
from IPython.display import Image, display  # noqa: E402

from foodlens.config import load_config  # noqa: E402
from foodlens.eda.run import frame_stats_path, run_eda  # noqa: E402

RECOMPUTE = False
cfg = load_config()
tables = cfg.path("reports_dir") / "tables"
figures = cfg.path("reports_dir") / "figures"
HAVE_DATA = frame_stats_path(cfg).is_file() and (tables / "eda_trends.csv").is_file()
if RECOMPUTE and frame_stats_path(cfg).is_file():
    print(run_eda(cfg))
if HAVE_DATA:
    stats = pd.read_parquet(frame_stats_path(cfg))
    print(f"{len(stats)} usable photos, {stats['session_id'].nunique()} sessions")
    print(stats.groupby(["session_id", "light"]).size().unstack())
else:
    print("Missing inputs: run `uv run foodlens eda-stats && uv run foodlens eda` first.")


def show(name: str, width: int = 1100) -> None:
    """Display a saved figure, downscaled to ``width`` px so the notebook stays small (the full
    resolution file is in reports/figures)."""
    from io import BytesIO

    from PIL import Image as PILImage

    f = figures / name
    if not f.is_file():
        print(f"missing figure {f.name}")
        return
    with PILImage.open(f) as im:
        im.thumbnail((width, 4 * width))
        buf = BytesIO()
        im.convert("RGB").save(buf, "PNG", optimize=True)
    display(Image(data=buf.getvalue()))


def table(name: str) -> pd.DataFrame:
    """Read a saved table (empty if missing)."""
    f = tables / name
    return pd.read_csv(f) if f.is_file() else pd.DataFrame()


# %% [markdown]
# ## 1. Which pixels are measured? (open question 6)
# Yellow = the lit ROI used for all features. Cyan = the whole meat including its shadowed side.
# The shadowed side is measured separately (`shd_*` columns) so we can see whether it adds
# anything. It is very dim (often 4-18 grey levels out of 255), so its values are noisy.

# %%
show("qa_zones.png")
display(table("eda_zones.csv"))

# %% [markdown]
# ## 2. Start, middle and end of each session (F6)
# The three photos of one light share **one** brightness stretch, so if the middle or end looks
# darker, the meat really is darker. The last panel is end minus start: blue = darker now.

# %%
for sid in sorted(stats["session_id"].unique()) if HAVE_DATA else []:
    show(f"F6_start_mid_end_{sid}.png", 900)
cols = ["session_id", "light", "roi_median_pct_change", "glint_count_pct_change"]
cols += ["rglint_count_pct_change", "glcm_contrast_pct_change", "step_between"]
display(table("eda_start_mid_end.csv")[cols].round(1))

# %% [markdown]
# ## 3. Time series (F5)
# Dots are single photos, the line is a rolling median. In `exp2_D2A` the camera's colour
# balance jumped at about 1.9 h (dashed line); brightness values are corrected for that jump,
# glint values are not (treat D2A glints after the jump with care).
# The pooled figure puts all sessions on one scale: every series starts at 1.0.

# %%
show("F5_relative_all_sessions.png")
show("F5_timeseries_exp2_D2A.png")

# %% [markdown]
# ### Trend statistics
# Per session x light x feature: Spearman rho, its p-value and the Theil-Sen slope (a robust
# straight-line slope) in % of the start value per hour. Photos minutes apart are not
# independent, so single-session p-values are optimistic; the pooled model below and the
# agreement across sessions (section 5) are the honest summaries.

# %%
tr = table("eda_trends.csv")
key = ["roi_mean", "roi_mean_rel_ref", "ref_mean", "rglint_density", "glint_density"]
view = tr[(tr["variant"] == "step_corrected") & tr["feature"].isin(key)]
display(view.pivot_table(index=["feature", "light"], columns="session_id", values="rho").round(2))

# %% [markdown]
# ### Pooled mixed model: does it change the same way in every piece?
# `value ~ hours` with a separate starting level per session. On the "relative to start" scale
# the slope reads as "fraction of the starting value lost (-) or gained (+) per hour".

# %%
mm = table("eda_mixed_models.csv")
display(mm[mm["scale"] == "relative to start"].drop(columns=["scale"]).round(4))

# %% [markdown]
# ### Control: is it the meat or the lamp?
# If the LEDs dimmed while warming up, everything in the box would get darker. `ref_mean` is the
# brightness of the lit tray / holder / walls (not meat). It stays flat or slightly rises while
# the meat darkens, so the darkening belongs to the meat. `roi_mean_rel_ref` (meat / surroundings)
# cancels lamp and camera gain changes.

# %%
display(
    view[view["feature"].isin(["ref_mean", "roi_mean"])]
    .pivot_table(index=["feature", "light"], columns="session_id", values="slope_pct_h")
    .round(1)
)

# %% [markdown]
# ## 4. Glint dynamics (F7)
# Glints in consecutive photos of the same light are matched: born, died, merged, split or
# persisted. "Evaporation" = fewer and smaller glints; "film breaking" = more and smaller;
# "merging" = fewer and bigger. Two glint rules are compared: `abs` (fixed brightness floor 40,
# the P2 rule) and `rel` (at least 1.75 x the surrounding meat, the same under every light).

# %%
show("F7_glint_dynamics_rel.png")
dyn = table("eda_glint_dynamics.csv")
display(
    dyn.pivot_table(
        index=["session_id", "light"], columns="variant", values="pattern", aggfunc="first"
    )
)

# %% [markdown]
# ## 5. Which light carries the most information? (F8)
# For each feature and light: the mean rho over sessions and how many of the 5 sessions agree on
# the direction. A feature is **promising** only if at least 4 of 5 sessions agree.
# `TRIPLET` = ratios between the red, green and blue photos of one 3-minute cycle.

# %%
show("F8_signal_heatmap.png", 800)
display(table("eda_light_comparison.csv").round(2))
display(table("feature_signal_ranking.csv").head(20).round(3))

# %% [markdown]
# ## What this means for modelling (details in `docs/07_EDA_FINDINGS.md`)
# - Brightness of the meat falls steadily in all sessions and all lights; it is the strongest,
#   most consistent signal, and it is not a lamp effect (control above).
# - Absolute values differ a lot between sessions (different pieces, positions, white balance),
#   so a single photo must be judged with ratios / normalised features; P5/P6 must test this
#   with leave-one-session-out evaluation.
# - Glint counts depend on the rule; the relative rule gives a more consistent RED trend.
# - With only 5 sessions, every "consistent in 5/5" claim is still weak evidence.
