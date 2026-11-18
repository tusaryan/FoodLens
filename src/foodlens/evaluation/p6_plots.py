"""P6 figures: F12 light ablation, F13 reliability diagram, F14 permutation importance,
F15 partial dependence. Inputs are the tables written by ``models.train`` / ``models.finalize``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from foodlens.viz import save_fig  # noqa: E402

LIGHT_COLORS = {"RED": "#c0392b", "GREEN": "#27ae60", "BLUE": "#2e64c8", "RG": "#b08d22",
                "RB": "#8e44ad", "GB": "#16a085", "RGB_all": "#555555"}  # fmt: skip


def light_ablation_figure(table: pd.DataFrame, chance: float, baselines: dict,
                          figures_dir: Path) -> Path:  # fmt: skip
    """F12: per target (column) and model family (x groups), one bar per light set + 95 % CI.
    ``baselines`` = {target: (name, value)} drawn as a dashed line; chance on the class panel."""
    targets = list(dict.fromkeys(table["target"]))
    fams = list(dict.fromkeys(table["family"]))
    sets = list(dict.fromkeys(table["feature_set"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.0))
    w = 0.8 / len(sets)
    for ax, target in zip(np.atleast_1d(axes), targets, strict=True):
        t = table[table["target"] == target]
        for i, fs in enumerate(sets):
            r = t[t["feature_set"] == fs].set_index("family").reindex(fams)
            err = np.vstack([r["value"] - r["ci_lo"], r["ci_hi"] - r["value"]])
            ax.bar(np.arange(len(fams)) + (i - len(sets) / 2 + 0.5) * w, r["value"], w,
                   yerr=err, capsize=2, color=LIGHT_COLORS.get(fs, "#999"), label=fs,
                   error_kw={"lw": 0.7})  # fmt: skip
        if target in baselines:
            name, val = baselines[target]
            ax.axhline(val, ls="--", c="k", lw=1, label=f"{name} = {val:.2f}")
        if t["metric"].iloc[0] == "balanced_accuracy":
            ax.axhline(chance, ls=":", c="grey", lw=1, label=f"chance = {chance:.2f}")
        ax.set_xticks(np.arange(len(fams)), fams)
        mae = t["metric"].iloc[0] == "mae"
        unit = " (equiv. h, lower = better)" if mae else " (higher = better)"
        ax.set_title(f"{target}: {t['metric'].iloc[0]}{unit}", fontsize=9)
        ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("F12 light ablation, leave-one-piece-out (9 pieces), 95 % group-bootstrap CI",
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    return save_fig(fig, "F12_light_ablation", figures_dir)


def reliability_figure(rel: dict[str, pd.DataFrame], ece: dict[str, float],
                       figures_dir: Path) -> Path:  # fmt: skip
    """F13: top-label reliability diagram, uncalibrated vs calibrated (marker size = count)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")
    for (kind, r), c in zip(rel.items(), ("#c0504d", "#3a6ea5"), strict=False):
        ax.plot(r["confidence"], r["accuracy"], "-o", c=c, ms=3,
                label=f"{kind} (ECE {ece[kind]:.3f})")  # fmt: skip
        ax.scatter(r["confidence"], r["accuracy"], s=r["n"] / r["n"].max() * 120, c=c, alpha=0.3)
    ax.set(xlabel="predicted confidence (top class)", ylabel="observed accuracy",
           xlim=(0, 1), ylim=(0, 1))  # fmt: skip
    ax.legend(fontsize=8)
    ax.set_title("F13 reliability, leave-one-piece-out", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "F13_reliability", figures_dir)


def importance_figure(summ: dict[str, pd.DataFrame], top_n: int, figures_dir: Path) -> Path:
    """F14: per task, top single features and all families (mean over held-out pieces +- std)."""
    fig, axes = plt.subplots(2, len(summ), figsize=(6.4 * len(summ), 9.5), squeeze=False)
    for j, (task, s) in enumerate(summ.items()):
        for i, kind in enumerate(("feature", "family")):
            d = s[s["kind"] == kind].head(top_n if kind == "feature" else len(s))[::-1]
            ax = axes[i, j]
            ax.barh(d["item"], d["mean"], xerr=d["std"], color="#3a6ea5", error_kw={"lw": 0.6})
            ax.axvline(0, c="k", lw=0.6)
            ax.tick_params(axis="y", labelsize=6.5)
            ax.set_title(f"{task}: {kind} importance (metric drop when shuffled)", fontsize=9)
    fig.suptitle("F14 grouped permutation importance on held-out pieces", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "F14_permutation_importance", figures_dir)


def pdp_figure(pdp: pd.DataFrame, target: str, figures_dir: Path) -> Path:
    """F15: partial dependence of the final regressor on its top features."""
    feats = list(dict.fromkeys(pdp["feature"]))
    fig, axes = plt.subplots(1, len(feats), figsize=(4.0 * len(feats), 3.4), squeeze=False)
    for ax, f in zip(axes[0], feats, strict=True):
        d = pdp[pdp["feature"] == f]
        ax.plot(d["value"], d["mean_prediction"], "-o", ms=3, c="#3a6ea5")
        ax.set(xlabel=f, ylabel=f"mean predicted {target} (h)")
        ax.xaxis.label.set_size(7)
    fig.suptitle("F15 partial dependence (final model, all pieces)", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "F15_partial_dependence", figures_dir)
