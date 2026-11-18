"""Which light / feature carries a consistent time signal? (P3.T5, F8)

For each feature x light, the per-session Spearman rho with elapsed hours (``trends.trend_table``
on step-corrected values) is summarised over sessions: mean rho, mean |rho|, the number of
sessions with the majority sign, and how many of those are significant (``eda.alpha``). A feature
is "promising" only if its sign agrees in >= ``eda.promising_min_sessions`` sessions (out of 5).
Cross-light ratios on triplets get light = "TRIPLET".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def signal_ranking(trends: pd.DataFrame, alpha: float, min_same: int) -> pd.DataFrame:
    """One row per feature x light, ranked (promising first, then agreement, then mean |rho|)."""
    rows = []
    for (feat, light), g in trends.groupby(["feature", "light"]):
        g = g[np.isfinite(g["rho"])]
        n = len(g)
        if n == 0:
            continue
        n_pos, n_neg = int((g["rho"] > 0).sum()), int((g["rho"] < 0).sum())
        sign = 1 if n_pos >= n_neg else -1
        n_same = max(n_pos, n_neg)
        sig_same = int(((np.sign(g["rho"]) == sign) & (g["p"] < alpha)).sum())
        rows.append(
            {
                "feature": feat,
                "light": light,
                "n_sessions": n,
                "majority_sign": "+" if sign > 0 else "-",
                "n_same_sign": n_same,
                "n_sig_same_sign": sig_same,
                "consistency": n_same / n,
                "consistency_label": f"{n_same}/{n}",
                "mean_rho": float(g["rho"].mean()),
                "mean_abs_rho": float(g["rho"].abs().mean()),
                "median_slope_pct_h": float(g["slope_pct_h"].median()),
                "promising": n_same >= min_same,
            }
        )
    out = pd.DataFrame(rows)
    keys = ["promising", "n_same_sign", "n_sig_same_sign", "mean_abs_rho"]
    return out.sort_values(keys, ascending=False).reset_index(drop=True)


def light_comparison(rank: pd.DataFrame, sc: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    """Per light (and TRIPLET): how many features are promising / fully consistent, plus the
    signal-to-noise of the meat brightness (``roi_mean``, step-corrected): median |trend| in % of
    the start per hour over sessions, divided by the frame-to-frame noise in % (1.4826 x MAD of
    consecutive relative differences / sqrt 2). Higher = the change shows sooner above the noise."""
    rows = []
    for light, g in rank.groupby("light"):
        top = g.sort_values("mean_abs_rho", ascending=False).head(5)
        row = {
            "light": light,
            "n_features": len(g),
            "n_promising": int(g["promising"].sum()),
            "n_all_sessions_significant": int((g["n_sig_same_sign"] == g["n_sessions"]).sum()),
            "top5_mean_abs_rho": float(top["mean_abs_rho"].mean()),
            "top5_features": ", ".join(top["feature"]),
        }
        s = sc[sc["light"] == light].sort_values("ts")
        if len(s):
            rel = s.groupby("session_id")["roi_mean"].transform(lambda x: x.pct_change())
            noise = 100 * 1.4826 * float((rel - rel.median()).abs().median()) / np.sqrt(2)
            tr = trends[(trends["light"] == light) & (trends["feature"] == "roi_mean")]
            trend = float(tr.loc[tr["variant"] == "step_corrected", "slope_pct_h"].abs().median())
            row |= {
                "roi_mean_level": float(s["roi_mean"].median()),
                "roi_mean_trend_pct_h": trend,
                "roi_mean_noise_pct": noise,
                "roi_mean_snr_per_h": trend / noise if noise else np.nan,
            }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_all_sessions_significant", ascending=False)


def key_numbers(
    sc: pd.DataFrame,
    trends: pd.DataFrame,
    mixed: pd.DataFrame,
    rank: pd.DataFrame,
    dyn: pd.DataFrame,
) -> pd.DataFrame:
    """Headline EDA numbers quoted by docs/07 (metric, light, value, note)."""
    rows = [
        ("frames", "all", len(sc), "usable frames"),
        ("sessions", "all", sc["session_id"].nunique(), ""),
    ]
    rel = mixed[(mixed["scale"] == "relative to start") & (mixed["feature"] == "roi_median")]
    for r in rel.itertuples():
        rows.append(("mixedlm_roi_median_pct_per_h", r.light, 100 * r.slope_h,
                     f"95% CI {100 * r.ci_lo:.1f} to {100 * r.ci_hi:.1f}"))  # fmt: skip
    sc_tr = trends[trends["variant"] == "step_corrected"]
    for light, g in sc.groupby("light"):
        start = g.sort_values("ts").groupby("session_id")["roi_mean"].first()
        rows.append(("start_roi_mean_min", light, start.min(), "between-session spread of level"))
        rows.append(("start_roi_mean_max", light, start.max(), ""))
        for f in ("roi_mean", "ref_mean"):
            s = sc_tr[(sc_tr["light"] == light) & (sc_tr["feature"] == f)]["slope_pct_h"]
            rows.append((f"{f}_slope_pct_h_median", light, s.median(),
                         f"range {s.min():.1f} to {s.max():.1f} over sessions"))  # fmt: skip
    rows.append(("promising_features", "all", int(rank["promising"].sum()), f"of {len(rank)}"))
    for v, g in dyn.groupby("variant"):
        smaller = int(((g["size_rho"] < 0) & (g["size_p"] < 0.05)).sum())
        rows.append((f"glint_size_shrinks_{v}", "all", smaller, f"of {len(g)} session x light"))
        for light, h in g.groupby("light"):
            n_fewer = int(((h["count_rho"] < 0) & (h["count_p"] < 0.05)).sum())
            rows.append((f"glint_count_falls_{v}", light, n_fewer, f"of {len(h)} sessions"))
    return pd.DataFrame(rows, columns=["metric", "light", "value", "note"])
