"""``foodlens finalize`` (P6.T3-T5): select, test, calibrate, interpret and save the final model.

Steps (all under leave-one-piece-out unless stated):
1. selection rule (``models.select``) on ``model_comparison.csv`` -> ``model_selection.json``;
2. grouped permutation test of the selected classifier (full nested CV per repetition);
3. calibration before/after + Mahalanobis OOD abstention -> ``tables/calibration.csv``,
   ``reliability.csv``, ``ood_abstention.csv``, ``ood_per_piece.csv``, F13;
4. grouped permutation importance (class + equivalent-hour tasks) -> ``importance_*.csv``, F14;
5. final bundle on ALL pieces -> ``models/<final_model_name>/`` + partial dependence F15;
6. F12 light-ablation figure from ``tables/light_ablation.csv`` + the meat-vs-reference trend
   check per light (``tables/light_meat_vs_ref_trends.csv``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from foodlens.config import Config
from foodlens.evaluation.cv import evaluate
from foodlens.evaluation.importance import (
    grouped_permutation_importance,
    partial_dependence,
    summarize_importance,
)
from foodlens.evaluation.splits import get_groups, logo_splits
from foodlens.logging_utils import get_logger
from foodlens.models.calibration import evaluate_calibration
from foodlens.models.card import model_card
from foodlens.models.classical import tuned_factory
from foodlens.models.feature_sets import feature_set
from foodlens.models.select import fit_final, save_final, select_model
from foodlens.models.train import limited_table
from foodlens.stages import tables_dir

log = get_logger("models.finalize")


def _permutation(fam: str, df, feats, groups, cfg: Config) -> dict:
    target = cfg.evaluation.class_target
    res = evaluate(tuned_factory(fam, cfg), "classification", df, feats, target, groups,
                   logo_splits(groups), cfg, True, cfg.train.n_permutations)  # fmt: skip
    return res["permutation"]


def _calibration(fam, df, feats, groups, cfg: Config, figs) -> dict:
    from foodlens.evaluation.p6_plots import reliability_figure

    cal = evaluate_calibration(fam, df, feats, groups, cfg)
    tab = tables_dir(cfg)
    rows = [{"model": f"{fam} {k}", **{m: v for m, v in d.items() if m != "ci"},
             **{f"{m}_ci_{s}": d["ci"][m][i] for m in d["ci"] for i, s in enumerate(("lo", "hi"))}}
            for k, d in cal["metrics"].items()]  # fmt: skip
    pd.DataFrame(rows).to_csv(tab / "calibration.csv", index=False)
    pd.concat([r.assign(kind=k) for k, r in cal["reliability"].items()]).to_csv(
        tab / "reliability.csv", index=False)  # fmt: skip
    pd.DataFrame(cal["ood"]["overall"]).to_csv(tab / "ood_abstention.csv", index=False)
    cal["ood"]["per_group"].to_csv(tab / "ood_per_piece.csv", index=False)
    reliability_figure(cal["reliability"], {k: d["ece"] for k, d in cal["metrics"].items()}, figs)
    return {"metrics": cal["metrics"], "ood": cal["ood"]["overall"],
            "use_calibrated": cal["use_calibrated"]}  # fmt: skip


def _importance(fam, df, feats, groups, cfg: Config, figs) -> dict[str, pd.DataFrame]:
    from foodlens.evaluation.p6_plots import importance_figure

    ev, out = cfg.evaluation, {}
    for task, target in (("classification", ev.class_target), ("regression", ev.targets[0])):
        imp = grouped_permutation_importance(tuned_factory(fam, cfg), task, df, feats, target,
                                             groups, cfg)  # fmt: skip
        out[task] = summarize_importance(imp)
        imp.to_csv(tables_dir(cfg) / f"importance_{task}_folds.csv", index=False)
        out[task].to_csv(tables_dir(cfg) / f"importance_{task}.csv", index=False)
    importance_figure(out, cfg.train.importance_top_n, figs)
    return out


def _light_figure(cfg: Config, table: pd.DataFrame, figs) -> None:
    from foodlens.evaluation.p6_plots import light_ablation_figure

    path = tables_dir(cfg) / "light_ablation.csv"
    if not path.is_file():
        log.warning("light_ablation.csv missing: run `foodlens train --experiment light_ablation`")
        return
    from foodlens.evaluation.light_check import meat_vs_reference_trends

    meat_vs_reference_trends(cfg).to_csv(tables_dir(cfg) / "light_meat_vs_ref_trends.csv",
                                         index=False)  # fmt: skip
    la = pd.read_csv(path)
    base = table[(table["model"] == "baseline_red_median") & (table["scheme"] == "LOGO_piece_id")]
    lines = {r["target"]: ("RED-median baseline", r["value"]) for _, r in base.iterrows()}
    light_ablation_figure(la, 1 / len(cfg.labels.classes), lines, figs)


def per_piece_failures(cfg: Config, model: str) -> pd.DataFrame:
    """Per held-out piece metrics of ``model`` (model_zoo out-of-fold) + n evaluated triplets
    and the piece's max equivalent hours -> ``tables/per_piece_selected.csv``."""
    from foodlens.evaluation.paired import per_piece_table
    from foodlens.models.train import oof_path

    oof = pd.concat([pd.read_parquet(oof_path(cfg, "model_zoo", t))
                     for t in ("classification", "regression")], ignore_index=True)  # fmt: skip
    t = per_piece_table(oof, model, cfg).set_index("group")
    head = oof[(oof["model"] == model) & (oof["target"] == cfg.evaluation.class_target)
               & (oof["scheme"] == f"LOGO_{cfg.evaluation.group_cols[0]}")]  # fmt: skip
    t["n_eval"] = head.groupby("group").size()
    lab = pd.read_csv(tables_dir(cfg) / "label_summary.csv").set_index("session_id")
    t = t.join(lab[["equiv_h_max"]]).reset_index()
    t.to_csv(tables_dir(cfg) / "per_piece_selected.csv", index=False)
    return t


def run_finalize(cfg: Config, limit: int | None, permute: bool = True) -> dict:
    """Run steps 1-6; returns the selection decision with the added results."""
    figs = cfg.path("reports_dir") / "figures"
    table = pd.read_csv(tables_dir(cfg) / "model_comparison.csv")
    dec = select_model(table, cfg)
    log.info("selected %s (%s = %.3f, meets rule: %s)", dec["model"], dec["metric"],
             dec["value"], dec["meets_rule"])  # fmt: skip
    df = limited_table(cfg, limit)
    feats = feature_set(dec["feature_set"], cfg)
    groups = get_groups(df, cfg, [cfg.evaluation.group_cols[0]])
    fam = dec["family"]
    dec["permutation"] = _permutation(fam, df, feats, groups, cfg) if permute else {}
    dec["calibration"] = _calibration(fam, df, feats, groups, cfg, figs)
    imp = _importance(fam, df, feats, groups, cfg, figs)
    bundle = fit_final(df, feats, groups, fam, cfg, dec["calibration"]["use_calibrated"])
    top = imp["regression"].query("kind == 'feature'")["item"].head(cfg.train.pdp_top_k)
    pdp = pd.concat([partial_dependence(bundle["regressor"], df, feats, f, cfg.train.pdp_grid)
                     for f in top], ignore_index=True)  # fmt: skip
    pdp.to_csv(tables_dir(cfg) / "partial_dependence.csv", index=False)
    from foodlens.evaluation.p6_plots import pdp_figure

    pdp_figure(pdp, cfg.evaluation.targets[0], figs)
    _light_figure(cfg, table, figs)
    per_piece_failures(cfg, dec["model"])
    dec["final_params"] = {"classifier": bundle["classifier"].best_params_,  # tuned params
                           "regressor": bundle["regressor"].best_params_}  # fmt: skip
    bundle["selection"] = dec
    out = save_final(bundle, model_card(dec, table, imp, bundle, cfg), cfg)
    (tables_dir(cfg) / "model_selection.json").write_text(json.dumps(dec, indent=2, default=_js))
    log.info("final model saved to %s", out)
    return dec


def _js(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)
