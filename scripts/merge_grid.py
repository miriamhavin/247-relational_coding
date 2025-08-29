#!/usr/bin/env python3
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.expanduser("~/.cache/mpl"))
import matplotlib
matplotlib.use("Agg")  # headless plotting

import re, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon, ttest_rel
import matplotlib.pyplot as plt

MERGED_DIR = "results/combined"
MERGED_PKL = os.path.join(MERGED_DIR, "all_spaces_summary.pkl")
OUT_DIR    = os.path.join(MERGED_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# only keep metrics that actually exist in file
ALL_METRICS = ["diag_mean","r_start_vs_end_RSM","E_pred_vs_E_corr","delta_median","auroc"]

def _read_merged():
    print("1) reading merged pickle…", flush=True)
    if not os.path.exists(MERGED_PKL):
        raise FileNotFoundError(f"Missing {MERGED_PKL}. Run your merge first.")
    df = pd.read_pickle(MERGED_PKL)
    if "scope" not in df.columns:
        raise ValueError("Expected 'scope' column not found in merged file.")
    return df

def _sid_from_run_tag(tag):
    m = re.findall(r"\d+", str(tag))
    return m[-1] if m else str(tag)

def _se(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan

def _agg_stats(df, index_cols, metrics):
    agg = (df.groupby(index_cols, dropna=False)
             .agg({m: ['mean','std',_se] for m in metrics})
             .rename(columns={'_se': 'se'}))
    agg.columns = [f"{a}_{b}" for a,b in agg.columns]
    return agg.reset_index()

def _paired_stats(subj_df, metric, condA, condB, pair_cols):
    a_df = subj_df.merge(condA, on=list(condA.columns), how="inner")
    b_df = subj_df.merge(condB, on=list(condB.columns), how="inner")
    on = ["sid"] + list(pair_cols)
    merged = a_df.merge(b_df, on=on, suffixes=("_A","_B"), how="inner")
    if merged.empty: return dict(n=0, wilcoxon_p=np.nan, ttest_p=np.nan)
    a = merged[f"{metric}_A"].to_numpy(); b = merged[f"{metric}_B"].to_numpy()
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a)==0: return dict(n=0, wilcoxon_p=np.nan, ttest_p=np.nan)
    try: w_p = wilcoxon(a, b, zero_method="wilcox").pvalue
    except Exception: w_p = np.nan
    try: t_p = ttest_rel(a, b, nan_policy="omit").pvalue
    except Exception: t_p = np.nan
    return dict(n=int(len(a)), wilcoxon_p=float(w_p), ttest_p=float(t_p))

def _barplot(df_stats, metric, outdir=OUT_DIR):
    if not {f"{metric}_mean", f"{metric}_se", "feature_modality","task_modality"}.issubset(df_stats.columns):
        return
    modes = ["embedding","neural"]; x = np.arange(len(modes)); width = 0.35
    fig = plt.figure(figsize=(6,4)); ax = plt.gca()
    def pick(task):
        vals, ses = [], []
        for m in modes:
            row = df_stats[(df_stats.feature_modality==m)&(df_stats.task_modality==task)]
            vals.append(float(row[f"{metric}_mean"]) if not row.empty else np.nan)
            ses.append(float(row[f"{metric}_se"]) if not row.empty else np.nan)
        return np.array(vals), np.array(ses)
    prod, prod_se = pick("production"); comp, comp_se = pick("comprehension")
    ax.bar(x - width/2, prod, width, yerr=prod_se, label="production")
    ax.bar(x + width/2, comp, width, yerr=comp_se, label="comprehension")
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_ylabel(metric); ax.set_title(f"{metric}: mean ± SE across subjects")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"bar_{metric}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

def main():
    big = _read_merged()

    print("2) filtering to global rows…", flush=True)
    df = big[big["scope"]=="global"].copy()
    if df.empty: raise SystemExit("No global rows found in merged summary.")
    df["sid"] = df["run_tag"].apply(_sid_from_run_tag)

    present_metrics = [m for m in ALL_METRICS if m in df.columns]
    keep_cols = ["sid","run_tag","feature_modality","task_modality","scope"] + present_metrics
    df = df[keep_cols].copy()
    print(f"   metrics found: {present_metrics}", flush=True)

    print("3) subject-level grouping…", flush=True)
    subj_tbl = (df.groupby(["feature_modality","task_modality","sid"], dropna=False)[present_metrics]
                  .mean().reset_index())
    subj_tbl.to_csv(os.path.join(OUT_DIR,"subject_level_global.csv"), index=False)

    print("4) across-subject stats…", flush=True)
    across = _agg_stats(subj_tbl, ["feature_modality","task_modality"], present_metrics)
    # ---------- PRED vs RESID second-order summary ----------
    need = ["r_start_vs_end_RSM","r_start_vs_endPred_RSM","r_start_vs_endResid_RSM","E_pred_vs_E_corr"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print("Missing columns (likely older runs):", missing)
    else:
        # subject-level table first (one row per subject × feature × task)
        subj_pred = (df.dropna(subset=["feature_modality","task_modality"])
                    .groupby(["feature_modality","task_modality","sid"], dropna=False)[need]
                    .mean()
                    .reset_index())

        # across-subject means/SE
        def se(x): 
            x = pd.Series(x).dropna()
            return (x.std(ddof=1)/len(x)**0.5) if len(x)>1 else float("nan")

        agg = (subj_pred
            .groupby(["feature_modality","task_modality"], dropna=False)
            .agg({
                "r_start_vs_end_RSM":["mean",se],
                "r_start_vs_endPred_RSM":["mean",se],
                "r_start_vs_endResid_RSM":["mean",se],
                "E_pred_vs_E_corr":["mean",se],
            }))

        # clean columns
        agg.columns = ['_'.join(col).replace("<lambda>","se") for col in agg.columns]
        agg = agg.reset_index()

        # deltas that answer your question explicitly
        agg["Δ(Pred − Plain)_mean"]  = agg["r_start_vs_endPred_RSM_mean"]  - agg["r_start_vs_end_RSM_mean"]
        agg["Δ(Pred − Resid)_mean"]  = agg["r_start_vs_endPred_RSM_mean"]  - agg["r_start_vs_endResid_RSM_mean"]

        # save + print
        out_csv = os.path.join(OUT_DIR, "second_order_pred_vs_resid_global.csv")
        agg.to_csv(out_csv, index=False)
        print("\n=== Second-order: Pred vs Plain vs Resid (means ± SE) ===")
        print(agg[[
            "feature_modality","task_modality",
            "r_start_vs_end_RSM_mean","r_start_vs_end_RSM_se",
            "r_start_vs_endPred_RSM_mean","r_start_vs_endPred_RSM_se",
            "r_start_vs_endResid_RSM_mean","r_start_vs_endResid_RSM_se",
            "E_pred_vs_E_corr_mean","E_pred_vs_E_corr_se",
            "Δ(Pred − Plain)_mean","Δ(Pred − Resid)_mean"
        ]].to_string(index=False))
        print("→ wrote:", out_csv)

    across.to_csv(os.path.join(OUT_DIR,"across_subjects_global.csv"), index=False)

    print("5) paired stats…", flush=True)
    stats_rows = []
    for metric in present_metrics:
        for task in ["production","comprehension"]:
            res = _paired_stats(subj_tbl, metric,
                                pd.DataFrame({"feature_modality":["embedding"],"task_modality":[task]}),
                                pd.DataFrame({"feature_modality":["neural"],   "task_modality":[task]}),
                                pair_cols=("task_modality",))
            stats_rows.append({"contrast":f"embedding_vs_neural | {task}","metric":metric, **res})
        for mod in ["embedding","neural"]:
            res = _paired_stats(subj_tbl, metric,
                                pd.DataFrame({"feature_modality":[mod],"task_modality":["production"]}),
                                pd.DataFrame({"feature_modality":[mod],"task_modality":["comprehension"]}),
                                pair_cols=("feature_modality",))
            stats_rows.append({"contrast":f"production_vs_comprehension | {mod}","metric":metric, **res})
    pd.DataFrame(stats_rows).to_csv(os.path.join(OUT_DIR,"paired_stats_global.csv"), index=False)

    print("6) plotting…", flush=True)
    for metric in present_metrics:
        _barplot(across, metric, OUT_DIR)

    print("✓ done. wrote to", OUT_DIR)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e); sys.exit(1)
