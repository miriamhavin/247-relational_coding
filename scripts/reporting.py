# reporting.py
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV
from spaces import _rsm, second_order_corr, Space
from stats import same_diff, perm_columns, mantel, _mw_auc, _predict_end_from_start, _z, perm_same_vs_diff_group
from plot_all import *
# -------------------- small helpers --------------------
import os, json, time
import numpy as np

def _meta_dict(**kw):
    kw["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return kw

def save_space_npz(outdir, space, meta, pred_end=None, ridge=None, nulls=None):
    os.makedirs(outdir, exist_ok=True)
    words = np.array(space.words, dtype=object)
    payload = {
        "words": words,
        "start_vecs": np.asarray(space.start_vecs, dtype=np.float32),
        "end_vecs":   np.asarray(space.end_vecs,   dtype=np.float32),
    }
    if pred_end is not None:
        payload["pred_end_vecs"] = np.asarray(pred_end, dtype=np.float32)
    if ridge is not None:
        # ridge={"W": (D,D), "b": (D,), "alpha": float}
        payload["ridge_W"] = np.asarray(ridge.get("W"), dtype=np.float32)
        payload["ridge_b"] = np.asarray(ridge.get("b"), dtype=np.float32)
        payload["ridge_alpha"] = np.float32(ridge.get("alpha", np.nan))
    if nulls is not None and "null_rsm_corrs" in nulls:
        payload["null_rsm_corrs"] = np.asarray(nulls["null_rsm_corrs"], dtype=np.float32)

    np.savez_compressed(os.path.join(outdir, "space.npz"), **payload)
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def load_space_npz(path):
    import numpy as np, json, os
    d = np.load(os.path.join(path, "space.npz"), allow_pickle=True)
    with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    words = d["words"].tolist()
    start_vecs = d["start_vecs"]; end_vecs = d["end_vecs"]
    pred_end = d["pred_end_vecs"] if "pred_end_vecs" in d.files else None
    ridge = None
    if "ridge_W" in d.files:
        ridge = {"W": d["ridge_W"], "b": d["ridge_b"], "alpha": float(d["ridge_alpha"])}
    nulls = {"null_rsm_corrs": d["null_rsm_corrs"]} if "null_rsm_corrs" in d.files else {}
    return words, start_vecs, end_vecs, pred_end, ridge, meta, nulls


def _get(space, key, default=None):
    # supports both dict-like (space['cross_rsm']) and attr-like (space.cross_rsm)
    if isinstance(space, dict): return space.get(key, default)
    return getattr(space, key, default)

def _save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

# -------------------- main API --------------------
def report_space(space, label, outdir, *, B_perm_cols=2000, B_mantel=5000, seed=42):

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    words      = np.asarray(_get(space, 'words', np.array([])))
    start_vecs = np.asarray(_get(space, 'start_vecs', np.empty((0, 0))), float)
    end_vecs   = np.asarray(_get(space, 'end_vecs',   np.empty((0, 0))), float)
    start_rsm  = np.asarray(_get(space, 'start_rsm',  np.zeros((0, 0))), float)
    end_rsm    = np.asarray(_get(space, 'end_rsm',    np.zeros((0, 0))), float)
    cross      = np.asarray(_get(space, 'cross_rsm',  np.zeros((0, 0))), float)
    n          = int(cross.shape[0])
    diag       = np.diag(cross) if cross.size else np.array([])
    time_gaps  = np.asarray(_get(space, 'time_gaps', np.array([])), float)


    row = {'space': label, 'n_words': n}
    plots_dir = os.path.join(outdir, "plots")

    # --- first order
    row['diag_mean'] = float(np.nanmean(diag)) if diag.size else np.nan
    
    # safe time gap stats (avoid RuntimeWarning on empty or all-NaN)
    if time_gaps.size and np.isfinite(time_gaps).any():
        row['time_gap_mean'] = float(np.nanmean(time_gaps))
        row['time_gap_median'] = float(np.nanmedian(time_gaps))
        # also hours (samples @512Hz -> hours)
        row['time_gap_mean_hours'] = row['time_gap_mean'] / 512.0 / 3600.0
        row['time_gap_median_hours'] = row['time_gap_median'] / 512.0 / 3600.0
    else:
        row['time_gap_mean'] = np.nan
        row['time_gap_median'] = np.nan
        row['time_gap_mean_hours'] = np.nan
        row['time_gap_median_hours'] = np.nan
    # time gap histogram (prefers space.time_gaps, falls back to start/end times)
    if outdir:
        try:
            plot_time_gap_hist(space, label, outpath=os.path.join(plots_dir, f"{label}_time_gap_hist_samples.png"), units='samples')
            plot_time_gap_hist(space, label, outpath=os.path.join(plots_dir, f"{label}_time_gap_hist_hours.png"), units='hours')
        except Exception as e:
            print(f"[warn] time gap plot failed for {label}: {e}")
    

    per_word_diag = diag.copy() if diag.size else np.array([])
    per_word_diag_mean = group_mean_by_word(words, per_word_diag) if per_word_diag.size else np.array([])
    out_png = os.path.join(plots_dir, f"{label}_start_end_corr_matrix.png")
    plot_start_end_heatmap(space, f"{label} – start-end correlation (ordered by diag)", outpath=out_png)
    hist(per_word_diag_mean,
            title=f"{label} – histogram of corr(start_i, end_i) per WORD",
            xlabel="per-word mean corr(start, end)",
            outpath=os.path.join(plots_dir, f"{label}_hist_perword_start_end.png"),
            bins=40, xlim=(-1, 1))
    # --- second order baseline
    if start_rsm.size and end_rsm.size:
        r2 = second_order_corr(start_rsm, end_rsm)
        row['r_start_vs_end_RSM'] = float(r2)
        row['z_fisher'] = _z(r2)
        mantel_out = mantel(space, B=B_mantel, seed=seed, return_null=True)
        row['mantel_p'] = mantel_out['p_perm']
        plot_second_order_corr_hist(space.start_rsm, space.end_rsm,
            label, outdir=os.path.join(outdir, "plots"),
            bins=40,
            null_corrs=mantel_out.get("null_corrs"), 
            fisher=False
        )
        try:
            plot_first_vs_second_order(
                row['diag_mean'], row['r_start_vs_end_RSM'], label,
                outpath=os.path.join(plots_dir, f"{label}_first_vs_second_order.png"),
                ylim=(-1,1)
            )
        except Exception as e:
            print(f"[warn] failed first-vs-second order plot for {label}: {e}")
    else:
        row.update({'r_start_vs_end_RSM': np.nan, 'z_fisher': np.nan, 'mantel_p': np.nan})

    # --- prediction-based second order
    if start_vecs.size and end_vecs.size and start_rsm.size:
        Ehat, corr_pred = _predict_end_from_start(start_vecs, end_vecs)
        row['E_pred_vs_E_corr'] = corr_pred

        end_pred_rsm  = _rsm(Ehat)
        resid         = end_vecs - Ehat
        end_resid_rsm = _rsm(resid)
        setattr(space, "end_pred_rsm",  end_pred_rsm)
        setattr(space, "end_resid_rsm", end_resid_rsm)

        row['r_start_vs_endPred_RSM'] = second_order_corr(start_rsm, end_pred_rsm)
        row['r_start_vs_endResid_RSM'] = second_order_corr(start_rsm, end_resid_rsm)

        # Mantel on (start_rsm vs end_pred_rsm): wrap into a lightweight Space
        man_pred  = mantel(Space(words, start_vecs, Ehat,  start_rsm, end_pred_rsm,  cross, diag),
                           B=B_mantel, seed=seed)
        man_resid = mantel(Space(words, start_vecs, resid, start_rsm, end_resid_rsm, cross, diag),
                           B=B_mantel, seed=seed)
        row['mantel_p_pred']  = man_pred['p_perm']
        row['mantel_p_resid'] = man_resid['p_perm']
        bars_three(
            end_val=float(row['r_start_vs_end_RSM']),
            pred_val=float(row['r_start_vs_endPred_RSM']),
            resid_val=float(row['r_start_vs_endResid_RSM']),
            title=f"{label} – start vs (end / pred / resid) – second-order r",
            outpath=os.path.join(plots_dir, f"{label}_compare_end_pred_resid.png"),
            ylim=(-1, 1)
        )
    else:
        row.update({
            'E_pred_vs_E_corr': np.nan,
            'r_start_vs_endPred_RSM': np.nan,  'mantel_p_pred':  np.nan,
            'r_start_vs_endResid_RSM': np.nan, 'mantel_p_resid': np.nan
        })

    # --- same ≠ different
    same_per, same_sum = same_diff(space, label)
    row.update({
        'delta_median': float(same_sum['delta_median'].iloc[0]) if len(same_sum) else np.nan,
        'wilcoxon_p':   float(same_sum['wilcoxon_p'].iloc[0])   if len(same_sum) else np.nan,
        'auroc':        float(same_sum['auroc'].iloc[0])        if len(same_sum) else np.nan
    })
    per_word_delta = same_per['delta'].to_numpy() if 'delta' in same_per.columns else np.array([])


    # column-permutation nulls 
    perm_cols = perm_columns(cross, B=B_perm_cols, metric='mean_delta', seed=seed)
    row['perm_meanDelta_p'] = perm_cols['p_perm']
    # group-size matched nulls
    group_perm = perm_same_vs_diff_group(cross, B=B_perm_cols, seed=seed, return_null=False)
    row['perm_groupDelta_p'] = group_perm['p_perm']

    # --- save consolidated df
    df = pd.DataFrame([row])
    if outdir:
        outpath = os.path.join(outdir, f"{label}_all_summary.csv")
        # NOTE: previously the summary CSV was never written, so columns like
        # 'time_gap_mean' / 'time_gap_median' would never show up for inspection.
        # We now persist it explicitly.
        try:
            df.to_csv(outpath, index=False)
        except Exception as e:
            print(f"[warn] failed to write summary CSV {outpath}: {e}")
        art = {
            "words": words,
            "start_vecs": start_vecs,
            "end_vecs": end_vecs,
            "start_rsm": start_rsm,
            "end_rsm": end_rsm,
            "cross_rsm": cross,
            "diag": per_word_diag,     
            "delta": per_word_delta,    

        }
        # save raw time gaps if present so downstream code can recompute stats
        if time_gaps is not None and time_gaps.size:
            art["time_gaps"] = time_gaps
    # span metrics removed
    row['time_span_mean_hours'] = np.nan
    row['time_span_median_hours'] = np.nan
        if hasattr(space, "start_times") and space.start_times is not None and len(space.start_times):
            gaps = space.end_times - space.start_times
            row["mean_time_gap"] = float(np.nanmean(gaps))
            row["median_time_gap"] = float(np.nanmedian(gaps))
        else:
            row["mean_time_gap"] = np.nan
            row["median_time_gap"] = np.nan

        # also prediction-based if available
        if hasattr(space, "end_pred_rsm"):
            art["end_pred_rsm"] = space.end_pred_rsm
        if hasattr(space, "end_resid_rsm"):
            art["end_resid_rsm"] = space.end_resid_rsm
        if mantel_out is not None and 'null_corrs' in mantel_out:
            art["mantel_null_corrs"] = mantel_out["null_corrs"]
        if 'null_vals' in perm_cols:
            art["perm_cols_null_vals"] = perm_cols["null_vals"]
        if 'null_vals' in group_perm:
            art["perm_group_null_vals"] = group_perm["null_vals"]

        np.savez_compressed(
            os.path.join(outdir, f"{label}_artifacts.npz"), **art
        )

    print(f"[{label}] summary saved")
    return df
