# reporting.py
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV
from spaces import _rsm, second_order_corr, Space
from plots import plot_all
from stats import same_diff, perm_columns, mantel, _mw_auc, _predict_end_from_start, _z, perm_same_vs_diff_group

# -------------------- small helpers --------------------
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

    row = {'space': label, 'n_words': n}

    # --- first order
    row['diag_mean'] = float(np.nanmean(diag)) if diag.size else np.nan

    # --- second order baseline
    if start_rsm.size and end_rsm.size:
        r2 = second_order_corr(start_rsm, end_rsm)
        row['r_start_vs_end_RSM'] = float(r2)
        row['z_fisher'] = _z(r2)
        row['mantel_p'] = mantel(space, B=B_mantel, seed=seed)['p_perm']
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


    # column-permutation nulls 
    row['perm_meanDelta_p'] = perm_columns(cross, B=B_perm_cols, metric='mean_delta', seed=seed)['p_perm']
    # group-size matched nulls
    group_perm = perm_same_vs_diff_group(cross, B=B_perm_cols, seed=seed, return_null=False)
    row['perm_groupDelta_p'] = group_perm['p_perm']

    # --- save consolidated df
    df = pd.DataFrame([row])
    if outdir:
        outpath = os.path.join(outdir, f"{label}_all_summary.csv")
        _save(df, outpath)
        plot_all(space, label, outdir)

    print(f"[{label}] summary saved")
    return df
