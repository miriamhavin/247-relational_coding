# tfsdrift/stats.py
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from spaces import second_order_corr
from sklearn.linear_model import RidgeCV

def _z(x):  # Fisher z
    x = float(x) if np.isscalar(x) else x
    return float(np.arctanh(np.clip(x, -0.999999, 0.999999))) if np.isfinite(x) else np.nan

def same_diff(space, space_name='space'):
    C = np.asarray(space.cross_rsm, float); n = C.shape[0]
    if n==0:
        return (pd.DataFrame(columns=['space','word','self','others_mean','delta']),
                pd.DataFrame([{'space': space_name, 'n_words': 0,
                               'delta_median': np.nan, 'wilcoxon_p': np.nan, 'auroc': np.nan}]))
    diag = np.diag(C)
    rows = []
    for i in range(n):
        others = np.delete(C[i], i)
        rows.append({'space':space_name,'word':space.words[i],
                     'self':float(diag[i]), 'others_mean':float(others.mean()),
                     'delta':float(diag[i]-others.mean())})
    per = pd.DataFrame(rows)
    w = wilcoxon(per['delta'], zero_method='wilcox', alternative='greater', method='approx')
    # AUROC via Mann–Whitney:
    off = C[~np.eye(n, dtype=bool)]
    auroc = _mw_auc(diag, off)
    summ = pd.DataFrame([{'space': space_name, 'n_words': n,
                          'delta_median': float(per['delta'].median()),
                          'wilcoxon_p': float(getattr(w,'pvalue', np.nan)),
                          'auroc': float(auroc)}])
    return per, summ

def perm_same_vs_diff_group(C, B=2000, seed=42, return_null=False):
    """
    Permutation test for Same≠Different using *group-size matched* randomization.
    Null: the true diagonal is not special; any single column per row could be 'self'.

    For each perm b:
      - For each row i, randomly choose a column j as the pseudo-self.
      - Compute delta_i = C[i, j] - mean(C[i, -j])
      - Stat_b = mean_i delta_i
    Compare observed mean delta (true diagonal) to this null.

    Returns:
      {'observed': float, 'p_perm': float, 'B': int, 'null': np.ndarray (optional)}
    """
    C = np.asarray(C, float)
    n = C.shape[0]
    if C.ndim != 2 or C.shape[0] != C.shape[1] or n == 0:
        out = {'observed': np.nan, 'p_perm': np.nan, 'B': int(B)}
        if return_null: out['null'] = np.array([])
        return out

    rng = np.random.default_rng(seed)

    # Observed mean delta (true diagonal vs mean of others per row)
    diag = np.diag(C)
    row_sum = C.sum(axis=1)                 # shape (n,)
    obs_others_mean = (row_sum - diag) / (n - 1)
    obs_deltas = diag - obs_others_mean
    observed = float(np.nanmean(obs_deltas))

    # Permutation null: pick one pseudo-self per row
    null_stats = np.empty(B, dtype=float)
    for b in range(B):
        j = rng.integers(0, n, size=n)                  # one column per row
        chosen = C[np.arange(n), j]                     # shape (n,)
        others_mean = (row_sum - chosen) / (n - 1)      # exclude chosen col
        deltas = chosen - others_mean
        null_stats[b] = float(np.nanmean(deltas))

    # one-sided p: probability null >= observed
    ge = np.count_nonzero(null_stats >= observed)
    p = float((1 + ge) / (1 + B))

    out = {'observed': observed, 'p_perm': p, 'B': int(B)}
    if return_null:
        out['null'] = null_stats
    return out


def perm_columns(C, B=2000, metric='mean_delta', seed=42):
    rng = np.random.default_rng(seed)
    C = np.asarray(C, float); n = C.shape[0]
    def stat(Cm):
        d = np.diag(Cm)
        if metric=='mean_delta':
            vals = []
            for i in range(n):
                others = np.delete(Cm[i], i)
                vals.append(d[i]-others.mean())
            return float(np.mean(vals)) if vals else np.nan
        if metric=='top1':
            return float(np.mean(np.argmax(Cm,1)==np.arange(n)))
        if metric=='auroc':
            o = Cm[~np.eye(n, dtype=bool)]
            return _mw_auc(d, o)
        raise ValueError
    obs = stat(C); ge=0
    for _ in range(B):
        Cp = C[:, rng.permutation(n)]
        ge += (stat(Cp) >= obs)
    return {'metric':metric,'observed':float(obs),'p_perm':float((1+ge)/(1+B)),'B':int(B)}

def mantel(space, B=5000, seed=42, ci_boot=0, return_null=False):
    """
    Mantel-style permutation test on second-order correlation between start_rsm and end_rsm.

    Returns:
      {
        'r': float,                 # observed second-order correlation
        'p_perm': float,            # permutation p-value (>= observed, two-tailed-style on one side)
        'boot_CI': (lo, hi) | None, # optional bootstrap CI on r (your original behavior)
        'null_corrs': np.ndarray    # (valid,) permutation null correlations (present iff return_null=True)
        'valid_perms': int,         # number of finite null draws actually used
        'n': int                    # number of words (matrix size)
      }
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(space.start_rsm, float)
    Bm = np.asarray(space.end_rsm,   float)
    n  = A.shape[0]
    # shape checks
    if A.shape != Bm.shape or n == 0:
        out = {'r': np.nan, 'p_perm': np.nan, 'boot_CI': None, 'valid_perms': 0, 'n': int(n)}
        if return_null:
            out['null_corrs'] = np.array([], dtype=float)
        return out
    # observed
    r_obs = second_order_corr(A, Bm)
    ge = 0
    valid = 0
    nulls = [] if return_null else None
    # permutations
    for _ in range(int(B)):
        perm = rng.permutation(n)
        r = second_order_corr(A, Bm[perm][:, perm])
        if np.isfinite(r):
            valid += 1
            ge += (r >= r_obs)
            if return_null:
                nulls.append(r)

    # p-value with +1 smoothing (and guard for valid==0)
    p = (1 + ge) / (1 + valid if valid else 1)

    # optional bootstrap CI over r (as in your original)
    CI = None
    if ci_boot and n > 1:
        rs = []
        for _ in range(int(ci_boot)):
            idx = rng.integers(0, n, size=n)
            r_b = second_order_corr(A[np.ix_(idx, idx)], Bm[np.ix_(idx, idx)])
            if np.isfinite(r_b):
                rs.append(r_b)
        if rs:
            rs = np.asarray(rs, float)
            CI = (float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975)))

    out = {
        'r': float(r_obs),
        'p_perm': float(p),
        'boot_CI': CI,
        'valid_perms': int(valid),
        'n': int(n),
    }
    if return_null:
        out['null_corrs'] = np.asarray(nulls, dtype=float)
    return out

def _mw_auc(x,y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size==0 or y.size==0: return np.nan
    vals = np.concatenate([x,y])
    ranks = pd.Series(vals).rank(method='average').to_numpy()
    nx = x.size; rx = ranks[:nx].sum()
    U = rx - nx*(nx+1)/2.0
    return float(U / (nx*y.size + 1e-12))

def _predict_end_from_start(S, E, alphas=np.logspace(-3, 3, 7)):
    """Return Ê given S, and overall vector corr(Ê, E)."""
    S = np.asarray(S, float); E = np.asarray(E, float)
    if S.size == 0 or E.size == 0 or S.shape[0] != E.shape[0]:
        return np.empty_like(E), np.nan
    model = RidgeCV(alphas=alphas).fit(S, E)
    Ehat = model.predict(S)
    corr = np.corrcoef(Ehat.ravel(), E.ravel())[0, 1] if Ehat.size and E.size else np.nan
    return Ehat, float(corr)
