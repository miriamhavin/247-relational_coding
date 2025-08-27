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

def mantel(space, B=5000, seed=42, ci_boot=0):
    rng = np.random.default_rng(seed)
    A = np.asarray(space.start_rsm, float); Bm = np.asarray(space.end_rsm, float); n = A.shape[0]
    if A.shape!=Bm.shape or n==0: return {'r':np.nan,'p_perm':np.nan,'boot_CI':None}
    r_obs = second_order_corr(A,Bm)
    ge=0; valid=0
    for _ in range(B):
        perm = rng.permutation(n)
        r = second_order_corr(A, Bm[perm][:,perm])
        if np.isfinite(r): valid+=1; ge += (r >= r_obs)
    p = (1+ge)/(1+valid if valid else 1)
    CI=None
    if ci_boot>0:
        rs=[]
        for _ in range(ci_boot):
            idx = rng.integers(0,n,size=n)
            rs.append(second_order_corr(A[np.ix_(idx,idx)], Bm[np.ix_(idx,idx)]))
        rs=[x for x in rs if np.isfinite(x)]
        if rs: CI = (float(np.quantile(rs,0.025)), float(np.quantile(rs,0.975)))
    return {'r':float(r_obs),'p_perm':float(p),'boot_CI':CI}

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
