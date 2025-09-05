# tfsdrift/spaces.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from plot_all import occurrence_matrix_plot

@dataclass
class Space:
    words: np.ndarray          # (n,)
    start_vecs: np.ndarray     # (n,d)
    end_vecs: np.ndarray       # (n,d)
    start_rsm: np.ndarray      # (n,n)
    end_rsm: np.ndarray        # (n,n)
    cross_rsm: np.ndarray      # (n,n)
    diag: np.ndarray           # (n,)
    start_times: np.ndarray | None = None
    end_times: np.ndarray | None = None
    time_gaps: np.ndarray | None = None

def _to_2d(x): x = np.asarray(x); return x if x.ndim==2 else x[:,None]

def _zscore_rows(M, eps=1e-8):
    M = np.asarray(M, float); m = M.mean(1, keepdims=True); s = M.std(1, keepdims=True)
    return (M - m) / (s + eps)

def _rsm(X):
    if X.size == 0: return np.empty((0,0))
    Z = _zscore_rows(X)
    return (Z @ Z.T) / Z.shape[1]

def _rowwise_corr(A, B):
    A = _zscore_rows(A); B = _zscore_rows(B)
    return (A @ B.T) / A.shape[1]

def _split_first_last(rows, min_occ):
    n = len(rows)
    if n < min_occ: return [], []
    mid = n // 2
    return rows[:mid], rows[mid:]

def _avg_halves(FEAT, words, onsets, min_occ):
    # force numeric times; non-numeric → NaN
    t = pd.to_numeric(pd.Series(onsets), errors='coerce').to_numpy()
    df = pd.DataFrame({'w': words, 't': t, 'i': np.arange(len(words))}) \
           .sort_values('t', kind='mergesort')
    groups = df.groupby('w')[['i','t']].apply(lambda g: (g['i'].to_list(), g['t'].to_list()))

    keep, S, E, Ts, Te = [], [], [], [], []
    for w, (idxs, ts) in groups.items():
        if len(idxs) < min_occ: 
            continue
        mid = len(idxs)//2
        first, last = idxs[:mid], idxs[mid:]
        tf, tl = np.asarray(ts[:mid], float), np.asarray(ts[mid:], float)
        print(f"\nWord={w}, n={len(idxs)}")
        print(" first times:", tf[:10], "...")   # show first few
        print(" last times:", tl[:10], "...")

        # nan-safe means
        tf_mean = np.nanmean(tf) if tf.size else np.nan
        tl_mean = np.nanmean(tl) if tl.size else np.nan
        if np.isnan(tf_mean) or np.isnan(tl_mean):
            # skip words where either half has no valid time
            continue

        S.append(FEAT[first].mean(0)); E.append(FEAT[last].mean(0)); keep.append(w)
        Ts.append(tf_mean); Te.append(tl_mean)

    if not keep:
        d = FEAT.shape[1]
        return np.array([]), np.empty((0,d)), np.empty((0,d)), np.array([]), np.array([])
    return np.asarray(keep), np.vstack(S), np.vstack(E), np.asarray(Ts), np.asarray(Te)

def compute_space(df, FEAT, *, word_col='word', onset_col='adjusted_onset',
                  min_occ=10, remove_global_mean=False, normalize=None, plot=True):
    FEAT = _to_2d(FEAT)
    words = df[word_col].to_numpy()
    onsets = df[onset_col].to_numpy()
    X = FEAT - FEAT.mean(0, keepdims=True) if remove_global_mean else FEAT
    keep, S, E, Ts, Te = _avg_halves(X, words, onsets, min_occ)
    if normalize == 'unit':
        S, E = unit_norm(S), unit_norm(E)
    if keep.size == 0:
        return Space(keep, S, E, np.empty((0,0)), np.empty((0,0)), np.empty((0,0)), np.array([]))

    start_rsm = _rsm(S); end_rsm = _rsm(E)
    cross = _rowwise_corr(S, E); diag = np.diag(cross).copy()
    gaps = Te - Ts if (Ts.size and Te.size) else np.array([])
    return Space(keep, S, E, start_rsm, end_rsm, cross, diag, Ts, Te, gaps)


def pool_aligned(elems, how='intersection'):
    """
    elems: list of (words, S, E) per electrode, where S/E are (n_i,d).
    Align by word, nanmean across electrodes.
    """
    if not elems: return Space(np.array([]),*(np.empty((0,0)),)*5, np.array([]))
    sets = [set(map(str, w)) for (w,_,_) in elems]
    vocab = sorted(set.intersection(*sets) if how=='intersection' else set.union(*sets))
    if not vocab: return Space(np.array([]),*(np.empty((0,0)),)*5, np.array([]))
    idx = {w:i for i,w in enumerate(vocab)}
    d = elems[0][1].shape[1]; m = len(elems); n = len(vocab)
    S_stack = np.full((m,n,d), np.nan); E_stack = np.full((m,n,d), np.nan)
    for e, (w,S,E) in enumerate(elems):
        w = list(map(str, w)); pos = [idx.get(x) for x in w]
        keep = [p is not None for p in pos]
        if any(keep):
            posk = np.array([p for p in pos if p is not None])
            S_stack[e,posk,:] = S[np.where(keep)[0],:]
            E_stack[e,posk,:] = E[np.where(keep)[0],:]
    S = np.nanmean(S_stack,0); E = np.nanmean(E_stack,0)
    valid = ~(np.isnan(S).all(1) | np.isnan(E).all(1))
    words = np.asarray(vocab)[valid]; S=S[valid]; E=E[valid]
    start_rsm = _rsm(S); end_rsm = _rsm(E); cross = _rowwise_corr(S,E); diag = np.diag(cross).copy()
    return Space(words, S, E, start_rsm, end_rsm, cross, diag)

def second_order_corr(A, B):
    if A.shape != B.shape or A.ndim!=2 or A.shape[0]!=A.shape[1]: return np.nan
    iu = np.triu_indices_from(A, k=1)
    a = A[iu] - A[iu].mean(); b = B[iu] - B[iu].mean()
    den = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)
    return float((a @ b) / den)

def unit_norm(X):
    """Normalize each row (token embedding) to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms
