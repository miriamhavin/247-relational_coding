import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- inline plotting helpers (readable & correct) ----------
import math

PLOT_DPI   = 180
FONTSIZE   = 10
TITLE_SIZE = 12


def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def setup_fig(w=6, h=4):
    plt.rcParams.update({
        "figure.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    return plt.figure(figsize=(w, h))

def sanitize_corr(values):
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    return vals

def hist(values, title, xlabel, outpath, bins=40, xlim=None):
    vals = sanitize_corr(values)
    if vals.size == 0: return
    setup_fig(6,4)
    plt.hist(vals, bins=bins, edgecolor='white')
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=180); plt.close()

def null_hist(null_values, observed_value, title, xlabel, outpath, bins=40, xlim=None):
    nulls = sanitize_corr(null_values)
    if nulls.size == 0: return
    setup_fig(6,4)
    plt.hist(nulls, bins=bins, edgecolor='white')
    if np.isfinite(observed_value):
        plt.axvline(float(observed_value), linestyle="--", linewidth=1.5)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=180); plt.close()

def group_mean_by_word(words, per_occ_values):
    """Return per-word MEAN values (correct for 'across words' histogram)."""
    words = np.asarray(words, dtype=object)
    vals  = np.asarray(per_occ_values, float)
    ok = np.isfinite(vals)
    if ok.sum() == 0: 
        return np.array([])
    df = pd.DataFrame({"word": words[ok], "v": vals[ok]})
    agg = df.groupby("word", sort=False)["v"].mean().to_numpy()
    return agg

def occurrence_matrix_plot(
    df, FEAT, *, word_col='word', min_occ=50, max_instances=3000,
    outpath=None, figsize=(6,6), dpi=180, vmax=1.0,
    title=None, order_by='word_then_index', time_col=None,
):
    if df is None or FEAT is None:  # basic checks
        return {"C": None, "cuts": None, "order": np.array([], int), "n_kept": 0,
                "words_sequence": [], "word_blocks_df": pd.DataFrame(), "n_chunks": 0, "chunks": []}
    words = df[word_col].to_numpy(); Xall = np.asarray(FEAT, float)
    if Xall.ndim != 2 or len(words) != Xall.shape[0]:
        return {"C": None, "cuts": None, "order": np.array([], int), "n_kept": 0,
                "words_sequence": [], "word_blocks_df": pd.DataFrame(), "n_chunks": 0, "chunks": []}

    # keep words with >= min_occ and order
    keeps = set(pd.Series(words).value_counts()[lambda s: s>=min_occ].index)
    idx = [i for i,w in enumerate(words) if w in keeps]
    if not idx:
        return {"C": None, "cuts": None, "order": np.array([], int), "n_kept": 0,
                "words_sequence": [], "word_blocks_df": pd.DataFrame(), "n_chunks": 0, "chunks": []}
    if order_by=='time' and time_col and time_col in df.columns:
        order = np.array(sorted(idx, key=lambda i:(words[i], df.iloc[i][time_col])), int)
    else:
        order = np.array(sorted(idx, key=lambda i:(words[i], i)), int)

    words_seq_all = [words[i] for i in order]
    s_all = pd.Series(words_seq_all)
    gb = s_all.groupby(s_all, sort=False)
    block_words_all = list(gb.size().index)
    block_sizes_all = gb.size().to_list()

    # split into chunks without cutting a word block
    chunks, cur, acc = [], 0, 0
    for b in block_sizes_all:
        if acc and acc + b > max_instances:
            chunks.append((cur, cur+acc)); cur += acc; acc = 0
        acc += b
    if acc: chunks.append((cur, cur+acc))
    n_chunks = len(chunks)

    def process_chunk(k, start, end):
        sub_idx = order[start:end]; X = Xall[sub_idx]
        if X.shape[0] < 2:
            return {"chunk_idx": k, "chunk_slice": (start,end), "C": None, "cuts": np.array([],int),
                    "order": sub_idx, "n_kept": int(end-start),
                    "words_sequence": [], "word_blocks_df": pd.DataFrame()}
        X = X - X.mean(0, keepdims=True)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        C = X @ X.T

        ws = words_seq_all[start:end]; s = pd.Series(ws); gg = s.groupby(s, sort=False).size()
        block_sizes = gg.to_list(); block_words = list(gg.index)
        cuts = (np.cumsum(block_sizes)[:-1]).astype(int) if len(block_sizes)>1 else np.array([], int)
        word_blocks_df = pd.DataFrame({
            "word": block_words, "count": block_sizes,
            "start": np.r_[0, cuts], "end": np.r_[cuts, len(ws)]
        })

        if outpath:
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            stem, ext = os.path.splitext(outpath); p = f"{stem}_part{k+1}of{n_chunks}{ext or '.png'}"
            plt.figure(figsize=figsize, dpi=dpi)
            im = plt.imshow(C, vmin=-1, vmax=vmax, interpolation="none")
            for c in cuts: plt.axhline(c-0.5, color="k", lw=0.4); plt.axvline(c-0.5, color="k", lw=0.4)
            plt.xticks([]); plt.yticks([])
            plt.title((title or f"Occurrence neural corr (min_occ={min_occ})")+f" — part {k+1}/{n_chunks} (n={end-start})", fontsize=11)
            plt.colorbar(im, shrink=0.8).ax.tick_params(labelsize=9)
            plt.tight_layout(); plt.savefig(p); plt.close()
            with open(f"{stem}_part{k+1}of{n_chunks}_words_sequence.txt","w",encoding="utf-8") as f:
                f.write("\n".join(ws))
            word_blocks_df.to_csv(f"{stem}_part{k+1}of{n_chunks}_word_blocks.csv", index=False)

        return {"chunk_idx": k, "chunk_slice": (start,end), "C": C, "cuts": cuts, "order": sub_idx,
                "n_kept": int(end-start), "words_sequence": ws, "word_blocks_df": word_blocks_df}

    chunk_results = [process_chunk(k, a, b) for k,(a,b) in enumerate(chunks)]

    if n_chunks == 1:  # backward compatible
        r = chunk_results[0].copy(); r.pop("chunk_idx", None); r.pop("chunk_slice", None); return r
    return {
        "C": None, "cuts": None, "order": order, "n_kept": int(order.size),
        "words_sequence": words_seq_all,
        "word_blocks_df": pd.DataFrame({"word": block_words_all, "count": block_sizes_all}),
        "n_chunks": n_chunks, "chunks": chunk_results,
    }

def bars_three(end_val, pred_val, resid_val, title, outpath, ylim=(-1, 1)):
    """Comparison bars (single-space); keep readable and correctly scaled."""
    vals = [end_val, pred_val, resid_val]
    xs   = np.arange(3)
    setup_fig(5,4)
    plt.bar(xs, vals)
    plt.xticks(xs, ["end", "pred", "resid"])
    plt.ylabel("second-order corr (start vs ·)")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=0.6)
    plt.title(title)
    plt.tight_layout(); ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=180); plt.close()

def plot_second_order_corr_hist(
    A, B, label, outdir=None, bins=40, null_corrs=None, fisher=False, B_boot=2000, seed=0
):
    """
    Histogram of *second-order correlations* between two RSMs:
      - If `null_corrs` is provided: plot its histogram and mark observed r with a vertical line.
      - Else: bootstrap the correlation from A/B upper-triangle entries (with replacement)
              to create a sampling histogram, and mark the observed r.

    Args
    ----
    A, B : square np.ndarray, same shape (RSMs)
    label : str for title/filename
    outdir : dir to save PNG (if None, just closes)
    bins : int, histogram bins
    null_corrs : array-like of null correlation values (e.g., Mantel permutations)
    fisher : bool, if True plot on Fisher z scale (z = atanh(r)) for symmetry
    B_boot : int, # bootstraps if null_corrs is None
    seed : int, RNG seed for bootstrap
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1] or A.size == 0:
        return

    iu = np.triu_indices_from(A, k=1)
    x, y = A[iu], B[iu]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return

    # observed second-order correlation
    r_obs = np.corrcoef(np.vstack([x, y]))[0, 1]

    # choose data to histogram
    if null_corrs is not None:
        vals = np.asarray(null_corrs, float)
        vals = vals[np.isfinite(vals)]
        mode = "null"
        # simple two-sided p wrt null if you like (not required for the plot)
        if vals.size:
            p_two = (np.sum(np.abs(vals) >= abs(r_obs)) + 1) / (vals.size + 1)
        else:
            p_two = np.nan
    else:
        # bootstrap the correlation from the upper-triangle entries
        rng = np.random.default_rng(seed)
        n = x.size
        idx = rng.integers(0, n, size=(B_boot, n))
        xb = x[idx]; yb = y[idx]
        # corr for each bootstrap replicate
        xb = xb - xb.mean(axis=1, keepdims=True)
        yb = yb - yb.mean(axis=1, keepdims=True)
        num = np.sum(xb * yb, axis=1)
        den = np.linalg.norm(xb, axis=1) * np.linalg.norm(yb, axis=1) + 1e-12
        vals = num / den
        mode = "bootstrap"
        p_two = np.nan  # not a permutation p-value

    # optional Fisher z transform for symmetry
    if fisher:
        def _z(r): 
            r = np.clip(r, -0.999999, 0.999999)
            return np.arctanh(r)
        vals_plot = _z(vals)
        r_line    = _z(r_obs)
        xlab = "Fisher z(r)"
    else:
        vals_plot = vals
        r_line    = r_obs
        xlab = "second-order correlation r"

    # plot
    plt.figure(figsize=(6,4), dpi=180)
    plt.hist(vals_plot, bins=bins, edgecolor='white')
    plt.axvline(r_line, color='k', linestyle='--', linewidth=1.5, label=f"observed r={r_obs:.3f}")
    plt.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    subtitle = f"{mode} hist (n={len(vals_plot)})"
    if mode == "null" and np.isfinite(p_two):
        subtitle += f", two-sided p≈{p_two:.3g}"
    plt.title(f"{label} – second-order correlation\n{subtitle}")
    plt.xlabel(xlab); plt.ylabel("count")
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{label}_second_order_corr_hist.png"))
    plt.close()

def plot_start_end_corr_matrix(space, outpath, title=None, clip=(-1, 1)):
    """
    Plot start×end Pearson correlation matrix for a Space, ordering both axes
    by the diagonal (per-word start↔end correlation).
    Saves to outpath and returns {'order', 'diag'}.
    """
    C = np.asarray(space.cross_rsm, float)
    if C.ndim != 2 or C.shape[0] == 0 or C.shape[0] != C.shape[1]:
        return None

    diag = np.diag(C).copy()
    order = np.argsort(-diag)  # descending by self-corr

    M = C[order][:, order]
    words = np.asarray(space.words) if getattr(space, "words", None) is not None else None
    words = words[order] if words is not None and len(words) == len(order) else None
    n = M.shape[0]

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig = plt.figure(figsize=(6.2, 5.6), dpi=150)
    ax = plt.gca()
    im = ax.imshow(M, vmin=clip[0], vmax=clip[1], cmap="coolwarm", interpolation="nearest")
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("corr(start_i, end_j)")

    if title:
        ax.set_title(title, fontsize=11, pad=8)

    # light diagonal markers to orient
    ax.plot(np.arange(n), np.arange(n), ls="none", marker=".", ms=2, c="k", alpha=0.6)

    # label words only if small
    if words is not None and n <= 30:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(words, rotation=90, fontsize=7)
        ax.set_yticklabels(words, fontsize=7)
    else:
        ax.set_xticks([]); ax.set_yticks([])

    # tiny inset to show ordered diagonal values (QA)
    inset = fig.add_axes([0.83, 0.11, 0.12, 0.77])
    inset.plot(np.diag(M), np.arange(n), lw=1)
    inset.set_ylim(n-0.5, -0.5)
    inset.set_xlabel("diag corr")
    inset.set_yticks([])
    inset.grid(alpha=0.3, linestyle=":", linewidth=0.6)

    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    return {"order": order, "diag": diag}
