import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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
    plt.tight_layout()
    plt.savefig(outpath, dpi=180); plt.close()

def _maybe_save(outpath):
    """Internal: save current Matplotlib figure if a path is provided, then close.

    This helper previously was referenced but not defined; adding it here keeps
    style consistent and avoids NameError when calling functions using it.
    """
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=180)
    plt.close()

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
    plt.tight_layout()
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
    plt.tight_layout()
    plt.savefig(outpath, dpi=180); plt.close()

def plot_second_order_corr_hist(
    A, B, label, outdir=None, bins=40, null_corrs=None, fisher=False, B_boot=2000, seed=0):
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

def plot_start_end_heatmap(space, label, outpath=None, clip=(-1,1)):
    """
    Heat map of corr(start_i, end_j) from space.cross_rsm,
    ordered by diagonal (self) correlation.
    """
    C = np.asarray(getattr(space, "cross_rsm", np.zeros((0,0))), float)
    if C.ndim != 2 or C.shape[0] == 0 or C.shape[0] != C.shape[1]:
        return
    diag = np.diag(C).copy()
    order = np.argsort(-diag)
    M = C[order][:, order]

    words = getattr(space, "words", None)
    if words is not None and len(words) == len(order):
        words = np.asarray(words)[order]
    else:
        words = None
    if outpath and words is not None:
        txtpath = outpath.replace(".png", "_words.txt")
        with open(txtpath, "w", encoding="utf-8") as f:
            for w in words:
                f.write(f"{w}\n")

    plt.figure(figsize=(6,5))
    im = plt.imshow(M, vmin=clip[0], vmax=clip[1], cmap="coolwarm", interpolation="nearest")
    plt.colorbar(im).set_label("corr(start_i, end_j)")
    plt.plot(np.arange(M.shape[0]), np.arange(M.shape[0]),
             ls="none", marker=".", ms=2, c="k", alpha=0.6)

    plt.title(f"{label} – start×end heatmap (ordered)")
    plt.tight_layout()
    _maybe_save(outpath)

    return {"order": order, "diag": diag}

def plot_time_gap_hist(space, label, outpath=None, bins=40, show_stats=True, units='samples'):
    """Plot histogram of time differences between start and end vectors.

    A Space may contain either:
      1. precomputed `time_gaps` (space.time_gaps), OR
      2. `start_times` and `end_times` from which gaps = end - start.

    Parameters
    ----------
    space : Space-like
        Object with attributes (time_gaps) OR (start_times & end_times).
    label : str
        Identifier used in plot title / filename.
    outpath : str | None
        If provided, plot is saved to this path; directories created as needed.
    bins : int
        Number of histogram bins.
    show_stats : bool
        Overlay vertical lines for mean & median.

    Returns
    -------
    dict with keys: n, mean, median (all np.nan if no data)
    """
    gaps = getattr(space, 'time_gaps', None)
    if gaps is None or (isinstance(gaps, np.ndarray) and gaps.size == 0):
        st = getattr(space, 'start_times', None)
        et = getattr(space, 'end_times', None)
        if st is not None and et is not None and len(st) == len(et) and len(st):
            try:
                gaps = np.asarray(et) - np.asarray(st)
            except Exception:
                gaps = None
    if gaps is None:
        return {'n': 0, 'mean': np.nan, 'median': np.nan}
    gaps = np.asarray(gaps, float)
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size == 0:
        return {'n': 0, 'mean': np.nan, 'median': np.nan}

    # convert units if requested
    # base units = samples at 512 Hz
    factor = 1.0
    xlab_unit = 'samples'
    if units == 'seconds':
        factor = 1.0 / 512.0
        xlab_unit = 'seconds'
    elif units == 'hours':
        factor = 1.0 / 512.0 / 3600.0
        xlab_unit = 'hours'
    elif units == 'minutes':
        factor = 1.0 / 512.0 / 60.0
        xlab_unit = 'minutes'
    scaled = gaps * factor

    gmean = float(np.nanmean(scaled))
    gmed  = float(np.nanmedian(scaled))

    plt.figure(figsize=(6,4), dpi=180)
    plt.hist(scaled, bins=bins, edgecolor='white')
    plt.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    if show_stats:
        plt.axvline(gmean, color='k', linestyle='--', linewidth=1.2, label=f"mean={gmean:.3g}")
        plt.axvline(gmed, color='r', linestyle=':', linewidth=1.2, label=f"median={gmed:.3g}")
        plt.legend(fontsize=9)
    plt.title(f"{label} – time gap (end − start)")
    plt.xlabel(f"time gap ({xlab_unit})")
    plt.ylabel("count")
    plt.tight_layout()
    _maybe_save(outpath)
    return {'n': int(gaps.size), 'mean': gmean, 'median': gmed, 'units': xlab_unit}

def plot_first_vs_second_order(diag_mean, second_order_r, label, outpath, ylim=(-1,1)):
    """Side-by-side bar plot: first-order (mean diag corr) vs second-order (RSM corr).

    Parameters
    ----------
    diag_mean : float | nan
        Mean of diagonal of cross start-end correlation matrix.
    second_order_r : float | nan
        Second-order correlation between start and end RSMs.
    label : str
        Identifier for title and filename.
    outpath : str
        File path to save figure.
    ylim : tuple
        y-axis limits.
    """
    if not np.isfinite(diag_mean) and not np.isfinite(second_order_r):
        return
    vals = [diag_mean, second_order_r]
    labs = ["first-order (diag mean)", "second-order (RSM r)"]
    plt.figure(figsize=(6.5,4), dpi=180)
    xs = np.arange(len(vals))
    plt.bar(xs, vals, color=['#4c72b0','#dd8452'])
    plt.xticks(xs, labs, rotation=15, ha='right')
    if ylim:
        plt.ylim(*ylim)
    plt.ylabel("correlation")
    plt.title(f"{label} – first vs second order")
    plt.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=0.6)
    plt.tight_layout()
    _maybe_save(outpath)

def plot_time_span_hist(space, label, outpath=None, bins=40, units='hours'):
    """Histogram of real-time span (latest - earliest onset) per word.

    Uses Space.time_spans if present (in sample units @512Hz). Converts to desired units.
    units: 'samples' | 'seconds' | 'minutes' | 'hours'.
    """
    spans = getattr(space, 'time_spans', None)
    if spans is None or (isinstance(spans, np.ndarray) and spans.size == 0):
        return
    spans = np.asarray(spans, float)
    spans = spans[np.isfinite(spans) & (spans >= 0)]
    if spans.size == 0:
        return
    factor = 1.0; xlab = 'samples'
    if units == 'seconds':
        factor = 1/512.0; xlab='seconds'
    elif units == 'minutes':
        factor = 1/512.0/60.0; xlab='minutes'
    elif units == 'hours':
        factor = 1/512.0/3600.0; xlab='hours'
    scaled = spans * factor
    plt.figure(figsize=(6,4), dpi=180)
    plt.hist(scaled, bins=bins, edgecolor='white')
    plt.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    plt.title(f"{label} – word real-time span")
    plt.xlabel(f"span ({xlab})")
    plt.ylabel('count')
    plt.tight_layout()
    _maybe_save(outpath)
