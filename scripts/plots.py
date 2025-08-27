# tfsdrift/plots.py
import os
import numpy as np
import matplotlib.pyplot as plt


# -------------------- basic helpers --------------------
def _maybe_save(outpath):
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


# -------------------- first-order & heatmaps --------------------
def plot_first_order_hist(space, title, outpath=None, bins=30):
    d = getattr(space, "diag", np.array([]))
    if d.size == 0:
        return
    plt.figure(figsize=(5, 4))
    plt.hist(d, bins=bins)
    plt.title(f"{title} — per-word start↔end (n={len(d)})")
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    _maybe_save(outpath)

def plot_mat(M, title, outpath, vmin=-1.0, vmax=1.0):
    M = np.asarray(M)
    if M.size == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(M, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    _maybe_save(outpath)


# -------------------- Same≠Different: delta histogram --------------------
def plot_delta_hist(cross, label, outdir=None, bins=30):
    cross = np.asarray(cross)
    if cross.size == 0:
        return
    n = cross.shape[0]
    diag = np.diag(cross)
    deltas = []
    for i in range(n):
        others = np.delete(cross[i], i)
        deltas.append(diag[i] - others.mean())
    plt.figure(figsize=(5, 4))
    plt.hist(deltas, bins=bins)
    plt.title(f"{label} – Δ(self−others)")
    plt.xlabel("Δ")
    plt.ylabel("Count")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{label}_delta_hist.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.close()


# -------------------- Permutation null for meanΔ --------------------
def plot_perm_null(stat_values, observed, label, outdir=None):
    if stat_values is None or len(stat_values) == 0:
        return
    plt.figure(figsize=(5, 4))
    plt.hist(stat_values, bins=40)
    plt.axvline(observed, lw=2)
    plt.title(f"{label} – permutation null (meanΔ)")
    plt.xlabel("meanΔ under null")
    plt.ylabel("Count")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{label}_perm_meanDelta_null.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.close()


# -------------------- Second-order: upper-triangle scatter --------------------
def plot_rsm_scatter(A, B, label, outdir=None, suffix="start_vs_end"):
    A = np.asarray(A); B = np.asarray(B)
    if A.size == 0 or B.size == 0 or A.shape != B.shape:
        return
    iu = np.triu_indices_from(A, k=1)
    x, y = A[iu], B[iu]
    r = np.corrcoef(x, y)[0, 1] if x.size and y.size else np.nan
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=8, alpha=0.5)
    plt.xlabel("RSM A (upper)")
    plt.ylabel("RSM B (upper)")
    plt.title(f"{label} – RSM scatter (r={r:.3f})")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{label}_rsm_scatter_{suffix}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.close()


# -------------------- RSM panel: start / end / predicted / residual --------------------
def plot_rsms_panel(start_rsm, end_rsm, end_pred_rsm, end_resid_rsm, label, outdir=None):
    mats = [start_rsm, end_rsm, end_pred_rsm, end_resid_rsm]
    if any(m is None or np.size(m) == 0 for m in mats):
        return
    plt.figure(figsize=(16, 4))
    titles = ["start", "end", "end_pred", "end_resid"]
    # individual subplots to control colorbar across all
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    im = None
    for ax, M, t in zip(axes, mats, titles):
        im = ax.imshow(M, vmin=-1, vmax=1)
        ax.set_title(t)
        ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"{label}_rsm_panel.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)



# -------------------- Master plot wrapper --------------------
def plot_all(space, label, outdir=None, *, B_perm_cols=2000, seed=42):
    """
    Generate all default plots for a Space:
      - first-order histogram
      - start/end/cross RSM heatmaps
      - same≠different delta histogram
      - second-order scatter (start vs end)
      - prediction-based panel + scatters (if Ehat/resid RSMs attached)
      - identifiability rank histogram & CMC
    """
    base = (lambda s: None) if not outdir else (lambda s: os.path.join(outdir, f"{label}_{s}.png"))

    # 1) first-order + heatmaps
    plot_first_order_hist(space, f"{label} – 1st Order", base("first_order_hist"))
    plot_mat(getattr(space, "start_rsm", np.zeros((0, 0))), f"{label} – start_rsm", base("start_rsm"))
    plot_mat(getattr(space, "end_rsm",   np.zeros((0, 0))), f"{label} – end_rsm",   base("end_rsm"))
    plot_mat(getattr(space, "cross_rsm", np.zeros((0, 0))), f"{label} – cross_rsm", base("cross_rsm"))

    # 2) Same≠Different
    if getattr(space, "cross_rsm", np.zeros((0, 0))).size:
        plot_delta_hist(space.cross_rsm, label, outdir)

    # 3) Second-order scatter
    if getattr(space, "start_rsm", np.zeros((0, 0))).size and getattr(space, "end_rsm", np.zeros((0, 0))).size:
        plot_rsm_scatter(space.start_rsm, space.end_rsm, label, outdir, suffix="start_vs_end")

    # 4) Prediction-based (expect attributes set by report_space)
    # If you attach via: setattr(space, "end_pred_rsm", end_pred_rsm) and "...end_resid_rsm", they’ll render here.
    if hasattr(space, "end_pred_rsm") and hasattr(space, "end_resid_rsm"):
        if getattr(space, "end_pred_rsm").size and getattr(space, "end_resid_rsm").size:
            plot_rsms_panel(space.start_rsm, space.end_rsm, space.end_pred_rsm, space.end_resid_rsm, label, outdir)
            plot_rsm_scatter(space.start_rsm, space.end_pred_rsm, label, outdir, suffix="start_vs_endPred")
            plot_rsm_scatter(space.start_rsm, space.end_resid_rsm, label, outdir, suffix="start_vs_endResid")

