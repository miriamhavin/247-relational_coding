# tfsdrift/plots.py
import os, numpy as np, matplotlib.pyplot as plt

def plot_first_order_hist(space, title, outpath=None, bins=30):
    d = space.diag
    if d.size==0: return
    plt.figure(figsize=(5,4)); plt.hist(d, bins=bins)
    plt.title(f"{title} — per-word start↔end (n={len(d)})")
    plt.xlabel("Correlation"); plt.ylabel("Count")
    if outpath: os.makedirs(os.path.dirname(outpath), exist_ok=True); plt.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close()

def plot_mat(M, title, outpath, vmin=-1.0, vmax=1.0):
    if M.size==0: return
    plt.figure(figsize=(6,5)); plt.imshow(M, vmin=vmin, vmax=vmax); plt.colorbar(); plt.title(title)
    if outpath: os.makedirs(os.path.dirname(outpath), exist_ok=True); plt.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close()

def plot_all(space, label, outdir=None):
    base = (lambda s: None) if not outdir else (lambda s: os.path.join(outdir, f"{label}_{s}.png"))
    plot_first_order_hist(space, f"{label} – 1st Order", base("first_order_hist"))
    plot_mat(space.start_rsm, f"{label} – start_rsm", base("start_rsm"))
    plot_mat(space.end_rsm,   f"{label} – end_rsm",   base("end_rsm"))
    plot_mat(space.cross_rsm, f"{label} – cross_rsm", base("cross_rsm"))
