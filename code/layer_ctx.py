import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

data = []
top_format = 'results/podcast-ctx-layer-mwf5/' # parent folder
formats = glob.glob(top_format + '*') # all layers and index result folder
# formats = [format for format in formats if '48' in format or '47' in format]

print('Aggregating Data')
for format in formats:
    files = glob.glob(format + '/*/*_comp.csv')
    layer_idx = format.rfind('-')
    layer = format[(layer_idx + 1):]
    partial_format = format[:layer_idx]
    ctx_len = partial_format[(partial_format.rfind('-') + 1):]
    if layer.isdigit() and ctx_len.isdigit():
        layer = int(layer)
        ctx_len = int(ctx_len)
        print('ctx:', ctx_len, ' layer:', layer)
        subdata = []
        for resultfn in files:
            df = pd.read_csv(resultfn, header=None)
            df.insert(0, 'layer', layer)
            df.insert(0, 'ctx', ctx_len)
            subdata.append(df)
        subdata = pd.concat(subdata)
        data.append(subdata.describe().loc[['mean'],])


print('Organizing Data')
df = pd.concat(data) # should be length: 48 * 7 = 336
df[["layer", "ctx"]] = df[["layer", "ctx"]].astype(int) # change to int
df = df.sort_values(by=['ctx','layer']) # sort by ctx and layer
df.set_index(['ctx','layer'], inplace = True)

# df.idxmax(axis=1)
# df.max(axis=1)

plot_max = df.max(axis=1).unstack(0)
plot_idxmax = df.idxmax(axis=1).unstack(0)
idx_vals = list(range(-500,505,5))

def get_idx_val(idx):
    return idx_vals[idx]

plot_idxmax = plot_idxmax.apply(np.vectorize(get_idx_val))
# sns.dark_palette("#69d", reverse=True, as_cmap=True)

print('Plotting')
pdf = PdfPages('results/figures/layer_ctx.pdf')

fig, ax = plt.subplots(figsize=(15,6))
ax = sns.heatmap(plot_max, cmap='Blues')
pdf.savefig(fig)
plt.close()

fig, ax = plt.subplots(figsize=(15,6))
ax = sns.heatmap(plot_idxmax, cmap='Blues')
pdf.savefig(fig)
plt.close()

pdf.close()
