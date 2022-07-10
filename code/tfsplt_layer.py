import glob
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable


sid = 676
modes = ['comp','prod']
conditions = ['all','correct','incorrect'] # gpt2
conditions = ['all','flip'] # blenderbot

layer_num = 16
layers = np.arange(1, layer_num+1)
ticks = [1, layer_num/2, layer_num]

lags =  np.arange(-10000, 10001, 25)

has_ctx = False # is context length considered
sig_elec = True # only run sig elecs
top_format = f'results/tfs/bbot-layers-{str(sid)}/' # parent folder
pdfname = f'results/figures/{str(sid)}_ericplots.pdf'
if sig_elec:
    pdfname = f'results/figures/{str(sid)}_ericplots_sig.pdf'

# Formats
formats = {}
for condition in conditions:
    format = glob.glob(top_format + f'*-{condition}-*') # all children with the formats
    if condition == 'all':
        format = [ind_format for ind_format in format if 'flip' not in ind_format] # blenderbot
    formats[condition] = format
    # formats = [format for format in formats if '48' in format or '47' in format] # for testing

# Sig Elecs
sigelecs = {}
if sig_elec:
    for mode in modes:
        sig_file_name = f'tfs-sig-file-{str(sid)}-sig-1.0-{mode}.csv'
        sig_file = pd.read_csv('data/' + sig_file_name)
        sig_file['sid_electrode'] = sig_file['subject'].astype(str) + '_' + sig_file['electrode']
        elecs = sig_file['sid_electrode'].tolist()
        sigelecs[mode] = set(elecs)

# Aggregate Data
def aggregate_data(dir_path, mode, condition):
    files = glob.glob(dir_path + f'/*/*_{mode}.csv')
    layer_idx = dir_path.rfind('-')
    layer = dir_path[(layer_idx + 1):]
    if layer.isdigit():
        layer = int(layer)
        print('cond:', condition, ' mode:', mode, ' layer:', layer)
        subdata = []
        for resultfn in files:
            elec = os.path.basename(resultfn).replace('.csv', '')[:-5]
            if len(sigelecs) and elec not in sigelecs[mode]:
                continue
            df = pd.read_csv(resultfn, header=None)
            subdata.append(df)
        subdata = pd.concat(subdata)
        subdata_ave = subdata.describe().loc[['mean'],]
        subdata_ave = subdata_ave.assign(layer=layer,mode=mode,condition=condition)
        return subdata_ave
    else:
        return None

def aggregate_data_ctx(dir_path, mode):
    files = glob.glob(dir_path + f'/*/*_{mode}.csv')
    layer_idx = dir_path.rfind('-')
    layer = dir_path[(layer_idx + 1):]
    partial_format = dir_path[:layer_idx]
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
            df.insert(0, 'mode', mode)
            subdata.append(df)
        subdata = pd.concat(subdata)
        return subdata.describe().loc[['mean'],]
    else:
        return None

print('Aggregating Data')
data = []
p = Pool(10)
if has_ctx:
        for mode in modes:
            for result in p.map(partial(aggregate_data_ctx, mode=mode), formats):
                data.append(result)
else:
    for condition in conditions:
        for mode in modes:
            for result in p.map(partial(aggregate_data, mode=mode, condition=condition), formats[condition]):
                data.append(result)


print('Organizing Data')
df = pd.concat(data)
assert len(df) == layer_num * len(conditions) * len(modes)
if has_ctx:
    df = df.sort_values(by=['mode','ctx','layer']) # sort by ctx and layer
    df.set_index(['mode','ctx','layer'], inplace = True)
else:
    df = df.sort_values(by=['condition','mode','layer']) # sort by layer
    df.set_index(['condition','mode','layer'], inplace = True)


print('Plotting')
pdf = PdfPages(pdfname)
cmap = plt.cm.get_cmap('jet')
marker_diff = 0.01

fig, axes = plt.subplots(len(conditions),len(modes),figsize=(18,5*len(conditions)))
for i, condition in enumerate(conditions):
    for j, mode in enumerate(modes):
        ax = axes[i,j]
        encrs = df.loc[(condition, mode), :]
        max_lags = encrs.idxmax(axis=1)
        yheights = np.ones(encrs.shape[-1]) * 1.01 + marker_diff
        encrs = encrs.divide(encrs.max(axis=1),axis=0)
        for layer in layers:
            ax.plot(lags, encrs.loc[layer], c=cmap(layer/layer_num), zorder=-1, lw=0.5)
            maxlag = max_lags[layer]
            ax.scatter(lags[maxlag], yheights[maxlag], marker='o', color=cmap(layer/layer_num))
            yheights[maxlag] += marker_diff
        ax.set(ylim=(0.8, 1.2), xlim=(-2000, 1000))
        if i == 0:
            ax.title.set_text(mode)
        if j == 0:
            ax.set_ylabel(condition)
        if j == 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='4%', pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax = cax,orientation="vertical")
            # cbar = fig.colorbar(sm, ax=ax)
            cbar.set_ticks([0,0.25, 0.5, 0.75,1])
            cbar.set_ticklabels([1, int(layer_num/4), int(layer_num/2),int(3*layer_num/4), layer_num])

pdf.savefig(fig)
plt.close()

colors = ('blue','red','green','black')
fig, axes = plt.subplots(len(conditions),len(modes), figsize=(18,5*len(conditions)))
for i, condition in enumerate(conditions):
    for j, mode in enumerate(modes):
        ax = axes[i,j]
        encrs = df.loc[(condition, mode), :]
        max_time = lags[encrs.idxmax(axis=1)]  # find the lag for the maximum 
        ax.scatter(layers, max_time, c=colors[j])
        # Fit line through scatter
        linefit = stats.linregress(layers, max_time)
        ax.plot(layers, layers*linefit.slope + linefit.intercept, ls='--', c=colors[j])
        ax.set(ylim=(-2000,1000))
        ax.text(3, ax.get_ylim()[1]-200, f'r={linefit.rvalue:.2f} p={linefit.pvalue:.2f}')
        if i == 0:
            ax.set_title(mode)
        if j == 0:
            ax.set_ylabel(condition)
pdf.savefig(fig)
plt.close()


fig, axes = plt.subplots(len(conditions),len(modes), figsize=(18,5*len(conditions)))

for i, condition in enumerate(conditions):
    for j, mode in enumerate(modes):
        ax = axes[i,j]
        chosen_lag_idx = [idx for idx, element in enumerate(lags) if element <=2000 and element >= -2000]
        encrs = df.loc[(condition, mode), chosen_lag_idx]
        mean = encrs.mean(axis=1)
        errs = encrs.std(axis=1)**2
        
        ax.bar(layers,mean,yerr=errs,align='center',alpha=0.5,ecolor='black',capsize=0)
        ax.set(xlim=(0.5, layer_num+0.5), xticks=ticks, xticklabels=ticks)
        if i == 0:
            ax.set_title(f'{mode}-mean(2s)')
        if j == 0:
            ax.set_ylabel(condition)

pdf.savefig(fig)
plt.close()


fig, axes = plt.subplots(len(conditions),len(modes), figsize=(18,5*len(conditions)))

for i, condition in enumerate(conditions):
    for j, mode in enumerate(modes):
        ax = axes[i,j]
        encrs = df.loc[(condition, mode),:]
        max = encrs.max(axis=1)
        
        ax.scatter(layers,max)
        ax.set(xlim=(0.5, layer_num+0.5), xticks=ticks, xticklabels=ticks)
        if i == 0:
            ax.set_title(f'{mode}-max')
        if j == 0:
            ax.set_ylabel(condition)

pdf.savefig(fig)
plt.close()

pdf.close()


# plot_max = df.max(axis=1).unstack(0).sort_values(by='layer',ascending=False)
# plot_idxmax = df.idxmax(axis=1).unstack(0)
# idx_vals = list(range(-10000,10025,25))

# def get_idx_val(idx):
#     return idx_vals[idx]

# plot_idxmax = plot_idxmax.apply(np.vectorize(get_idx_val))
# sns.dark_palette("#69d", reverse=True, as_cmap=True)


# print('Plotting')
# pdf = PdfPages('results/figures/layer_ctx.pdf')

# fig, ax = plt.subplots(figsize=(15,6))
# ax = sns.heatmap(plot_max, cmap='Blues')
# pdf.savefig(fig)
# plt.close()

# fig, ax = plt.subplots(figsize=(15,6))
# ax = sns.heatmap(plot_idxmax, cmap='Blues')
# pdf.savefig(fig)
# plt.close()

# fig, ax = plt.subplots(figsize=(15,6))
# ax = sns.scatterplot(x=plot_max.index, y=plot_max)
# pdf.savefig(fig)

# # plot_idxmax_melt = plot_idxmax.melt('layer',var_name='mode')
# fig, ax = plt.subplots(figsize=(15,6))
# ax = sns.scatterplot(x=plot_idxmax.index, y=plot_idxmax)
# pdf.savefig(fig)
# pdf.close()
