import glob
import argparse
import os
import pandas as pd
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument("--formats", nargs="+", required=True)
parser.add_argument("--labels", nargs="+",  required=True)
parser.add_argument("--values", nargs="+", type=float, required=True)
parser.add_argument("--keys", nargs="+",  required=True)
parser.add_argument("--sid", type=int, default=625)
parser.add_argument("--sig-elec-file", nargs="+", default=[])
parser.add_argument("--outfile", default='results/figures/tmp.pdf')
args = parser.parse_args()

assert len(args.labels) == len(args.formats)

elecdir = f'/projects/HASSON/247/data/elecimg/{args.sid}/'

# Assign a unique color/style to each label/mode combination
# i.e. gpt2 will always be blue, prod will always be full line
#      glove will always be red, comp will always be dashed
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
styles = ['-', '--', '-.', ':']
cmap = {}  # color map
smap = {}  # style map
for key, color in zip(args.keys, colors):
    for label, style in zip(args.labels, styles):
        if label == 'prod_comp':
            key = "test on " + key
        cmap[(label, key)] = color
        smap[(label, key)] = style

def get_elecbrain(electrode):
    name = electrode.replace('EEG', '').replace('REF', '').replace('\\', '')
    name = name.replace('_', '').replace('GR', 'G')
    imname = elecdir + f'thumb_{name}.png'  # + f'{args.sid}_{name}.png'
    return imname

# Read significant electrode file(s)
sigelecs = {}
if len(args.sig_elec_file) == 1 and len(args.keys) > 1:
    for fname, mode in zip(args.sig_elec_file, args.keys):
        elecs = pd.read_csv('data/' + fname % mode)['electrode'].tolist()
        sigelecs[mode] = set(elecs)
if len(args.sig_elec_file) == len(args.keys):
    for fname, mode in zip(args.sig_elec_file, args.keys):
        elecs = pd.read_csv('data/' + fname)['electrode'].tolist()
        sigelecs[mode] = set(elecs)

print('Aggregating data')
data = []
erp_data = []
for fmt, label in zip(args.formats, args.labels):
    for key in args.keys:
        fname = fmt % key
        files = glob.glob(fname)
        assert len(files) > 0, f"No results found under {fname}"

        for resultfn in files:
            elec = os.path.basename(resultfn).replace('.csv', '')[:-5]
            # Skip electrodes if they're not part of the sig list
            if len(sigelecs) and elec not in sigelecs[key]:
                # print('Skipping', elec)
                continue
            df = pd.read_csv(resultfn, header=None)
            if 'prod-comp' in fmt: # if it is erp data
                df.insert(0, 'mode', "test on " + key)
            else:
                df.insert(0, 'mode', key)
            df.insert(0, 'electrode', elec)
            df.insert(0, 'label', label)
            if 'erp' in fmt: # if it is erp data
                erp_data.append(df)
            else:
                data.append(df)

if not len(data) or not len(erp_data):
# if not len(erp_data):
    print('No data found')
    exit(1)

df1 = pd.concat(data)
df_erp = pd.concat(erp_data)
df1.set_index(['label', 'electrode', 'mode'], inplace=True)
df_erp.set_index(['label', 'electrode', 'mode'], inplace=True)
# lags = list(range(len(df.columns)))
lags = [value / 1000 for value in args.values]
# erp_lag_ws = int(args.window_size / 2 / 1000 * 512)
# lags_erp = list(range(erp_lag_ws*(-1),erp_lag_ws+1))
# lags_erp = [lag / 512 for lag in lags_erp]
# n_av, n_av2, n_df, n_df2 = len(args.values), len(lags_erp), len(df1.columns), len(df_erp.columns)
# n_av, n_df = len(args.values), len(df1.columns)
# assert n_av == n_df, \
#     'args.values length ({n_av}) munst be same size as results ({n_df})'
# assert n_av2 == n_df2, \
#     'erp values length ({n_av2}) must be same size as results ({n_df2})'
df = pd.concat([df_erp,df1],axis=0)
# df = df_erp

print('Plotting')
pdf = PdfPages(args.outfile)

quardra = True # are we plotting quardra

lag_ticks = lags
lag_ticks_out = []
if quardra:
    lag_ticks_out = [12,13,14,15,16,18,20,22,24,26,28] # quardra
    lag_tick_locations = [-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]
    lag_ticklabels = [-300,-250,-200,-150,-120,-90,-60,-40,-20,-10,-8,-6,-4,-2,0,2,4,6,8,10,20,40,60,90,120,150,200,250,300]
# lag_ticks_out = [12,13,14,15,16,18,20,22] # triple
for lag in lag_ticks_out:
    lag_ticks.insert(0,lag*-1)
    lag_ticks.append(lag)


# Plot results for each key (i.e. average)
# plot each key/mode in its own subplot
fig, axes = plt.subplots(len(args.labels),1, figsize=(18,5))
for ax, (label, subdf) in zip(axes, df.groupby('label', axis=0)):
    for mode, subsubdf in subdf.groupby('mode', axis=0):
        # if label != 'erp':
        #     subsubdf = subsubdf.iloc[:,0:n_av]
        vals = subsubdf.mean(axis=0)
        err = subsubdf.sem(axis=0)
        key = (label,mode)
        # if label == 'erp':
        #     ax.fill_between(lags_erp, vals - err, vals + err, alpha=0.2, color=cmap[key])
        #     ax.plot(lags_erp, vals, label=f'{mode} ({len(subsubdf)})', color=cmap[key], ls=smap[key])
        # else:
        if quardra:
            ax.set_xticks(lag_tick_locations)
            ax.set_xticklabels(lag_ticklabels)
        ax.fill_between(lag_ticks, vals - err, vals + err, alpha=0.2, color=cmap[key])
        ax.plot(lag_ticks, vals, label=f'{mode} ({len(subsubdf)})', color=cmap[key], ls='solid')
    ax.legend(loc='upper right', frameon=False)
    ax.axhline(0,ls='dashed',alpha=0.3,c='k')
    ax.axvline(0,ls='dashed',alpha=0.3,c='k')
    if label == 'gpt2':
        ax.set(xlabel='Lag (s)', ylabel = '')
    else:
        ax.axes.xaxis.set_visible(False)
    ax.set(title=f'{label} global average')
pdf.savefig(fig)
plt.close()

# # plot all keys together
# fig, ax = plt.subplots()
# for mode, subdf in df.groupby(['label', 'mode'], axis=0):
#     # if mode in [('bbot_dec', 'comp'), ('bbot_enc', 'prod')]:
#     #     continue
#     vals = subdf.mean(axis=0)
#     err = subdf.sem(axis=0)
#     ax.fill_between(lags, vals - err, vals + err, alpha=0.2, color=cmap[mode])
#     label = '-'.join(mode)
#     ax.plot(lags, vals, label=f'{label} ({len(subdf)})', color=cmap[mode], ls=smap[mode])
# ax.legend(loc='upper right', frameon=False)
# ax.set(xlabel='Lag (s)', ylabel='Correlation (r)', title='Global average')
# pdf.savefig(fig)
# plt.close()

# Plot each electrode separately
vmax, vmin = df1.max().max(), df1.min().min()
erpmax, erpmin = df_erp.max().max(), df_erp.min().min()
for electrode, subdf in df.groupby('electrode', axis=0):
    fig, axes = plt.subplots(len(args.labels),1, figsize=(18,5))
    for ax, (label, subsubdf) in zip(axes, subdf.groupby('label')):
        if label != 'erp':
            subsubdf = subsubdf
        elif label == 'erp' and subsubdf.shape[0] == 2: # if both prod and comp significant
            end_point = len(subsubdf.columns)
            mid_point = int(end_point / 2)
            # corr_all, _ = pearsonr(subsubdf.iloc[0], subsubdf.iloc[1])
            # corr_bonset, _ = pearsonr(subsubdf.iloc[0,0:mid_point+1], subsubdf.iloc[1,0:mid_point+1])
            # corr_aonset, _ = pearsonr(subsubdf.iloc[0,mid_point:end_point], subsubdf.iloc[1,mid_point:end_point])
            # corrs = "cor: " + str(round(corr_all,3)) + "\ncorr -onset: " + str(round(corr_bonset,3)) + "\ncorr +onset: " + str(round(corr_aonset,3))
            # ax.text(0.05,0.82,corrs,fontsize = 10,ha='left',va='center',transform=ax.transAxes)
        for row, values in subsubdf.iterrows():
            # print(mode, label, type(values))
            # print(subsubdf)
            mode = row[2]
            key = (label, mode)
            # if label == 'erp':
            #     ax.plot(lags_erp, values, label=mode, color=cmap[key], ls=smap[key])
            # else:
            if quardra:
                ax.set_xticks(lag_tick_locations)
                ax.set_xticklabels(lag_ticklabels)
            ax.plot(lag_ticks, values, label=mode, color=cmap[key], ls='solid')
        ax.legend(loc='upper left', frameon=False)
        if label == 'erp':
            ax.set_ylim(erpmin - 0.05, erpmax + 0.05)  # erp limit
        else:
            ax.set_ylim(vmin - 0.05, vmax + .05)  # cor limit
        ax.axhline(0,ls='dashed',alpha=0.3,c='k')
        ax.axvline(0,ls='dashed',alpha=0.3,c='k')
        if label == 'gpt2':
            ax.set(xlabel='Lag (s)', ylabel = '')
        else:
            ax.axes.xaxis.set_visible(False)
        ax.set(title=f'{electrode} {label}')
    imname = get_elecbrain(electrode)
    if os.path.isfile(imname):
        arr_image = plt.imread(imname, format='png')
        fig.figimage(arr_image,
                     fig.bbox.xmax - arr_image.shape[1],
                     fig.bbox.ymax - arr_image.shape[0], zorder=5)
    pdf.savefig(fig)
    plt.close()

pdf.close()
