import glob
import argparse
import os
import pandas as pd

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
parser.add_argument("--window-size", type=int, default=4000)
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
        if label != 'erp':
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
            if 'erp' in fmt: # if it is erp data
                df.insert(0, 'mode', key)
            else:
                df.insert(0, 'mode', "test on " + key)
            df.insert(0, 'electrode', elec)
            df.insert(0, 'label', label)
            if 'erp' in fmt: # if it is erp data
                erp_data.append(df)
            else:
                data.append(df)

if not len(data) or not len(erp_data):
    print('No data found')
    exit(1)
df1 = pd.concat(data)
df_erp = pd.concat(erp_data)
df1.set_index(['label', 'electrode', 'mode'], inplace=True)
df_erp.set_index(['label', 'electrode', 'mode'], inplace=True)
# lags = list(range(len(df.columns)))
lags = args.values
erp_lag_ws = int(args.window_size / 2 / 1000 * 512)
lags_erp = list(range(erp_lag_ws*(-1),erp_lag_ws+1))
lags_erp = [lag / 512 * 1000 for lag in lags_erp]
n_av, n_av2, n_df, n_df2 = len(args.values), len(lags_erp), len(df1.columns), len(df_erp.columns)
assert n_av == n_df, \
    'args.values length ({n_av}) munst be same size as results ({n_df})'
assert n_av2 == n_df2, \
    'erp values length ({n_av2}) must be same size as results ({n_df2})'
df = pd.concat([df_erp,df1],axis=0)

print('Plotting')
pdf = PdfPages(args.outfile)

# Plot results for each key (i.e. average)
# plot each key/mode in its own subplot
fig, axes = plt.subplots(1, len(args.labels), figsize=(12, 6))
for ax, (label, subdf) in zip(axes, df.groupby('label', axis=0)):
    for mode, subsubdf in subdf.groupby('mode', axis=0):
        if label != 'erp':
            subsubdf = subsubdf.iloc[:,0:n_av]
        vals = subsubdf.mean(axis=0)
        err = subsubdf.sem(axis=0)
        key = (label,mode)
        if label == 'erp':
            ax.fill_between(lags_erp, vals - err, vals + err, alpha=0.2, color=cmap[key])
            ax.plot(lags_erp, vals, label=f'{mode} ({len(subsubdf)})', color=cmap[key], ls=smap[key])
        else:
            ax.fill_between(lags, vals - err, vals + err, alpha=0.2, color=cmap[key])
            ax.plot(lags, vals, label=f'{mode} ({len(subsubdf)})', color=cmap[key], ls=smap[key])
    ax.set_title(label + ' global average')
    ax.legend(loc='upper right', frameon=False)
    if label == 'erp':
        ax.set(xlabel='Lag (s)', ylabel='Power')
    else:
        ax.set(xlabel='Lag (s)', ylabel='Correlation (r)')
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
erpmax, erpmin = df_erp.max().max(), df.min().min()
for electrode, subdf in df.groupby('electrode', axis=0):
    fig, axes = plt.subplots(1, len(args.keys), figsize=(12, 6))
    for ax, (label, subsubdf) in zip(axes, subdf.groupby('label')):
        for row, values in subsubdf.iterrows():
            # print(mode, label, type(values))
            # print(subsubdf)
            mode = row[2]
            key = (label, mode)
            if label == 'erp':
                ax.plot(lags_erp, values, label=mode, color=cmap[key], ls=smap[key])
            else:
                values = values[0:n_av]
                ax.plot(lags, values, label=mode, color=cmap[key], ls=smap[key])
            
        ax.legend(loc='upper left', frameon=False)
        if label == 'erp':
            ax.set_ylim(erpmin - 5, erpmax + 5)  # .35
            ax.set(xlabel='Lag (s)', ylabel='Power',
                title=f'{electrode} {mode}')
        else:
            ax.set_ylim(vmin - 0.05, vmax + .05)  # .35
            ax.set(xlabel='Lag (s)', ylabel='Correlation (r)',
                title=f'{electrode} {label}')
    imname = get_elecbrain(electrode)
    if os.path.isfile(imname):
        arr_image = plt.imread(imname, format='png')
        fig.figimage(arr_image,
                     fig.bbox.xmax - arr_image.shape[1],
                     fig.bbox.ymax - arr_image.shape[0], zorder=5)
    pdf.savefig(fig)
    plt.close()

pdf.close()
