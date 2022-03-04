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
parser.add_argument("--window-size", type=int, default=1024)
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
for label, color in zip(args.labels, colors):
    for key, style in zip(args.keys, styles):
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
            df.insert(0, 'mode', key)
            df.insert(0, 'electrode', elec)
            df.insert(0, 'label', label)
            data.append(df)

if not len(data):
    print('No data found')
    exit(1)
df = pd.concat(data)
df.set_index(['label', 'electrode', 'mode'], inplace=True)
# lags = list(range(len(df.columns)))
lags = args.values
n_av, n_df = len(args.values), len(df.columns)
assert n_av == n_df, \
    'args.values length ({n_av}) must be same size as results ({n_df})'

print('Plotting')
pdf = PdfPages(args.outfile)
lag_ticks = range(-30000,30500,500)
lag_ticks = [lag / 1000 for lag in lag_ticks]
# lag_ticks_out = [3.0, 3.5, 4.0, 4.5, 5.0]
# lag_ticks_out = [3.0, 4.0]
lag_ticks_out = [36.0]
# lag_ticks_out = []
for lag in lag_ticks_out:
    lag_ticks.insert(0,lag*-1)
    lag_ticks.append(lag)

lag_tick_locations = [-36,-30,-24,-18,-12,-6,0,6,12,18,24,30,36]
lag_ticklabels = [-150,-30,-24,-18,-12,-6,0,6,12,18,24,30,150]

# lag_tick_locations = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
# lag_ticklabels = [-150,-90,-30,-2,-1,0,1,2,30,90,150]

# lag_ticks = ['-6','','-2','','-1','','0','','1','','2','','6']
# Plot results for each key (i.e. average)
# plot each key/mode in its own subplot
fig, axes = plt.subplots(1, len(args.keys), figsize=(12, 6))
for ax, (mode, subdf) in zip(axes, df.groupby('mode', axis=0)):
    for label, subsubdf in subdf.groupby('label', axis=0):
        vals = subsubdf.mean(axis=0)
        err = subsubdf.sem(axis=0)
        key = (label, mode)
        ax.fill_between(lag_ticks, vals - err, vals + err, alpha=0.2, color=cmap[key])
        ax.plot(lag_ticks, vals, label=f'{label} ({len(subsubdf)})', color=cmap[key], ls=smap[key])
        ax.set_xticks(lag_tick_locations)
        ax.set_xticklabels(lag_ticklabels)
    ax.axhline(0,ls='dashed',alpha=0.3,c='k')
    ax.axvline(0,ls='dashed',alpha=0.3,c='k')
    ax.set_title(mode + ' global average')
    ax.legend(loc='upper right', frameon=False)
    ax.set(xlabel='Lag (s)', ylabel='Correlation (r)')
pdf.savefig(fig)
plt.close()

# plot all keys together
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
vmax, vmin = df.max().max(), df.min().min()
for electrode, subdf in df.groupby('electrode', axis=0):
    fig, axes = plt.subplots(1, len(args.keys), figsize=(12, 6))
    for ax, (mode, subsubdf) in zip(axes, subdf.groupby('mode')):
        for row, values in subsubdf.iterrows():
            # print(mode, label, type(values))
            # print(subsubdf)
            label = row[0]
            key = (label, mode)
            ax.plot(lag_ticks, values, label=label, color=cmap[key], ls=smap[key])
            ax.set_xticks(lag_tick_locations)
            ax.set_xticklabels(lag_ticklabels)
        ax.axhline(0,ls='dashed',alpha=0.3,c='k')
        ax.axvline(0,ls='dashed',alpha=0.3,c='k')
        ax.legend(loc='upper left', frameon=False)
        ax.set_ylim(vmin - 0.05, vmax + .05)  # .35
        ax.set(xlabel='Lag (s)', ylabel='Correlation (r)',
               title=f'{electrode} {mode}')
    imname = get_elecbrain(electrode)
    if os.path.isfile(imname):
        arr_image = plt.imread(imname, format='png')
        fig.figimage(arr_image,
                     fig.bbox.xmax - arr_image.shape[1],
                     fig.bbox.ymax - arr_image.shape[0], zorder=5)
    pdf.savefig(fig)
    plt.close()

pdf.close()
