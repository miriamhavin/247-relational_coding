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
parser.add_argument("--erp", nargs='?', type=str, default=None)
parser.add_argument('--window-size', type=int, default=4000)
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

if args.erp:
    half_window = round((args.window_size / 1000) * 512 / 2)
    window_values = [*range(-half_window, half_window + 1)]
    args.values = [value * 1000 / 512 for value in window_values]

lags = args.values
n_av, n_df = len(args.values), len(df.columns)
assert n_av == n_df, \
    'args.values length ({n_av}) must be same size as results ({n_df})'


print('Plotting')
pdf = PdfPages(args.outfile)

# Plot results for each key (i.e. average)
x_name = 'Lag (s)'
y_name = 'Correlation (r)'
if args.erp:
    y_name = 'Power'

fig, ax = plt.subplots()
for mode, subdf in df.groupby(['label', 'mode'], axis=0):
    vals = subdf.mean(axis=0)
    err = subdf.sem(axis=0)
    ax.fill_between(lags, vals - err, vals + err, alpha=0.2, color=cmap[mode])
    label = '-'.join(mode)
    ax.plot(lags, vals, label=f'{label} ({len(subdf)})', color=cmap[mode], ls=smap[mode])
ax.legend(loc='upper right', frameon=False)
ax.set(xlabel=x_name, ylabel=y_name, title='Global average')
pdf.savefig(fig)
plt.close()

# Plot each electrode separately
vmax, vmin = df.max().max(), df.min().min()
for electrode, subdf in df.groupby('electrode', axis=0):
    fig, ax = plt.subplots()
    for (label, _, mode), values in subdf.iterrows():
        mode = (label, mode)
        label = '-'.join(mode)
        ax.plot(lags, values, label=label, color=cmap[mode], ls=smap[mode])
    ax.legend(loc='upper left', frameon=False)
    ax.set_ylim(vmin - 0.05, vmax + .05)  # .35
    ax.set(xlabel=x_name, ylabel=y_name, title=f'{electrode}')
    imname = get_elecbrain(electrode)
    if os.path.isfile(imname):
        arr_image = plt.imread(imname, format='png')
        fig.figimage(arr_image,
                     fig.bbox.xmax - arr_image.shape[1],
                     fig.bbox.ymax - arr_image.shape[0], zorder=5)
    pdf.savefig(fig)
    plt.close()

pdf.close()
