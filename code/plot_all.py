import glob
import argparse
import os
import pandas as pd
import itertools

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument("--formats", nargs="+", required=True)
parser.add_argument("--labels", nargs="+",  required=True)
parser.add_argument("--values", nargs="+", type=float, required=True)
parser.add_argument("--keys", nargs="+",  required=True)
parser.add_argument("--sid", type=int, nargs="+", required=True)
parser.add_argument("--sig-elec-file", nargs="+", default=[])
parser.add_argument("--outfile", default='results/figures/tmp.pdf')
args = parser.parse_args()

assert len(args.labels) * len(args.sid) == len(args.formats)

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
multiple_sid = False
if len(args.sig_elec_file) == len(args.sid):
    for fname, sid in zip(args.sig_elec_file, args.sid):
        elecs = pd.read_csv('data/' + fname)['electrode'].tolist()
        sigelecs[sid] = set(elecs)
elif len(args.sig_elec_file) == len(args.sid) * len(args.keys):
    multiple_sid = True
    sid_key_tup = [x for x in itertools.product(args.sid, args.keys)]
    for fname, sid_key in zip(args.sig_elec_file, sid_key_tup):
        elecs = pd.read_csv('data/' + fname)['electrode'].tolist()
        sigelecs[sid_key] = set(elecs)

print('Aggregating data')
data = []
args.labels = args.labels * len(args.sid)

for fmt, label in zip(args.formats, args.labels):
    load_sid = 0
    for sid in args.sid:
        if str(sid) in fmt:
            load_sid = sid
    for key in args.keys:
        fname = fmt % key
        files = glob.glob(fname)
        assert len(files) > 0, f"No results found under {fname}"

        for resultfn in files:
            elec = os.path.basename(resultfn).replace('.csv', '')[:-5]
            # Skip electrodes if they're not part of the sig list
            if multiple_sid:
                if len(sigelecs) and elec not in sigelecs[(load_sid,key)]:
                    continue
            else:
                if len(sigelecs) and elec not in sigelecs[load_sid]:
                    continue
            df = pd.read_csv(resultfn, header=None)
            df.insert(0, 'sid', load_sid)
            df.insert(0, 'mode', key)
            df.insert(0, 'electrode', elec)
            df.insert(0, 'label', label)
            data.append(df)


if not len(data):
    print('No data found')
    exit(1)
df = pd.concat(data)
df.set_index(['label', 'electrode', 'mode','sid'], inplace=True)
lags = args.values
lags = [lag / 1000 for lag in lags]

print('Plotting')
pdf = PdfPages(args.outfile)

plot_mode = 'quardra'
plot_mode = 'none'
plot_mode = 'final'

lag_ticks = lags
if plot_mode == 'none':
    lag_ticks_out = []
elif plot_mode == 'quardra':
    lag_ticks_out = [12,13,14,15,16,18,20,22,24,26,28] # quardra
    lag_tick_locations = [-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]
    lag_ticklabels = [-300,-250,-200,-150,-120,-90,-60,-40,-20,-10,-8,-6,-4,-2,0,2,4,6,8,10,20,40,60,90,120,150,200,250,300]
elif plot_mode == 'final':
    lag_ticks_out = [12,14,16] # final
    lag_tick_locations = [-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16]
    lag_ticklabels = [-300,-60,-30,-10,-8,-6,-4,-2,0,2,4,6,8,10,30,60,300]
# lag_ticks_out = [12,13,14,15,16,18,20,22] # triple
for lag in lag_ticks_out:
    lag_ticks.insert(0,lag*-1)
    lag_ticks.append(lag)

# plot_mode = 'none'
# lag_idx = [i for i,lag in enumerate(lag_ticks) if (lag >= -2 and lag <= 2)] # only -2 to 2 s
# df = df[lag_idx]
# lag_ticks = [lag for lag in lag_ticks if (lag >= -2 and lag <= 2)]

# Plot results for each key (i.e. average)
fig, ax = plt.subplots(figsize=(15,6))
for mode, subdf in df.groupby(['label', 'mode'], axis=0):
    vals = subdf.mean(axis=0)
    err = subdf.sem(axis=0)
    ax.fill_between(lag_ticks, vals - err, vals + err, alpha=0.2, color=cmap[mode])
    label = '-'.join(mode)
    if plot_mode != 'none':
        ax.set_xticks(lag_tick_locations)
        ax.set_xticklabels(lag_ticklabels)
    ax.plot(lag_ticks, vals, label=f'{label} ({len(subdf)})', color=cmap[mode], ls=smap[mode])
ax.axhline(0,ls='dashed',alpha=0.3,c='k')
ax.axvline(0,ls='dashed',alpha=0.3,c='k')
ax.legend(loc='upper right', frameon=False)
ax.set(xlabel='Lag (s)', ylabel='Correlation (r)', title='Global average')
pdf.savefig(fig)
plt.close()



pdf.close()