import glob
import argparse
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import entropy
# from scipy.special import softmax


parser = argparse.ArgumentParser()
parser.add_argument("--formats", nargs="+", required=True)
parser.add_argument("--data", nargs="+", required=True)
parser.add_argument("--labels", nargs="+",  required=True)
parser.add_argument("--values", nargs="+", type=float, required=True)
parser.add_argument("--keys", nargs="+",  required=True)
parser.add_argument("--sig-elec-file", nargs="+", default=[])
parser.add_argument("--sid", type=int, default=625)
parser.add_argument("--outfile", default='results/figures/sig-file-')
parser.add_argument("--sig-percents", nargs="+", type=float, required=True)

args = parser.parse_args()

elecdir = f'/projects/HASSON/247/data/elecimg/{args.sid}/'

def get_elecbrain(electrode):
    name = electrode.replace('EEG', '').replace('REF', '').replace('\\', '')
    name = name.replace('_', '').replace('GR', 'G')
    imname = elecdir + f'thumb_{name}.png'  # + f'{args.sid}_{name}.png'
    return imname

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
styles = ['-', '--', '-.', ':']
cmap = {}  # color map
smap = {}  # style map
for label, color in zip(args.labels, colors):
    for key, style in zip(args.keys, styles):
        cmap[(label, key)] = color
        smap[(label, key)] = style

####################################################################################
print('Aggregating data')
####################################################################################

def load_data(files):
    data = []
    for fmt, label in zip(files, args.labels):
        for key in args.keys:
            fname = fmt % key
            files = glob.glob(fname)
            assert len(files) > 0, f"No results found under {fname}"

            for resultfn in files:
                elec = os.path.basename(resultfn).replace('.csv', '')[:-5]
                # Skip electrodes if they're not part of the sig list
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
    return df

df = load_data(args.formats)
# if args.formats[0] != args.data[0]: # criteria and plotting data is different
#     df_new = load_data(args.data)
# else:
#     df_new = df

####################################################################################
print('Writing to Sig File')
####################################################################################

df['sort_metric'] = df.max(axis = 1) - df.min(axis = 1) # row max

def save_sig_file(data, mode):
    df_partial = data[data.index.isin([mode], level='mode')].sort_values(by=['sort_metric'], ascending = False)
    for sig_percent in args.sig_percents:
        sig_num = int(sig_percent * len(df_partial.index))
        file_name = args.outfile + str(args.sid) + '-sig-test-' + str(sig_percent) + mode + '.csv'
        sig_elecs = df_partial.index.get_level_values('electrode')[0:sig_num]
        sig_elec_df = {'subject':args.sid, 'electrode':sig_elecs}
        sig_elec_df = pd.DataFrame(sig_elec_df)
        sig_elec_df.to_csv(file_name, index=False)
breakpoint()
save_sig_file(df, 'prod')
save_sig_file(df, 'comp')


####################################################################################
# print('Summary Statistics')
####################################################################################

# chosen_lag_idx = [idx for idx, element in enumerate(args.values) if (element >= -2000 and element <= 2000)]
# df_short = df.loc[:,chosen_lag_idx] # chose from -2 to 2

# chosen_lag_idx2 = [idx for idx, element in enumerate(args.values) if (element <= -20000)]
# df_short2 = df.loc[:,chosen_lag_idx2] # chose from -30 to -20

# df2 = pd.DataFrame(index=df_short.index)
# df2['elec_max'] = df_short.max(axis=1) # row max
# df2['elec_median'] = df_short.median(axis=1) # row max
# df2['elec_mean'] = df_short.mean(axis=1) # row mean
# df2['elec_sum'] = df_short.sum(axis=1) # row mean
# df2['elec_max2'] = df_short2.max(axis=1) # row max
# df2['elec_max_diff'] = df2.elec_max - df2.elec_max2 # row max

# df_en = pd.DataFrame.copy(df_short)
# df_en[df_en < 0] = 10 ** (-10)
# df2['elec_entr'] = entropy(df_en, axis=1) # row entropy

# df2.corr('pearson')
# df2.to_csv('results/figures/tfs-676-sig-test.csv')


####################################################################################
# print('Plot By Sorting Metric')
####################################################################################

# def plot_sorted(data, mode):
#     df_partial = data[data.index.isin([mode], level='mode')].sort_values(by=['sort_metric'], ascending = False)
#     df_partial = df_partial.iloc[: , :-1]
#     vmax, vmin = df_partial.max().max(), df_partial.min().min()
#     pdf_name = args.outfile + args.sid + '-sig-test-' + mode + '.pdf'
#     pdf = PdfPages(pdf_name)
#     for _, values in df_partial.iterrows():
#         fig, ax = plt.subplots(figsize=(15, 6))
#         ax.plot(args.values, values)
#         if args.formats[0] != args.data[0]:
#             ax.plot(args.values, df.loc[values.name], color='red',ls='dashed',alpha=0.5)
#         ax.axhline(0,ls='dashed',alpha=0.3,c='k')
#         ax.axvline(0,ls='dashed',alpha=0.3,c='k')
#         ax.set_ylim(vmin - 0.05, vmax + .05)
#         electrode = values.name[1]
#         ax.set(xlabel='Lag (s)', ylabel='Correlation (r)', title=f'{electrode}')
#         imname = get_elecbrain(electrode)
#         if os.path.isfile(imname):
#             arr_image = plt.imread(imname, format='png')
#             fig.figimage(arr_image,
#                 fig.bbox.xmax - arr_image.shape[1],
#                 fig.bbox.ymax - arr_image.shape[0], zorder=5)
#         pdf.savefig(fig)
#         plt.close()
#     pdf.close()

# plot_sorted(df_new, 'prod')
# plot_sorted(df_new, 'comp')


####################################################################################
# print('Summary Stats Plot')
####################################################################################

# def plot_fig(data, column):
#     fig, axes = plt.subplots(1, len(args.keys), figsize=(12, 6))
#     for ax, (mode, subdf) in zip(axes, data.groupby('mode')):
#         key = (label, mode)
#         ax.plot(subdf[column].tolist(), label=label, color=cmap[key], ls=smap[key])
#         ax.set(title=column)
#     return fig

# def plot_figs(data):
#     pdf = PdfPages('results/figures/tfs-676-sig-test.pdf')
#     for column in data:
#         fig = plot_fig(data, column)
#         pdf.savefig(fig)
#         plt.close()
#     pdf.close()

# plot_figs(df2)

