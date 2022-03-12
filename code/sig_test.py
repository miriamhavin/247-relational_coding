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
parser.add_argument("--labels", nargs="+",  required=True)
parser.add_argument("--values", nargs="+", type=float, required=True)
parser.add_argument("--keys", nargs="+",  required=True)
parser.add_argument("--sid", type=int, default=625)
parser.add_argument("--outfile", default='results/figures/tfs-sig-file-')
parser.add_argument("--sig-percents", nargs="+", type=float, required=True)

args = parser.parse_args()


####################################################################################
print('Aggregating data')
####################################################################################

data = []
for fmt, label in zip(args.formats, args.labels):
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

####################################################################################
print('Writing to Sig File')
####################################################################################

df['sort_metric'] = df.max(axis = 1) - df.min(axis = 1) # row max

def save_sig_file(data, mode):
    df_partial = data[data.index.isin([mode], level='mode')].sort_values(by=['sort_metric'], ascending = False)
    for sig_percent in args.sig_percents:
        sig_num = int(sig_percent * len(df_partial.index))
        file_name = args.outfile + str(args.sid) + '-sig-' + str(sig_percent) + '-' + mode + '.csv'
        sig_elecs = df_partial.index.get_level_values('electrode')[0:sig_num]
        sig_elec_df = {'subject':args.sid, 'electrode':sig_elecs}
        sig_elec_df = pd.DataFrame(sig_elec_df)
        sig_elec_df.to_csv(file_name, index=False)
breakpoint()
save_sig_file(df, 'prod')
save_sig_file(df, 'comp')