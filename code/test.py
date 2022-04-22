import glob
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

from utils import load_pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sid = '676'
electrodes = range(1,126)
bad_elec = ['EEGG_21REF']
good_elec = ['EEGG_48REF']

# Load electrode names
ds = load_pickle('/scratch/gpfs/kw1166/247-encoding/data/tfs/676/pickles/676_electrode_names.pkl')
df = pd.DataFrame(ds)

electrode_info = {
    key: next(
        iter(df.loc[(df.subject == str(sid)) & 
        (df.electrode_id == key), 'electrode_name']), None)
    for key in electrodes
}

electrode_info_2 = {
    next(
        iter(df.loc[(df.subject == str(sid)) & 
        (df.electrode_id == key), 'electrode_name']), None): key
    for key in electrodes
}

process_flag = 'preprocessed'
DATA_DIR = '/projects/HASSON/247/data/conversations-car'
convos = sorted(glob.glob(os.path.join(DATA_DIR, str(sid), '*')))

sig_elecs = pd.read_csv('data/tfs-sig-file-676-sig-1.0-comp.csv')# load significant elecs

def get_conv_signal(elec_id):
    all_signal = []
    # all_conv_signal = {}
    # conv_sorted = {}
    
    for convo_id, convo in enumerate(convos, 1):
    
        # convo_name = os.path.basename(convo)

        file = glob.glob(
            os.path.join(convo, process_flag, '*_' + str(elec_id) + '.mat'))[0]

        mat_signal = loadmat(file)['p1st']
        mat_signal = mat_signal.reshape(-1, 1)

        # mat_signal = trim_signal(mat_signal)

        if mat_signal is None:
            continue
        all_signal.append(mat_signal)
        # all_conv_signal.update({convo_name:mat_signal})
        # conv_sorted.update({mat_signal.shape[0]:convo_name})
    all_signal = np.vstack(all_signal)
    # return (all_signal, all_conv_signal, conv_sorted)
    return all_signal

# id = list(electrode_info.keys())[list(electrode_info.values()).index(bad_elec[0])]
# all_bad_sig, bad_sig, conv_sorted = get_conv_signal(id)

# id = list(electrode_info.keys())[list(electrode_info.values()).index(good_elec[0])]
# all_good_sig, good_sig, conv_sorted = get_conv_signal(id)


# auto_cor_bad = sm.tsa.acf(all_bad_sig, nlags = 500)
# auto_cor_good = sm.tsa.acf(all_good_sig, nlags = 500)

pdf = PdfPages('autocorr.pdf')
for row, values in sig_elecs.iterrows():
    conv_signal = get_conv_signal(electrode_info_2[values.electrode])
    auto_cor = sm.tsa.acf(conv_signal, nlags = 300)
    fig, ax = plt.subplots(figsize=(18,6))
    ax.plot(auto_cor, color='r')
    ax.set(xlabel='Lag', ylabel='Auto Correlation', title=f'{values.electrode}')
    pdf.savefig(fig)
    plt.close()
pdf.close()


# fig, ax = plt.subplots(figsize=(18,6))
# ax.plot(auto_cor_bad, color='g', label='bad-elec')
# ax.plot(auto_cor_good, color='r', label='bad-elec')
# pdf.savefig(fig)
# plt.close()
# pdf.close()


# def bin_signal(signal):
#     new_sig = []
#     signal_len = signal.shape[0]
#     bins = np.arange(0,signal_len,200)
#     for bin in range(1,len(bins)):
#         new_sig.append(signal[bins[bin-1]:bins[bin]].mean())
#     return new_sig


# for conv in sorted(conv_sorted.items()):
#     convname = conv[1]
#     pdf = PdfPages(f'results/signal/{convname}.pdf')
#     print(f'drawing signal for {convname}')
#     assert good_sig[convname].shape == good_sig[convname].shape
#     signal_len = good_sig[convname].shape[0]
#     plot_len = np.arange(0,signal_len,10000)
#     for plot in range(1,len(plot_len)):
#         fig, ax = plt.subplots(figsize=(18,6))
#         time = [*range(plot_len[plot-1],plot_len[plot])]
#         ax.plot(time, bad_sig[convname][plot_len[plot-1]:plot_len[plot]], color='g', label='bad-elec')
#         ax.plot(time, good_sig[convname][plot_len[plot-1]:plot_len[plot]], color='r', label='good-elec')
#         ax.set(xlabel='Time', ylabel='Signal', title=f'{convname}')
#         # fig = tsaplots.plot_acf(conv[1], color='g', title=convname)
#         pdf.savefig(fig)
#         plt.close()
#     pdf.close()


# elec_signal = np.vstack(all_signal)


