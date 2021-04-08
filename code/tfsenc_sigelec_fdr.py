import glob
import os

import numpy as np
import pandas as pd
from statsmodels.stats import multitest
from utils import load_pickle

if __name__ == "__main__":
    subjects = sorted(
        glob.glob(
            '/scratch/gpfs/hgazula/podcast-encoding/results/no-shuffle/*'))

    hemisphere_indicator = load_pickle(
        '/scratch/gpfs/hgazula/podcast_hemisphere_indicator.pkl')

    lags = np.arange(-2000, 2001, 25)

    pval_dict = dict()
    some_list = []
    for subject in subjects:
        subject_key = os.path.basename(subject)

        shuffle_elec_file_list = sorted(
            glob.glob(
                os.path.join(
                    '/scratch/gpfs/hgazula/podcast-encoding/results/phase-shuffle',
                    os.path.basename(subject), '*.csv')))

        main_elec_file_list = sorted(
            glob.glob(
                os.path.join(
                    '/scratch/gpfs/hgazula/podcast-encoding/results/no-shuffle',
                    os.path.basename(subject), '*.csv')))

        # curr_key = hemisphere_indicator.get(int(subject_key), None)

        # if not curr_key:
        #     pass
        # elif len(curr_key) == 2:
        #     shuffle_elec_file_list = list(
        #         filter(lambda x: os.path.basename(x).startswith(('L', 'DL')),
        #                shuffle_elec_file_list))
        #     main_elec_file_list = list(
        #         filter(lambda x: os.path.basename(x).startswith(('L', 'DL')),
        #                main_elec_file_list))
        # elif len(curr_key) == 1 and 'RH' in curr_key:
        #     continue
        # else:
        #     pass

        a = [os.path.basename(item) for item in shuffle_elec_file_list]
        b = [os.path.basename(item) for item in main_elec_file_list]

        assert set(a) == set(b), "Mismatch: Electrode Set"

        for elec_file1, elec_file2 in zip(shuffle_elec_file_list,
                                          main_elec_file_list):
            elecname1 = os.path.split(os.path.splitext(elec_file1)[0])[1]
            elecname2 = os.path.split(os.path.splitext(elec_file2)[0])[1]

            assert elecname1 == elecname2, 'Mismatch: Electrode Name'

            if elecname1.startswith(('SG', 'ECGEKG', 'EEGSG')):
                continue

            perm_result = pd.read_csv(elec_file1, header=None).values
            rc_result = pd.read_csv(elec_file2, header=None).values

            assert perm_result.shape[1] == rc_result.shape[
                1], "Mismatch: Number of Lags"

            if perm_result.shape[1] != len(lags):
                print('perm is wrong length')
            else:
                omaxs = np.max(perm_result, axis=1)

            s = 1 - (sum(np.max(rc_result) > omaxs) / perm_result.shape[0])
            some_list.append((subject_key, elecname1, s))

    df = pd.DataFrame(some_list, columns=['subject', 'electrode', 'score'])
    thresh = 0.01

    # df1 = df.copy(deep=True)
    # flag = np.logical_or(np.isclose(df1.score.values, thresh, atol=1e-6), df1.score.values > thresh)

    # df1 = df1[flag]
    # df1['electrode'] = df1['electrode'].str.strip('_comp')
    # df1.to_csv('pre_fdr.csv',
    #           index=False,
    #           columns=['subject', 'electrode'])

    _, pcor, _, _ = multitest.multipletests(df.score.values,
                                            method='fdr_bh',
                                            is_sorted=False)

    flag = np.logical_or(np.isclose(pcor, thresh), pcor < thresh)

    df = df[flag]
    df['electrode'] = df['electrode'].str.strip('_comp')
    df.to_csv('post_fdr.csv', index=False, columns=['subject', 'electrode'])

    filter_hemisphere = []
    for row in df.itertuples(index=False):
        subject = row.subject
        electrode = row.electrode

        curr_key = hemisphere_indicator.get(int(subject), None)

        if not curr_key:
            if int(subject) == 798:
                filter_hemisphere.append((subject, electrode))    
        elif len(curr_key) == 2:
            if electrode.startswith(('L', 'DL')):
                filter_hemisphere.append((subject, electrode))
        elif len(curr_key) == 1 and 'RH' in curr_key:
            continue
        else:
            filter_hemisphere.append((subject, electrode))

    df2 = pd.DataFrame(filter_hemisphere, columns=['subject', 'electrode'])
    df2.to_csv('post_fdr_lh.csv', index=False, columns=['subject', 'electrode'])

# phase-1000-sig-elec-glove50d-perElec-FDR-01-LH-hg