import glob
import os

import numpy as np
import pandas as pd
from statsmodels.stats import multitest

if __name__ == "__main__":
    subjects = sorted(
        glob.glob(
            '/scratch/gpfs/hgazula/247-encoding/results/tfs/colton-no-shuffle/*'
        ))

    lags = np.arange(-2000, 2001, 25)
    names = []
    pVals = []

    pval_dict = dict()
    some_list = []
    for subject in subjects:
        subject_key = os.path.basename(subject)
        if subject_key == '676':
            print('passing over')
            continue
        shuffle_elec_file_list = sorted(
            glob.glob(
                os.path.join(
                    '/scratch/gpfs/hgazula/247-encoding/results/tfs/colton-phase-shuffle',
                    os.path.basename(subject), '*_comp.csv')))

        main_elec_file_list = sorted(
            glob.glob(
                os.path.join(
                    '/scratch/gpfs/hgazula/247-encoding/results/tfs/colton-no-shuffle',
                    os.path.basename(subject), '*_comp.csv')))

        a = [os.path.basename(item) for item in shuffle_elec_file_list]
        b = [os.path.basename(item) for item in main_elec_file_list]

        print(len(set(a)), len(set(b)))

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

            print(omaxs[:10])

            s = 1 - (sum(np.max(rc_result) > omaxs) / perm_result.shape[0])
            print(elecname1, np.max(rc_result))
            print('++++++++++++++++++++++')
            some_list.append((subject_key, elecname1, s))

df = pd.DataFrame(some_list, columns=['subject', 'electrode', 'score'])
_, pcor, _, _ = multitest.multipletests(df.score.values,
                                        method='fdr_bh',
                                        is_sorted=False)

thresh = 0.01
flag = np.logical_or(np.isclose(pcor, thresh), pcor < thresh)

df = df[flag]
df['electrode'] = df['electrode'].str.strip('_comp')
df.to_csv('significant_electrodes-comp-hg.csh',
          index=False,
          columns=['subject', 'electrode'])
