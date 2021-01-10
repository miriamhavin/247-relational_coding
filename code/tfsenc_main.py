import argparse
import csv
import os
import pickle
from datetime import datetime

import pandas as pd
from scipy.io import loadmat

from tfsenc_pca import run_pca
from tfsenc_read_datum import read_datum
from tfsenc_utils import encoding_regression, load_header, setup_environ


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(file, 'rb') as fh:
        datum = pickle.load(fh)

    return datum


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-value', type=str, default='all')
    parser.add_argument('--window-size', type=int, default=200)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--shuffle', action='store_true', default=False)
    group1.add_argument('--phase-shuffle', action='store_true', default=False)

    parser.add_argument('--lags', nargs='+', type=int)
    parser.add_argument('--output-prefix', type=str, default='test')
    parser.add_argument('--emb-type', type=str, default=None)
    parser.add_argument('--context-length', type=int, default=0)
    parser.add_argument('--datum-emb-fn',
                        type=str,
                        default='podcast-datum-glove-50d.csv')
    parser.add_argument('--electrodes', nargs='*', type=int)
    parser.add_argument('--npermutations', type=int, default=1)
    parser.add_argument('--min-word-freq', nargs='?', type=int, default=1)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sid', nargs='?', type=int, default=None)
    group.add_argument('--sig-elec-file', nargs='?', type=str, default=None)

    parser.add_argument('--pca-flag', action='store_true', default=False)
    parser.add_argument('--reduce-to', type=int, default=0)

    parser.add_argument('--align-with', type=str, default=None)
    parser.add_argument('--align-target-context-length', type=int, default=0)

    args = parser.parse_args()

    if not args.pca_flag:
        args.reduce_to = 0

    if args.pca_flag and not args.reduce_to:
        parser.error("Cannot reduce PCA to 0 dimensions")

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")



    return args


def write_electrodes(args, electrode_names):
    with open(os.path.join(args.full_output_dir, 'electrodes.csv'), 'w') as f:
        write = csv.writer(f)
        write.writerows([electrode_names])


def process_subjects(args, datum):
    """Run encoding on particular subject (requires specifying electrodes)
    """
    trimmed_signal_dict = load_pickle(
        os.path.join(args.PICKLE_DIR, str(args.sid), args.signal_file))

    trimmed_signal = trimmed_signal_dict['trimmed_signal']
    electrode_ids = trimmed_signal_dict['electrode_ids']
    electrode_names = trimmed_signal_dict['electrode_names']

    if args.electrodes:
        indices = [electrode_ids.index(i) for i in args.electrodes]

        trimmed_signal = trimmed_signal[:, indices]
        electrode_names = [electrode_names[i] for i in indices]

    # Loop over each electrode
    for elec_signal, name in zip(trimmed_signal.T, electrode_names):
        encoding_regression(args, args.sid, datum, elec_signal, name)

    write_electrodes(args, electrode_names)

    return


def process_sig_electrodes(args, datum):
    """Run encoding on select significant electrodes specified by a file 
    """
    flag = 'prediction_presentation' if not args.tiger else ''

    # Read in the significant electrodes
    sig_elec_file = os.path.join(args.PROJ_DIR, flag, args.sig_elec_file)
    sig_elec_list = pd.read_csv(sig_elec_file, header=None)[0].tolist()

    # Loop over each electrode
    for sig_elec in sig_elec_list:
        subject_id, elec_name = sig_elec[:29], sig_elec[30:]

        # Read subject's header
        labels = load_header(args.CONV_DIR, subject_id)
        if not labels:
            print('Header Missing')
        electrode_num = labels.index(elec_name)

        # Read electrode data
        brain_dir = os.path.join(args.CONV_DIR, subject_id, args.BRAIN_DIR_STR)
        electrode_file = os.path.join(
            brain_dir, ''.join([
                subject_id, '_electrode_preprocess_file_',
                str(electrode_num + 1), '.mat'
            ]))
        try:
            elec_signal = loadmat(electrode_file)['p1st']
        except FileNotFoundError:
            print(f'Missing: {electrode_file}')
            continue

        # Perform encoding/regression
        encoding_regression(args, subject_id, datum, elec_signal, elec_name)

    return


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Locate and read datum
    datum = read_datum(args)

    if args.pca_flag:
        datum = run_pca(args, datum)

    # Processing significant electrodes or individual subjects
    if args.sig_elec_file:
        process_sig_electrodes(args, datum)
    else:
        process_subjects(args, datum)

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
