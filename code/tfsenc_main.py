import csv
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.io import loadmat

from tfsenc_parser import parse_arguments
from tfsenc_pca import run_pca
from tfsenc_phase_shuffle import phase_randomize_1d
from tfsenc_read_datum import read_datum
from tfsenc_utils import (append_jobid_to_string, create_output_directory,
                          encoding_regression, encoding_regression_pr,
                          load_header, setup_environ)
from utils import load_pickle, main_timer, write_config


def trim_signal(signal):
    bin_size = 32  # 62.5 ms (62.5/1000 * 512)
    signal_length = signal.shape[0]

    if signal_length < bin_size:
        print("Ignoring conversation: Small signal")
        return None

    cutoff_portion = signal_length % bin_size
    if cutoff_portion:
        signal = signal[:-cutoff_portion, :]

    return signal


def load_electrode_data(args, elec_id):
    '''Loads specific electrodes mat files
    '''
    DATA_DIR = '/projects/HASSON/247/data/conversations-car'
    convos = sorted(glob.glob(os.path.join(DATA_DIR, str(args.sid), '*')))

    all_signal = []
    for convo_id, convo in enumerate(convos, 1):
        file = glob.glob(
            os.path.join(convo, 'preprocessed',
                         '*' + str(elec_id) + '.mat'))[0]
        mat_signal = loadmat(file)['p1st']

        # mat_signal = trim_signal(mat_signal)

        if mat_signal is None:
            continue
        all_signal.append(mat_signal)

    elec_signal = np.vstack(all_signal)

    return elec_signal


def process_datum(args, df):
    df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())

    # drop empty embeddings
    df = df[~df['is_nan']]

    # use columns where token is root
    if 'gpt2' in [args.align_with, args.emb_type]:
        df = df[df['gpt2_token_is_root']]
    elif 'bert' in [args.align_with, args.emb_type]:
        df = df[df['bert_token_is_root']]
    else:
        pass

    df = df[~df['glove50_embeddings'].isna()]

    if args.emb_type == 'glove50':
        df['embeddings'] = df['glove50_embeddings']

    return df


def load_processed_datum(args):
    conversations = sorted(
        glob.glob(
            os.path.join(os.getcwd(), 'data', str(args.sid), 'conv_embeddings',
                         '*')))
    all_datums = []
    for conversation in conversations:
        datum = load_pickle(conversation)
        df = pd.DataFrame.from_dict(datum)
        df = process_datum(args, df)
        all_datums.append(df)

    concatenated_datum = pd.concat(all_datums, ignore_index=True)

    return concatenated_datum


def process_subjects(args):
    """Run encoding on particular subject (requires specifying electrodes)
    """
    # trimmed_signal = trimmed_signal_dict['trimmed_signal']

    # if args.electrodes:
    #     indices = [electrode_ids.index(i) for i in args.electrodes]

    #     trimmed_signal = trimmed_signal[:, indices]
    #     electrode_names = [electrode_names[i] for i in indices]

    electrode_info = load_pickle(
        os.path.join(args.PICKLE_DIR, args.electrode_file))

    if args.electrodes:
        electrode_info = {
            key: electrode_info.get(key, None)
            for key in args.electrodes
        }

    # # Loop over each electrode
    # for elec_id, elec_name in electrode_info.items():

    #     if elec_name is None:
    #         print(f'Electrode ID {elec_id} does not exist')
    #         continue

    #     elec_signal = load_electrode_data(args, elec_id)
    #     # datum = load_processed_datum(args)

    #     encoding_regression(args, datum, elec_signal, elec_name)

    # # write_electrodes(args, electrode_names)

    return electrode_info


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


def dumdum1(iter_idx, args, datum, signal, name):
    np.random.seed(iter_idx)
    new_signal = phase_randomize_1d(signal)
    (prod_corr, comp_corr) = encoding_regression_pr(args, datum, new_signal,
                                                    name)

    return (prod_corr, comp_corr)


def write_output(args, output_mat, name, output_str):

    output_dir = create_output_directory(args)

    if all(output_mat):
        trial_str = append_jobid_to_string(args, output_str)
        filename = os.path.join(output_dir, name + trial_str + '.csv')
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(output_mat)


def this_is_where_you_perform_regression(args, electrode_info, datum):

    # Loop over each electrode
    for elec_id, elec_name in electrode_info.items():

        if elec_name is None:
            print(f'Electrode ID {elec_id} does not exist')
            continue

        elec_signal = load_electrode_data(args, elec_id)

        # Perform encoding/regression
        if args.phase_shuffle:
            # with Pool() as pool:
            #     corr = pool.map(
            #         partial(dumdum1,
            #                 args=args,
            #                 datum=datum,
            #                 signal=elec_signal,
            #                 name=elec_name), range(args.npermutations))

            corr = []
            for i in range(args.npermutations):
                corr.append(dumdum1(i, args, datum, elec_signal, elec_name))

            prod_corr, comp_corr = map(list, zip(*corr))
            write_output(args, prod_corr, elec_name, 'prod')
            write_output(args, comp_corr, elec_name, 'comp')
        else:
            encoding_regression(args, datum, elec_signal, elec_name)

    return None


@main_timer
def main():
    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Saving configuration to output directory
    write_config(vars(args))

    # Locate and read datum
    datum = read_datum(args)

    if args.pca_flag:
        datum = run_pca(args, datum)

    # Processing significant electrodes or individual subjects
    electrode_info = process_subjects(args)
    this_is_where_you_perform_regression(args, electrode_info, datum)

    return


if __name__ == "__main__":
    main()
