import csv
import glob
import os
from functools import partial
from multiprocessing import Pool
from urllib.parse import _NetlocResultMixinBytes

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tfsenc_parser import parse_arguments
from tfsenc_phase_shuffle import phase_randomize_1d
from tfsenc_read_datum import read_datum
from tfsenc_load_signal import load_electrode_data
from tfsenc_utils import (append_jobid_to_string, create_output_directory,
                          encoding_regression, encoding_regression_pr,
                          load_header, setup_environ, build_XY, get_folds,
                          run_regression, write_encoding_results)
from utils import load_pickle, main_timer, write_config


def return_stitch_index(args):
    """[summary]
    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    stitch_file = os.path.join(args.PICKLE_DIR, args.stitch_file)
    stitch_index = [0] + load_pickle(stitch_file)
    return stitch_index


# def process_datum(args, df):
#     df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())

#     # drop empty embeddings
#     df = df[~df['is_nan']]

#     # use columns where token is root
#     if 'gpt2-xl' in [args.align_with, args.emb_type]:
#         df = df[df['gpt2-xl_token_is_root']]
#     elif 'bert' in [args.align_with, args.emb_type]:
#         df = df[df['bert_token_is_root']]
#     else:
#         pass

#     df = df[~df['glove50_embeddings'].isna()]

#     if args.emb_type == 'glove50':
#         df['embeddings'] = df['glove50_embeddings']

#     return df


# def load_processed_datum(args):
#     conversations = sorted(
#         glob.glob(
#             os.path.join(os.getcwd(), 'data', str(args.sid), 'conv_embeddings',
#                          '*')))
#     all_datums = []
#     for conversation in conversations:
#         datum = load_pickle(conversation)
#         df = pd.DataFrame.from_dict(datum)
#         df = process_datum(args, df)
#         all_datums.append(df)

#     concatenated_datum = pd.concat(all_datums, ignore_index=True)

#     return concatenated_datum


def process_subjects(args):
    """Process electrodes for subjects (requires electrode list or sig elec file)
    
    Args:
        args (namespace): commandline arguments

    Returns:
        electrode_info (dict): Dictionary in the format (sid, elec_id): elec_name
    """
    ds = load_pickle(os.path.join(args.PICKLE_DIR, args.electrode_file))
    df = pd.DataFrame(ds)

    if args.sig_elec_file: # sig elec files for 1 or more sid (used for 777)
        sig_elec_file = os.path.join(os.path.join(os.getcwd(), 'data', args.sig_elec_file))
        sig_elec_list = pd.read_csv(sig_elec_file).rename(columns={"electrode":"electrode_name"})
        sid_sig_elec_list = pd.merge(df,sig_elec_list,how='inner',on=['subject','electrode_name'])
        assert len(sig_elec_list) == len(sid_sig_elec_list), 'Sig Elecs Missing'
        electrode_info = {
            (values['subject'], values['electrode_id']): values['electrode_name']
            for _, values in sid_sig_elec_list.iterrows()
        }

    else: # electrode list for 1 sid
        assert args.electrodes, 'Need electrode list since no sig_elec_list'
        electrode_info = {
            (args.sid,key): next(
                iter(df.loc[(df.subject == str(args.sid)) &
                            (df.electrode_id == key), 'electrode_name']), None)
            for key in args.electrodes
        }

    return electrode_info


# def process_sig_electrodes(args, datum):
#     """Run encoding on select significant elctrodes specified by a file
#     """
#     # Read in the significant electrodes
#     sig_elec_file = os.path.join(
#         os.path.join(os.getcwd(), 'data', args.sig_elec_file))
#     sig_elec_list = pd.read_csv(sig_elec_file)

#     # Loop over each electrode
#     for subject, elec_name in sig_elec_list.itertuples(index=False):

#         assert isinstance(subject, int)
#         CONV_DIR = '/projects/HASSON/247/data/conversations'
#         if args.project_id == 'podcast':
#             CONV_DIR = '/projects/HASSON/247/data/podcast'
#         BRAIN_DIR_STR = 'preprocessed_all'

#         fname = os.path.join(CONV_DIR, 'NY' + str(subject) + '*')
#         subject_id = glob.glob(fname)
#         assert len(subject_id), f'No data found in {fname}'
#         subject_id = os.path.basename(subject_id[0])

#         # Read subject's header
#         labels = load_header(CONV_DIR, subject_id)
#         assert labels is not None, 'Missing header'
#         electrode_num = labels.index(elec_name) + 1

#         # Read electrode data
#         brain_dir = os.path.join(CONV_DIR, subject_id, BRAIN_DIR_STR)
#         electrode_file = os.path.join(
#             brain_dir, ''.join([
#                 subject_id, '_electrode_preprocess_file_',
#                 str(electrode_num), '.mat'
#             ]))
#         try:
#             elec_signal = loadmat(electrode_file)['p1st']
#             elec_signal = elec_signal.reshape(-1, 1)
#         except FileNotFoundError:
#             print(f'Missing: {electrode_file}')
#             continue

#         # Perform encoding/regression
#         encoding_regression(args, datum, elec_signal,
#                             str(subject) + '_' + elec_name)

#     return


# def dumdum1(iter_idx, args, datum, signal, name):
#     seed = iter_idx + (os.getenv("SLURM_ARRAY_TASk_ID", 0) * 10000)
#     np.random.seed(seed)
#     new_signal = phase_randomize_1d(signal)
#     (prod_corr, comp_corr) = encoding_regression_pr(args, datum, new_signal,
#                                                     name)

#     return (prod_corr, comp_corr)


# def write_output(args, output_mat, name, output_str):

#     output_dir = create_output_directory(args)

#     if all(output_mat):
#         trial_str = append_jobid_to_string(args, output_str)
#         filename = os.path.join(output_dir, name + trial_str + '.csv')
#         with open(filename, 'w') as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerows(output_mat)


def single_electrode_encoding(electrode, args, datum, stitch_index):
    """Doing encoding for one electrode

    Args:
        electrode: tuple in the form ((sid, elec_id), elec_name)
        args (namespace): commandline arguments
        datum: datum of words
        stitch_index: stitch_index

    Returns:
        tuple in the format (sid, electrode name, production len, comprehension len)
    """
    # Get electrode info
    (sid, elec_id), elec_name = electrode

    if elec_name is None:
        print(f'Electrode ID {elec_id} does not exist')
        return (args.sid, None, 0, 0)

    # Load signal Data
    elec_signal, missing_convos = load_electrode_data(args, sid, elec_id, stitch_index, False)

    # Modify datum based on signal
    if len(missing_convos) > 0: # signal missing convos
        elec_datum = datum.loc[~datum['conversation_name'].isin(missing_convos)] # filter missing convos
    else:
        elec_datum = datum

    if len(elec_datum) == 0: # datum has no words, meaning no signal
        print(f'{args.sid} {elec_name} No Signal')
        return (args.sid, elec_name, 0, 0)
    elif args.project_id == 'tfs' and elec_datum.conversation_id.nunique() < 5: # datum has less than 5 convos
        print(f'{args.sid} {elec_name} has less than 5 conversations')
        return (args.sid, elec_name, 1, 1)

    # Build design matrices
    X, Y = build_XY(args, elec_datum, elec_signal)

    # Get folds
    fold_num = 5
    fold_cat_prod, fold_cat_comp = get_folds(args, elec_datum, X, Y, fold_num)

    # Split into production and comprehension
    prod_X = X[elec_datum.speaker == 'Speaker1', :]
    comp_X = X[elec_datum.speaker != 'Speaker1', :]
    prod_Y = Y[elec_datum.speaker == 'Speaker1', :]
    comp_Y = Y[elec_datum.speaker != 'Speaker1', :]

    if args.sig_elec_file: # could have multiple sids (like 777)
        # Differentiates same electrode name from different subjects
        elec_name = str(sid) + '_' + elec_name
    print(f'{args.sid} {elec_name} Prod: {len(prod_X)} Comp: {len(comp_X)}')

    # Run regression and correlation
    if args.model_mod and 'pc-flip' in args.model_mod: # prod-comp flip
        if len(prod_X) > 0:
            prod_results = run_regression(args, prod_X, prod_Y, fold_cat_prod, comp_X, comp_Y, fold_cat_comp, fold_num)
        comp_results = run_regression(args, comp_X, comp_Y, fold_cat_comp, prod_X, prod_Y, fold_cat_prod, fold_num)
    else: # normal encoding
        if len(prod_X) > 0:
            prod_results = run_regression(args, prod_X, prod_Y, fold_cat_prod, prod_X, prod_Y, fold_cat_prod, fold_num)
        comp_results = run_regression(args, comp_X, comp_Y, fold_cat_comp, comp_X, comp_Y, fold_cat_comp, fold_num)
    
    # Save encoding results
    if len(prod_X) > 0:
        write_encoding_results(args, prod_results, elec_name, 'prod')
    write_encoding_results(args, comp_results, elec_name, 'comp')
    
    # # Perform encoding/regression
    # if args.phase_shuffle:
    #     if args.project_id == 'podcast':
    #         with Pool() as pool:
    #             corr = pool.map(
    #                 partial(dumdum1,
    #                         args=args,
    #                         datum=elec_datum,
    #                         signal=elec_signal,
    #                         name=elec_name), range(args.npermutations))
    #     else:
    #         corr = []
    #         for i in range(args.npermutations):
    #             corr.append(dumdum1(i, args, elec_datum, elec_signal,
    #                                 elec_name))

    #     prod_corr, comp_corr = map(list, zip(*corr))
    #     write_output(args, prod_corr, elec_name, 'prod')
    #     write_output(args, comp_corr, elec_name, 'comp')
    # else:
    #     encoding_regression(args, elec_datum, elec_signal, elec_name)

    return (sid, elec_name, len(prod_X), len(comp_X))


def parallel_encoding(args, electrode_info, datum, stitch_index, parallel = True):
    """Doing encoding for all electrodes in parallel

    Args:
        args (namespace): commandline arguments
        electrode_info: dictionary of electrode id and electrode names
        datum: datum of words
        stitch_index: stitch_index
        parallel: whether to encode for all electrodes in parallel or not

    Returns:
        None
    """

    if args.emb_type == 'gpt2-xl' and args.sid == 676:
        parallel = False
    if parallel:
        print('Running all electrodes in parallel')
        summary_file = os.path.join(args.full_output_dir,'summary.csv') # summary file
        p = Pool(4) # multiprocessing
        with open(summary_file, 'w') as f:
            writer=csv.writer(f, delimiter=",", lineterminator="\r\n") 
            writer.writerow(('sid','electrode','prod','comp'))
            for result in p.map(partial(single_electrode_encoding,
                    args = args,
                    datum = datum,
                    stitch_index = stitch_index,
                ), electrode_info.items()):
                writer.writerow(result)
    else:
        print('Running all electrodes')
        for electrode in electrode_info.items():
            single_electrode_encoding(electrode, args, datum, stitch_index)

    return None

def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
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
    stitch_index = return_stitch_index(args)
    datum = read_datum(args, stitch_index)

    # Processing significant electrodes or individual subjects
    electrode_info = process_subjects(args)
    parallel_encoding(args, electrode_info, datum, stitch_index)

    return


if __name__ == "__main__":
    main()
