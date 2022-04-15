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
from tfsenc_pca import run_pca
from tfsenc_phase_shuffle import phase_randomize_1d
from tfsenc_read_datum import read_datum
from tfsenc_utils import (append_jobid_to_string, create_output_directory,
                          encoding_regression, encoding_regression_pr,
                          load_header, setup_environ)
from tfsenc_read_datum import return_stitch_index
from utils import load_pickle, main_timer, write_config
import gensim.downloader as api
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



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


def detrend_signal(mat_signal): # Detrending

    y = mat_signal
    X = np.arange(len(y)).reshape(-1,1)
    pf = PolynomialFeatures(degree=2)
    Xp = pf.fit_transform(X)

    model = LinearRegression()
    model.fit(Xp, y)
    trend = model.predict(Xp)
    mat_signal = y - trend

    return mat_signal


def load_electrode_data(args, elec_id, stitch):
    '''Loads specific electrodes mat files
    '''
    if args.project_id == 'tfs':
        DATA_DIR = '/projects/HASSON/247/data/conversations-car'
        process_flag = 'preprocessed'
    elif args.project_id == 'podcast':
        DATA_DIR = '/projects/HASSON/247/data/podcast-data'
        process_flag = 'preprocessed_all'
    else:
        raise Exception('Invalid Project ID')

    convos = sorted(glob.glob(os.path.join(DATA_DIR, str(args.sid), 'NY*Part*conversation*')))

    all_signal = []
    missing_convos = []
    for convo_id, convo in enumerate(convos, 1):
        if args.conversation_id != 0 and convo_id != args.conversation_id:
            continue

        file = glob.glob(os.path.join(convo, process_flag, '*_' + str(elec_id) + '.mat'))
        if len(file) == 1: # conversation file exists
            file = file[0]

            mat_signal = loadmat(file)['p1st']
            mat_signal = mat_signal.reshape(-1, 1)
            # mat_signal = trim_signal(mat_signal)

            if mat_signal is None:
                continue

            mat_signal = detrend_signal(mat_signal) # detrend conversation signal

        elif len(file) == 0: # conversation file does not exist
            if args.sid != 7170:
                raise SystemExit(f'Error: Conversation file does not exist for electrode {elec_id} at {convo}')
            missing_convos.append(os.path.basename(convo)) # append missing convo name
            mat_len = stitch[convo_id]-stitch[convo_id-1] # mat file length
            mat_signal = np.empty((mat_len, 1))
            mat_signal.fill(np.nan)

        else: # more than 1 conversation files
            raise SystemExit(f'Error: More than 1 signal file exists for electrode {elec_id} at {convo}')

        all_signal.append(mat_signal) # append conversation signal

    if args.project_id == 'tfs':
        elec_signal = np.vstack(all_signal)
    else:
        elec_signal = np.array(all_signal)

    return elec_signal, missing_convos


def process_datum(args, df):
    df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())

    # drop empty embeddings
    df = df[~df['is_nan']]

    # use columns where token is root
    if 'gpt2-xl' in [args.align_with, args.emb_type]:
        df = df[df['gpt2-xl_token_is_root']]
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

    ds = load_pickle(os.path.join(args.PICKLE_DIR, args.electrode_file))
    df = pd.DataFrame(ds)

    if args.electrodes:
        electrode_info = {
            key: next(
                iter(df.loc[(df.subject == str(args.sid)) &
                            (df.electrode_id == key), 'electrode_name']), None)
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
    """Run encoding on select significant elctrodes specified by a file
    """
    # Read in the significant electrodes
    sig_elec_file = os.path.join(
        os.path.join(os.getcwd(), 'data', args.sig_elec_file))
    sig_elec_list = pd.read_csv(sig_elec_file)

    # Loop over each electrode
    for subject, elec_name in sig_elec_list.itertuples(index=False):

        assert isinstance(subject, int)
        CONV_DIR = '/projects/HASSON/247/data/conversations'
        if args.project_id == 'podcast':
            CONV_DIR = '/projects/HASSON/247/data/podcast'
        BRAIN_DIR_STR = 'preprocessed_all'

        fname = os.path.join(CONV_DIR, 'NY' + str(subject) + '*')
        subject_id = glob.glob(fname)
        assert len(subject_id), f'No data found in {fname}'
        subject_id = os.path.basename(subject_id[0])

        # Read subject's header
        labels = load_header(CONV_DIR, subject_id)
        assert labels is not None, 'Missing header'
        electrode_num = labels.index(elec_name) + 1

        # Read electrode data
        brain_dir = os.path.join(CONV_DIR, subject_id, BRAIN_DIR_STR)
        electrode_file = os.path.join(
            brain_dir, ''.join([
                subject_id, '_electrode_preprocess_file_',
                str(electrode_num), '.mat'
            ]))
        try:
            elec_signal = loadmat(electrode_file)['p1st']
            elec_signal = elec_signal.reshape(-1, 1)
        except FileNotFoundError:
            print(f'Missing: {electrode_file}')
            continue

        # Perform encoding/regression
        encoding_regression(args, datum, elec_signal,
                            str(subject) + '_' + elec_name)

    return


def dumdum1(iter_idx, args, datum, signal, name):
    seed = iter_idx + (os.getenv("SLURM_ARRAY_TASk_ID", 0) * 10000)
    np.random.seed(seed)
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


def this_is_where_you_perform_regression(electrode, args, datum, stitch_index):

    elec_id, elec_name = electrode # get electrode info

    if elec_name is None:
        print(f'Electrode ID {elec_id} does not exist')
        return None

    elec_signal, missing_convos = load_electrode_data(args, elec_id, stitch_index)

    if len(missing_convos) > 0: # signal missing convos
        elec_datum = datum.loc[~datum['conversation_name'].isin(missing_convos)] # filter missing convos
    else:
        elec_datum = datum

    if len(elec_datum) == 0: # no signal
        print(f'{args.sid} {elec_name} No Signal')
        return None
    elif elec_datum.conversation_id.nunique() < 5: # less than 5 convos
        print(f'{args.sid} {elec_name} has less than 5 conversations')
        return None

    # Perform encoding/regression
    if args.phase_shuffle:
        if args.project_id == 'podcast':
            with Pool() as pool:
                corr = pool.map(
                    partial(dumdum1,
                            args=args,
                            datum=elec_datum,
                            signal=elec_signal,
                            name=elec_name), range(args.npermutations))
        else:
            corr = []
            for i in range(args.npermutations):
                corr.append(dumdum1(i, args, elec_datum, elec_signal,
                                    elec_name))

        prod_corr, comp_corr = map(list, zip(*corr))
        write_output(args, prod_corr, elec_name, 'prod')
        write_output(args, comp_corr, elec_name, 'comp')
    else:
        encoding_regression(args, elec_datum, elec_signal, elec_name)

    return None

def parallel_regression(args, electrode_info, datum, stitch_index):
    parallel = True
    if args.emb_type == 'gpt2-xl' and args.sid == 676:
        parallel = False
    if parallel:
        print('Running all electrodes in parallel')
        with Pool(4) as p:
            p.map(
                partial(this_is_where_you_perform_regression,
                    args = args,
                    datum = datum,
                    stitch_index = stitch_index,
                ), electrode_info.items())
    else:
        print('Running all electrodes')
        for electrode in electrode_info.items():
            this_is_where_you_perform_regression(electrode, args, datum, stitch_index)

    return None

def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None

def mod_datum(args, datum):


    if args.datum_mod == "all": # no need for modification
        pass
    elif 'lag' in args.datum_mod: # trim the edges
        half_window = round((args.window_size / 1000) * 512 / 2)
        lag = int(60000 / 1000 * 512) # trim edges with set length
        lag = int(args.lags[-1] / 1000 * 512) # trim edges based on lag
        original_len = len(datum.index)
        datum = datum.loc[((datum['adjusted_onset'] - lag) >= (datum['convo_onset'] + half_window + 1)) & ((datum['adjusted_onset'] + lag) <= (datum['convo_offset'] - half_window - 1))]
        new_datum_len = len(datum.index)
        print(f'Trimming resulted in {new_datum_len} ({round(new_datum_len/original_len*100,5)}%) words')

        # if 'single-conv' in args.datum_mod:
        #     conv_name = "NY676_617_Part2_conversation2" # single-conv
        #     conv_name = "NY676_618_Part5-one_conversation1" # single-conv-2 # 2 hours
        #     conv_name = "NY676_618_Part6_conversation1" # single-conv-3 # 33 min
        #     conv_name = "NY676_617_Part2_conversation3" # single-conv-3 # 10 min
        #     datum = datum.loc[datum.conversation_name == conv_name]
        #     new_datum_len = len(datum.index)
        #     print(f'Selected {conv_name} with {new_datum_len} words')

    elif args.project_id == "tfs" and 'first' in args.datum_mod: # first n word/s in production utterance
        # datum['word_index'] = datum.groupby(datum.production.ne(datum.production.shift()).cumsum()).cumcount().add(1)

        datum2 = datum.sort_values(by='adjusted_onset')
        # get on/offsets of first word in each utterance
        datum.loc[:,'pc_change'] = ~datum.production.eq(datum.production.shift()) # shift between prod/comp
        datum.loc[:,'true_first_word'] = np.where((datum.pc_change) & (datum.word == datum.sentence.str.split(' ').str[0]), 1, -1)
        datum.loc[datum.pc_change == 0,['onset','offset','adjusted_onset','adjusted_offset','true_first_word']] = np.nan
        datum = datum.fillna(method="ffill")

        # truncate datum to start with comprehension
        first_comp = datum[datum.production == 0].index[0] # first comp word index
        datum = datum.loc[first_comp:,:] # get datum after first comprehension

        # get datum in production and only utterances with true first word
        datum = datum.loc[(datum.true_first_word == 1) & (datum.production == 1)]
        word_idx = int(re.findall(r'\b\d+\b',args.datum_mod)[0])

        # get first n word/words in production utterance
        first_indices = datum.loc[datum.pc_change == True].index # first words
        if 'union' in args.datum_mod: # first 1-n words
            first_n_indices = first_indices
            for i in range(1,word_idx):
                first_n_indices = first_n_indices.union(first_indices + i)
            datum = datum.loc[datum.index.intersection(first_n_indices)]
            # datum = datum[datum.word_index <= word_idx]
            print(f'Selected {len(datum.index)} first 1 to {word_idx} words of utterances')
        elif 'inters' not in args.datum_mod: # first n words
            datum = datum[datum.word_index == word_idx]
            print(f'Selected {len(datum.index)} first {word_idx} words of utterances')
        else: # first n words but intersection (only for idx 1 and 2)
            first_word_datum = datum.loc[(datum.word_index_new == 1) & (datum.word == datum.sentence.str.split(' ').str[0]),:]
            datum = datum.loc[datum.index.intersection(first_word_datum.index + 1)]
            if word_idx == 1:
                datum = first_word_datum.loc[first_word_datum.index.intersection(datum.index - 1)]
            print(f'Selected {len(datum.index)} first {word_idx} words of utterances (intersection)')

    elif args.project_id == 'podcast' and args.emb_type == "glove50": # change the args to load gpt2 embedding pickle (only for podcast)
        args.emb_type = 'gpt2-xl'
        args.align_with = 'gpt2-xl'
        args.layer_idx = 48
        stra = 'cnxt_' + str(args.context_length)
        args.emb_file = '_'.join([
            str(args.sid), args.pkl_identifier, args.emb_type, stra,
            f'layer_{args.layer_idx:02d}', 'embeddings.pkl'
        ])
        args.load_emb_file = args.emb_file.replace('__', '_')
        datum_gpt2 = read_datum(args)[['adjusted_onset','top1_pred']]

        # merge gpt2 prediction columns to datum
        datum = datum[datum.adjusted_onset.notna()]
        datum_gpt2 = datum_gpt2[datum_gpt2.adjusted_onset.notna()]
        datum = datum_gpt2.merge(datum, how='inner',on='adjusted_onset')

        # modify datum
        if args.datum_mod == "correct": # select words predicted by gpt2 correctly (top 1 pred)
            datum = datum[datum.word.str.lower() == datum.top1_pred.str.lower().str.strip()] # correct
            print(f'Selected {len(datum.index)} correct words')
        elif args.datum_mod == "incorrect": # select words predicted by gpt2 incorrectly (top 1 pred)
            datum = datum[datum.word.str.lower() != datum.top1_pred.str.lower().str.strip()] # incorrect
            print(f'Selected {len(datum.index)} incorrect words')
        elif args.datum_mod == "gpt2-pred": # for incorrectly predicted words, replace with gpt2 top 1 pred
            glove = api.load('glove-wiki-gigaword-50')
            datum['embeddings'] = datum.top1_pred.str.strip().apply(lambda x: get_vector(x.lower(), glove))
            datum = datum[datum.embeddings.notna()]
            print(f'Changed words into gpt2 top predictions')
        else: # exception
            raise Exception('Invalid Datum Modification')

    assert len(datum.index) > 0, "Empty Datum"
    return datum

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

    # modify datum if needed (args.datum_mod)
    datum = mod_datum(args, datum)

    if args.pca_to:
        print(f'PCAing to {args.pca_to}')
        datum = run_pca(args, datum)

    # Processing significant electrodes or individual subjects
    if args.sig_elec_file:
        process_sig_electrodes(args, datum)
    else:
        electrode_info = process_subjects(args)
        stitch_index = return_stitch_index(args) # load stitch index
        stitch_index = [0] + stitch_index
        parallel_regression(args, electrode_info, datum, stitch_index)
        # this_is_where_you_perform_regression(args, electrode_info, datum)

    return


if __name__ == "__main__":
    main()
