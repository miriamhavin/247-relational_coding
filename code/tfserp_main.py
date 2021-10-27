import csv
import glob
import os
from functools import partial
from multiprocessing import Pool
import argparse

import mat73
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tfsenc_parser import parse_arguments
from tfsenc_read_datum import read_datum
from tfsenc_utils import (append_jobid_to_string, setup_environ)
from utils import load_pickle, main_timer, write_config

def erp(args, datum, elec_signal, name):
    output_dir = args.full_output_dir
    datum = datum[datum.adjusted_onset.notna()]

    erp = calc_average(args, datum, elec_signal) # calculate average erp

    trial_str = append_jobid_to_string(args, 'comp')
    filename = os.path.join(output_dir, name + trial_str + '.csv')
    with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(erp)
    
    return


def calc_average(args, datum, brain_signal):
    """[summary]
    Args:
        args ([type]): [description]
        datum ([type]): [description]
        brain_signal ([type]): [description]

    Returns:
        [type]: [description]
    """
    word_onsets = datum.adjusted_onset.values
    convo_onsets = datum.convo_onset.values
    convo_offsets = datum.convo_offset.values
    half_window = round((args.window_size / 1000) * 512 / 2)
    brain_signal = brain_signal.reshape(-1, 1)

    erp = np.zeros((len(word_onsets), half_window * 2 + 1))

    index_onsets = np.minimum(
        convo_offsets - half_window - 1,
        np.maximum(convo_onsets + half_window + 1,
                    np.round_(word_onsets, 0, word_onsets))).astype(int)
    starts = index_onsets - half_window - 1
    stops = index_onsets + half_window

    for i, (start, stop) in enumerate(zip(starts, stops)):
        erp[i, :] = brain_signal[start:stop].reshape(-1) # take brain signal inside window
    
    erp = [np.mean(erp,axis=(0),dtype=np.float64).tolist()] # average by words

    return erp


def elec_name_to_id(convo_dir, elec_name):
    misc_dir = os.path.join(convo_dir, 'misc')
    header_file = os.path.join(misc_dir, os.path.basename(convo_dir) + '_header.mat')
    if not os.path.exists(header_file):
        print(f'[WARN] no header found in {misc_dir}')
        return
    header = mat73.loadmat(header_file)
    labels = header.header.label

    assert labels is not None, 'Missing header'

    elec_id = labels.index(elec_name) + 1
    return elec_id


def load_and_erp(args, elec_list, datum):
    """
    """
    if args.project_id == 'tfs':
        DATA_DIR = '/projects/HASSON/247/data/conversations-car'
        process_flag = 'preprocessed'
    elif args.project_id == 'podcast':
        DATA_DIR = '/projects/HASSON/247/data/podcast-data'
        process_flag = 'preprocessed_all'
    else:
        raise Exception('Invalid Project ID')
    
    for index, subject, elec_name in elec_list.itertuples(index=True):

        assert isinstance(subject, int) 

        elec_id = index + 1 # get electrode id

        if elec_name is None:
            print(f'Electrode ID {elec_id} does not exist')
            continue
        convos = sorted(glob.glob(os.path.join(DATA_DIR, str(subject), '*')))
        
        all_signal = []
        for convo_id, convo in enumerate(convos, 1):
            if args.conversation_id != 0 and convo_id != args.conversation_id:
                continue

            if args.sig_elec_file: # get electrode id from electrode name
                elec_id = elec_name_to_id(convo, elec_name)

            file = glob.glob(
                os.path.join(convo, process_flag, '*_' + str(elec_id) + '.mat'))[0]

            mat_signal = loadmat(file)['p1st']
            mat_signal = mat_signal.reshape(-1, 1)

            if mat_signal is None:
                continue
            all_signal.append(mat_signal)

        if args.project_id == 'tfs':
            elec_signal = np.vstack(all_signal)
        else:
            elec_signal = np.array(all_signal)

        erp(args, datum, elec_signal, elec_name)

    return

def process_electrodes(args, datum):
    """Run encoding on elctrodes specified by the subject id and electrode list
    """
    ds = load_pickle(os.path.join(args.PICKLE_DIR, args.electrode_file))
    df = pd.DataFrame(ds)

    if args.electrodes:
        electrode_info = {
            key: next(
                iter(df.loc[(df.subject == str(args.sid)) &
                            (df.electrode_id == key), 'electrode_name']), None)
            for key in args.electrodes
        } # electrode info is a dict with electrode_id : electrode_name
    
    assert args.sid != 777, 'Please provide sig_elec_file'

    elec_list = pd.DataFrame({'subject': [args.sid] * len(electrode_info.values()),
                'electrode': electrode_info.values()
                }) # organize into a pd dataframe
    
    load_and_erp(args, elec_list, datum)

    return


def process_sig_electrodes(args, datum):
    """Run encoding on select significant elctrodes specified by a file
    """
    # Read in the significant electrodes
    sig_elec_file = os.path.join(
        os.path.join(os.getcwd(), 'data', args.sig_elec_file))
    sig_elec_list = pd.read_csv(sig_elec_file)

    load_and_erp(args, sig_elec_list, datum)

    return


@main_timer
def main():
    
    # Read command line arguments
    args = parse_arguments()
    assert 'gpt2' in args.emb_type, 'Need gpt2 embedding' # use GPT2 embedding pickle

    # Setup paths to data
    args = setup_environ(args)

    # Saving configuration to output directory
    write_config(vars(args))

    # Locate and read datum
    datum = read_datum(args)

    if args.split:
        if args.split == "correct":
            datum = datum[datum.word.str.lower() == datum.top1_pred.str.lower().str.strip()] # correct
            print(f'Selected {len(datum.index)} correct words')
        elif args.split == "incorrect":
            datum = datum[datum.word.str.lower() != datum.top1_pred.str.lower().str.strip()] # incorrect
            print(f'Selected {len(datum.index)} incorrect words')

    # Processing significant electrodes or individual subjects
    if args.sig_elec_file:
        process_sig_electrodes(args, datum)
    else:
        process_electrodes(args, datum)

    return

if __name__ == "__main__":
    main()
