import csv
import glob
import os
from functools import partial
from multiprocessing import Pool

import mat73
import numpy as np
import pandas as pd
from numba import jit
from scipy import stats
from scipy.io import loadmat
from tfsenc_parser import parse_arguments
from tfsenc_read_datum import read_datum
from tfsenc_utils import setup_environ
from utils import main_timer, write_config
from tfsenc_main import write_output, mod_datum, process_subjects
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def erp(args, datum, elec_signal, name):
    datum = datum[datum.adjusted_onset.notna()]

    datum_comp = datum[datum.speaker != 'Speaker1'] # comprehension data
    datum_prod = datum[datum.speaker == 'Speaker1'] # production data
    print(f'{args.sid} {name} Prod: {len(datum_comp.index)} Comp: {len(datum_prod.index)}')

    erp_comp = calc_average(args.lags, datum_comp, elec_signal) # calculate average erp
    erp_prod = calc_average(args.lags, datum_prod, elec_signal) # calculate average erp

    print(f'writing output for electrode {name}')
    write_output(args, erp_comp, name, 'comp')
    write_output(args, erp_prod, name, 'prod')
    
    return


def calc_average(lags, datum, brain_signal):
    """[summary]
    Args:
        args ([type]): [description]
        datum ([type]): [description]
        brain_signal ([type]): [description]

    Returns:
        [type]: [description]
    """
    onsets = datum.adjusted_onset.values
    erp = np.zeros((len(onsets), len(lags)))

    for lag_idx, lag in enumerate(lags): # loop through each lag
        lag_amount = int(lag / 1000 * 512)
        index_onsets = np.round_(onsets, 0, onsets) + lag_amount # take correct idx for all words
        index_onsets = index_onsets.astype(int) # uncomment this if not running jit
        erp[:,lag_idx] = brain_signal[index_onsets].reshape(-1) # take the signal for that lag

    erp = [np.mean(erp,axis=(0),dtype=np.float64).tolist()] # average by words

    return erp


def load_and_erp(electrode, args, datum):
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
    
    elec_id, elec_name = electrode # get electrode info

    if elec_name is None:
        print(f'Electrode ID {elec_id} does not exist')
        return
    convos = sorted(glob.glob(os.path.join(DATA_DIR, str(args.sid), '*')))
        
    all_signal = []
    for convo_id, convo in enumerate(convos, 1):
        if args.conversation_id != 0 and convo_id != args.conversation_id:
            continue

        file = glob.glob(
            os.path.join(convo, process_flag, '*_' + str(elec_id) + '.mat'))[0]

        mat_signal = loadmat(file)['p1st']
        mat_signal = mat_signal.reshape(-1, 1)

        if mat_signal is None:
            continue

        # Detrending
        detrend = True
        if detrend:
            y = mat_signal
            X = np.arange(len(y)).reshape(-1,1)
            pf = PolynomialFeatures(degree=2)
            Xp = pf.fit_transform(X)

            model = LinearRegression()
            model.fit(Xp, y)
            trend = model.predict(Xp)
            mat_signal = y - trend
        
        # z-score
        z_score = True
        if z_score:
            mat_signal = stats.zscore(mat_signal)

        all_signal.append(mat_signal)

    if args.project_id == 'tfs':
        elec_signal = np.vstack(all_signal)
    else:
        elec_signal = np.array(all_signal)
    
    elec_signal = elec_signal.reshape(-1, 1)

    erp(args, datum, elec_signal, elec_name)

    return


def load_and_erp_parallel(args, electrode_info, datum):
    parallel = True
    if parallel:
        print('Running all electrodes in parallel')
        with Pool(4) as p:
            p.map(
                partial(load_and_erp,
                    args = args,
                    datum = datum
                ), electrode_info.items())
    else:
        for index, subject, elec_name in electrode_info.itertuples(index=True):
            load_and_erp(index, subject, elec_name, args, datum)


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
    datum = datum.drop('embeddings', 1) # trim datum to smaller size

    assert args.sig_elec_file == None, "Do not input significant electrode list"
    electrode_info = process_subjects(args)
    load_and_erp_parallel(args, electrode_info, datum)

    return

if __name__ == "__main__":
    main()
