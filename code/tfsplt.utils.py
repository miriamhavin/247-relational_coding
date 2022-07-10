import glob
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy import stats


def read_sig_file(filename, old_results = False):

    sig_file = pd.read_csv('data/' + filename)
    sig_file['sid_electrode'] = sig_file['subject'].astype(str) + '_' + sig_file['electrode']
    elecs = sig_file['sid_electrode'].tolist()

    if old_results: # might need to use this for old 625-676 results
        elecs = sig_file['electrode'].tolist() # no sid name in front

    return set(elecs)


def aggregate_data(file):

    return True