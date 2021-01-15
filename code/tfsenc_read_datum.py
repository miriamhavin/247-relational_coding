import os
import pickle

import numpy as np
import pandas as pd


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        dict/list: pickle contents returned as dataframe
    """
    with open(file, 'rb') as fh:
        datum = pickle.load(fh)

    return datum


def drop_nan_embeddings(df):
    """Drop rows containing all nan's for embedding
    """
    df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())
    df = df[~df['is_nan']]

    return df


def read_datum(args):
    """Read and process the datum based on input arguments

    Args:
        args (namespace): commandline arguments

    Raises:
        Exception: args.word_value should be one of ['top', 'bottom', 'all']

    Returns:
        DataFrame: processed datum
    """
    file_name = os.path.join(args.PICKLE_DIR, str(args.sid),
                             args.load_emb_file)
    datum = load_pickle(file_name)

    df = pd.DataFrame.from_dict(datum)
    df = drop_nan_embeddings(df)

    # use columns where token is root
    if 'gpt2' in [args.align_with, args.emb_type]:
        df = df[df['gpt2_token_is_root']]
    elif 'bert' in [args.align_with, args.emb_type]:
        df = df[df['bert_token_is_root']]
    else:
        pass

    df = df[~df['glove50_embeddings'].isna()]

    # if encoding is on glove embeddings copy them into 'embeddings' column
    if args.emb_type == 'glove50':
        df['embeddings'] = df['glove50_embeddings']

    return df
