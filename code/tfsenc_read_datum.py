import os
import pickle

import numpy as np
import pandas as pd

# import statistics


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


def read_datum(args):
    """Read and process the datum based on input arguments

    Args:
        args (namespace): commandline arguments

    Raises:
        Exception: args.word_value should be one of ['top', 'bottom', 'all']

    Returns:
        DataFrame: processed datum
    """
    file_name = os.path.join(args.PICKLE_DIR, str(args.sid), args.emb_file)
    datum = load_pickle(file_name)

    df = pd.DataFrame.from_dict(datum)
    # df = df.dropna(subset=['embeddings'])

    # is_embedding_nan()
    df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())

    # drop empty embeddings
    df = df[~df['is_nan']]

    # use columns where token is root
    if 'gpt2' in args.emb_type:
        df = df[df['gpt2_token_is_root']]
    elif 'bert' in args.emb_type:
        df = df[df['bert_token_is_root']]
    else:
        pass

    return df
