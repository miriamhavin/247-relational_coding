import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tfsenc_main import setup_environ
from tfsenc_read_datum import load_pickle


def run_pca():
    pass


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', nargs='?', type=int, default=None)
    parser.add_argument('--emb-type', type=str, default=None)
    parser.add_argument('--reduce-to', type=int, default=1)

    args = parser.parse_args()

    return args


def main():
    run_pca()


if __name__ == "__main__":
    args = parse_arguments()
    args = setup_environ(args)

    file_name, file_ext = os.path.splitext(args.emb_file)
    pca_output_file_name = file_name + '-pca' + str(args.reduce_to) + file_ext

    pca = PCA(n_components=args.reduce_to, svd_solver='auto')

    file_name = os.path.join(args.PICKLE_DIR, str(args.sid), args.emb_file)
    datum = load_pickle(file_name)

    df = pd.DataFrame.from_dict(datum)
    print(df.tail())
    raise Exception()
    print(df.sentence.tolist()[-1])
    # Reading only the embedding columns from the dataframe
    df_emb = df['embeddings']
    print(df_emb.shape)

    # Keep aside the first part of the data_frame for later concatenation
    df_meta = df.drop(columns=['embeddings'])

    df_emb = df_emb.dropna()
    
    print(df[df.isnull().any(axis=1)])
    print(df_emb.shape)
    raise Exception()
    # is_embedding_nan()
    df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())

    # drop empty embeddings
    df = df[~df['is_nan']]

    # Transform the data
    output1 = pca.fit_transform(df_c_orig)

    # saving the outputs back into a csv files
    out_columns = [str(item) for item in range(50)]
    output1_df = pd.DataFrame(output1,
                              index=df_c_orig.index,
                              columns=out_columns)

    sklearn_df = df_c_orig_meta.join(output1_df)
    sklearn_df.to_csv(pca_output_file_name, index=False)
