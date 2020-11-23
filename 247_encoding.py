import argparse
import glob
import os
import sys
from datetime import datetime

import mat73
import numpy as np
import pandas as pd
from scipy.io import loadmat

from encoding_247_utils import build_XY, encode_lags_numba

# % Encode all of data for a patient.

# % we need to build a matrix X of nx50 and Y of nxlags
# % separately for production and comprehension
# % where n is the number of words in all conversations
# % separately per electrode

hostname = os.environ['HOSTNAME']

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=None)
parser.add_argument('--output-folder', type=str, default=None)
parser.add_argument('--lags', nargs='+', type=int)
parser.add_argument('--emb-file', type=str, default=None)
parser.add_argument('--window-size', type=int, default=50)
parser.add_argument('--electrode', type=int, default=None)
parser.add_argument('--npermutations', type=int, default=1)
parser.add_argument('--shuffle', action='store_true', default=False)

# parser.add_argument('--sig-elec-name', type=str, default=None)
# parser.add_argument('--nonWords', action='store_false', default=True)
# parser.add_argument(
#     '--datum-emb-fn',
#     type=str,
#     default='podcast-datum-gpt2-xl-c_1024-previous-pca_50d.csv')
# parser.add_argument('--sid', type=int, default=None)
# parser.add_argument('--gpt2', type=int, default=None)
# parser.add_argument('--bert', type=int, default=None)
# parser.add_argument('--bart', type=int, default=None)
# parser.add_argument('--glove', type=int, default=1)

# parser.add_argument('--npermutations', type=int, default=5000)
args = parser.parse_args()

if "tiger" in hostname:
    tiger = 1
elif "scotty" in hostname:
    tiger = 0
    PROJ_DIR = '/mnt/bucket/labs/hasson/ariel/247/'
    conv_folder = 'conversations_car'
    conv_dir = os.path.join(PROJ_DIR, 'conversation_space/', conv_folder)
    emb_dir = os.path.join(PROJ_DIR, 'models/embeddings/')
    outdir = os.getcwd()
    datumdir_gpt2 = '/mnt/sink/scratch/247/contextual-embeddings-results/gpt2-xl-c_25_conversations/'
    datumdir_bert = '/mnt/sink/scratch/247/contextual-embeddings-results/bert-large-uncased-whole-word-masking-c_25_conversations_hs'
else:
    print("unknown host")
    sys.exit()

if args.subject is None:
    print("Please Enter a valid subject ID")
    sys.exit()

if args.output_folder is None:
    print("Please Enter a valid output folder name")
    sys.exit()

convs = glob.glob(os.path.join(datumdir_bert, 'NY' + str(args.subject) + '*'))

if args.subject == 625:
    header = mat73.loadmat(
        os.path.join(
            conv_dir,
            'NY625_418_Part3_conversation1/misc/NY625_418_Part3_conversation1_header.mat'
        ))
elif args.subject == 676:
    header = mat73.loadmat(
        os.path.join(
            conv_dir,
            'NY676_616_Part1_conversation1/misc/NY676_616_Part1_conversation1_header.mat'
        ))
elif args.subject == 717:
    header = mat73.loadmat(
        os.path.join(
            conv_dir,
            'NY717_311_Part5_conversation3/misc/NY717_311_Part5_conversation3_header.mat'
        ))
else:
    print("Subject doesn't exist")
    sys.exit()

electrode_name = header.header.label[args.electrode]

start_time = datetime.now()
print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

all_X, all_Y, all_df = [], [], []
for full_conv in convs:
    conv_name = os.path.split(full_conv)[-1]
    print(conv_name)

    signal_file = glob.glob(
        os.path.join(conv_dir, conv_name, 'preprocessed',
                     ''.join([conv_name, '*_',
                              str(args.electrode), '.mat'])))[0]

    if not os.path.exists(signal_file):
        print(f'Electrode {args.electrode} not found in {conv_name}')
        continue

    signal = loadmat(signal_file)['p1st']

    # get datum embeddings from first model
    datum_fn = os.path.join(datumdir_bert, conv_name, 'datum.csv')
    if not os.path.exists(datum_fn):
        continue

    df = pd.read_csv(datum_fn, skiprows=1)
    df.columns = ['word', 'onset', 'offset', 'accuracy', 'speaker'] + list(
        map(str, range(df.shape[1] - 5)))

    # I added this
    df_cols = df.columns.tolist()
    embedding_columns = df_cols[df_cols.index('0'):]
    df = df[~df['word'].isin(['sp', '{lg}', '{ns}', '{inaudible}'])]
    df = df.dropna()

    df['embeddings'] = df[embedding_columns].values.tolist()
    df = df.drop(columns=embedding_columns)
    
    X, Y = build_XY(df, signal, args.lags, args.window_size)
    
    all_X.append(X)
    all_Y.append(Y)
    all_df.append(df)

all_X = np.vstack(all_X)
all_Y = np.vstack(all_Y)
all_df = pd.concat(all_df, ignore_index=True)

prod_X = all_X[all_df.speaker == 'Speaker1', :]
comp_X = all_X[all_df.speaker == 'Speaker2', :]

prod_Y = all_Y[all_df.speaker == 'Speaker1', :]
comp_Y = all_Y[all_df.speaker == 'Speaker2', :]

print(prod_X.shape, prod_Y.shape, comp_X.shape, comp_Y.shape)

# run permutation
if prod_X.shape[0]:
    prod_rp = np.stack(
        [encode_lags_numba(prod_X, prod_Y) for _ in range(args.npermutations)])
else:
    print('Not encoding production due to lack of examples')

if comp_X.shape[0]:
    comp_rp = np.stack(
        [encode_lags_numba(comp_X, comp_Y) for _ in range(args.npermutations)])
else:
    print('Not encoding comprehension due to lack of examples')

print(prod_rp.shape, comp_rp.shape)

# TODO Plotting
# Saving output

end_time = datetime.now()
print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
