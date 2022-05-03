import glob
import os
import pandas as pd
import csv
from utils import load_pickle
from tfsenc_main import return_stitch_index
from tfsenc_read_datum import load_datum


def write_csv(filename, data):
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)


def main():
    sid = 7170
    DATA_DIR = '/projects/HASSON/247/data/conversations-car'
    process_flag = 'preprocessed'
    elec_num = 256
    datum_name = 'data/tfs/7170/pickles/7170_full_glove50_layer_01_embeddings.pkl'


    convo_paths = sorted(glob.glob(os.path.join(DATA_DIR, str(sid), 'NY*Part*conversation*')))

    data = []

    for convo_path in convo_paths:
        subdata = []
        subdata.append(os.path.basename(convo_path))
        for elec_id in range(1,elec_num+1):
            file = glob.glob(os.path.join(convo_path, process_flag, '*_' + str(elec_id) + '.mat'))
            if len(file) == 1:
                subdata.append(1)
            else:
                subdata.append(0)
        data.append(subdata)

    df = pd.DataFrame(data)
    df = df.rename(columns = {0:'conv'})
    df = df.set_index('conv', drop=True)

    ds = load_pickle('data/tfs/' + str(sid) + '/pickles/'+ str(sid) + '_electrode_names.pkl')
    df2 = pd.DataFrame(ds)

    df.columns = df2.electrode_name

    datum = load_datum(datum_name) # load datum
    print(f'Original datum length is {len(datum)}')
    breakpoint()

    original_len = len(df)
    while original_len > 21:
        remove_name = df.sum(axis=1).sort_values().index[0]
        df = df.drop(labels=remove_name,axis=0)
        datum = datum.loc[datum.conversation_name != remove_name,:]
        original_len = len(df)
        df_subset = df.loc[:,df.sum(axis=0)==original_len] # electrodes with all convos
        df2_subset = df2.loc[df2['electrode_name'].isin(df_subset.columns),:]
        electrode_len = len(df2_subset)
        print(f'Removed {remove_name}. Now we have {original_len} conversations and {electrode_len} electrodes. Datum length is {len(datum)}\nTotal data is {len(datum)*electrode_len}')


    
    breakpoint()


    df_subset = df.loc[:,df.sum(axis=0)!=original_len] # electrodes with all convos
    df2_subset = df2.loc[df2['electrode_name'].isin(df_subset.columns),:] # electrode names with all convos
    df2_out = df2_subset.rename(columns={'electrode_name':'electrode'})
    df2_out = df2_out.loc[:,('subject','electrode')]
    df2_out.to_csv('717_21-conv-elec-67.csv',index=False)

    # df.to_csv('convo_test.csv')

    return 0


if __name__ == "__main__":
    main()