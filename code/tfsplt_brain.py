import glob
import os

import numpy as np
import pandas as pd

sub = "676"

keys = ["prod", "comp"]

corr = "ind"

formats = [
    "results/tfs/kw-tfs-full-" + sub + "-glove50-triple/kw-200ms-all-" + sub + "/",
    "results/tfs/kw-tfs-full-" + sub + "-gpt2-xl-triple/kw-200ms-all-" + sub + "/",
    "results/tfs/kw-tfs-full-"
    + sub
    + "-blenderbot-small-triple/kw-200ms-all-"
    + sub
    + "/",
]

comp_sig_file = "data/tfs-sig-file-" + sub + "-sig-1.0-comp.csv"
prod_sig_file = "data/tfs-sig-file-" + sub + "-sig-1.0-prod.csv"

filename = "data/" + sub + "_" + corr + ".txt"

#################################################################################################
###################################### Compare two sources ######################################
#################################################################################################

data = pd.read_csv(filename, sep=" ", header=None)
data = data.set_index(0)
data = data.loc[:, 1:4]
print(f"\nFor subject {sub}:\ntxt has {len(data.index)} electrodes")

files = glob.glob(formats[0] + "*_prod.csv")
files = [os.path.basename(file) for file in files]
print(f"encoding has {len(files)} electrodes")

files = pd.DataFrame(data=files)
files["elec"] = ""
for row, values in files.iterrows():
    elec = os.path.basename(values[0]).replace(".csv", "")[:-8]
    elec = elec.replace("EEG", "").replace("GR_", "G").replace("_", "")
    files.loc[row, "elec"] = elec
files = files.set_index("elec")

set1 = set(data.index)
set2 = set(files.index)
print(f"txt and encoding share {len(set1.intersection(set2))} electrodes\n")
print(f"encoding does not have these electrodes: {sorted(set1-set2)}\n")
print(f"txt does not have these electrodes: {sorted(set2-set1)}\n")

#############################################################################################
###################################### Getting Results ######################################
#############################################################################################

df = pd.merge(data, files, left_index=True, right_index=True)

comp_sig_elecs = pd.read_csv(comp_sig_file)["electrode"].tolist()
comp_sig_elecs = [
    elec.replace("EEG", "").replace("GR_", "G").replace("_", "").replace("REF", "")
    for elec in comp_sig_elecs
]
prod_sig_elecs = pd.read_csv(prod_sig_file)["electrode"].tolist()
prod_sig_elecs = [
    elec.replace("EEG", "").replace("GR_", "G").replace("_", "").replace("REF", "")
    for elec in prod_sig_elecs
]

embs = ["glove", "gpt", "bbot"]

emb_key = [emb + "_" + key for emb in embs for key in keys]
for col in emb_key:
    df[col] = -1


def get_max(filename, path):
    filename = os.path.join(path, filename)
    elec_data = pd.read_csv(filename, header=None)
    return max(elec_data.loc[0])


for format in formats:
    if "glove50" in format:
        col_name = "glove"
    elif "gpt2-xl" in format:
        col_name = "gpt"
    elif "blenderbot-small" in format:
        col_name = "bbot"
    print(f"getting results for {col_name} embedding")
    for row, values in df.iterrows():
        col_name1 = col_name + "_prod"
        col_name2 = col_name + "_comp"
        prod_name = values[0]
        comp_name = values[0].replace("prod", "comp")
        if row in prod_sig_elecs:
            df.loc[row, col_name1] = get_max(prod_name, format)
        if row in comp_sig_elecs:
            df.loc[row, col_name2] = get_max(comp_name, format)


for col in emb_key:
    output_filename = "results/cor_tfs/" + sub + "_" + corr + "_" + col + ".txt"
    df_output = df.loc[:, [1, 2, 3, 4, col]]
    with open(output_filename, "w") as outfile:
        df_output.to_string(outfile)
