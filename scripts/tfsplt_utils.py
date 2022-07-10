import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats


def read_sig_file(filename, old_results=False):

    sig_file = pd.read_csv("data/" + filename)
    sig_file["sid_electrode"] = (
        sig_file["subject"].astype(str) + "_" + sig_file["electrode"]
    )
    elecs = sig_file["sid_electrode"].tolist()

    if old_results:  # might need to use this for old 625-676 results
        elecs = sig_file["electrode"].tolist()  # no sid name in front

    return set(elecs)


def read_data(
    data,
    files,
    sigelecs,
    sigelecs_key,
    load_sid="load_sid",
    mode="mode",
    elec="elec",
    label="label",
):
    for resultfn in files:
        elec = os.path.basename(resultfn).replace(".csv", "")[:-5]
        # Skip electrodes if they're not part of the sig list
        # if 'LGA' not in elec and 'LGB' not in elec: # for 717, only grid
        #     continue
        if len(sigelecs) and elec not in sigelecs[sigelecs_key]:
            continue
        df = pd.read_csv(resultfn, header=None)
        df.insert(0, "sid", load_sid)
        df.insert(0, "mode", mode)
        df.insert(0, "electrode", elec)
        df.insert(0, "label", label)
        data.append(df)
    return data
