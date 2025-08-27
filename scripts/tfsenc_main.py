import csv
import glob
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import os, numpy as np, pandas as pd
from spaces import compute_space, pool_aligned, Space
import numpy as np
import pandas as pd
from tfsenc_config import parse_arguments, setup_environ, write_config
from tfsenc_encoding import (encoding_setup, run_encoding, write_encoding_results, build_Y)
from tfsenc_load_signal import load_electrode_data
from tfsenc_read_datum import read_datum
from utils import load_pickle, main_timer
from reporting import report_space


def return_stitch_index(args):
    """Return convo onset and offset boundaries
    Args:
        args (namespace): arguments

    Returns:
        stitch_index (list): list of convo boundaries
    """
    stitch_index = [0] + load_pickle(args.stitch_file_path)
    return stitch_index


def process_electrodes(args):
    """Process electrodes for subjects (requires electrode list or sig elec file)

    Args:
        args (namespace): commandline arguments

    Returns:
        electrode_info (dict): each item in the format (sid, elec_id): elec_name
    """
    ds = load_pickle(args.electrode_file_path)
    df = pd.DataFrame(ds)
    if args.sig_elec_file is not None:  # sig elec files
        sig_elec_list = pd.read_csv(args.sig_elec_file_path).rename(
            columns={"electrode": "electrode_name"}
        )
        df["subject"] = df.subject.astype("int64")
        sid_sig_elec_list = pd.merge(
            df, sig_elec_list, how="inner", on=["subject", "electrode_name"]
        )
        assert len(sig_elec_list) == len(sid_sig_elec_list), "Sig Elecs Missing"
        electrode_info = {
            (values["subject"], values["electrode_id"]): values["electrode_name"]
            for _, values in sid_sig_elec_list.iterrows()
        }

    else:  # electrode list for 1 sid
        assert len(args.elecs > 0), "Need electrode list since no sig_elec_list"
        electrode_info = {
            (args.sid, key): next(
                iter(
                    df.loc[
                        (df.subject == str(args.sid)) & (df.electrode_id == key),
                        "electrode_name",
                    ]
                ),
                None,
            )
            for key in args.elecs
        }

    return electrode_info


def skip_elecs_done(summary_file, electrode_info):
    """Skip electrodes with encoding results already

    Args:
        summary_file (string): path to summary file of previous jobs
        electrode_info (dict): dictionary of electrodes

    Returns:
        electrode_info (dict): dictionary of electrodes without electrodes with encoding results
    """

    summary = pd.read_csv(summary_file, header=None)
    elecs_num = len(electrode_info)

    elecs_done = summary.iloc[:, 1].tolist()
    for elec in elecs_done:  # skipping electrodes
        print(f"Skipping elec {elec}")
        electrode_info = {
            key: val for key, val in electrode_info.items() if f"{key[0]}_{val}" != elec
        }
        elecs_num -= 1

    assert elecs_num == len(electrode_info), "Wrong number of elecs skipped"
    return electrode_info


def run_single(df, feat, outdir, label, min_occ=10, remove_global_mean=False):
    space = compute_space(df, feat, min_occ=min_occ, remove_global_mean=remove_global_mean)
    report_space(space, label, outdir)

def run_all_electrodes(args, electrode_info, datum, stitch_index):
    pooled = {'embedding': [], 'neural': []}
    all_rows = []  # collect all one-row summaries here

    for (sid, elec_id), elec_name in electrode_info.items():
        if elec_name is None:
            print(f"Electrode ID {elec_id} does not exist")
            continue

        # --- load signals & filter datum
        elec_signal, missing_convos = load_electrode_data(args, elec_id, stitch_index)
        elec_datum = datum.loc[~datum["conversation_name"].isin(missing_convos)] if missing_convos else datum
        if len(elec_datum) == 0:
            print(f"{sid} {elec_name} No Signal")
            continue

        # --- build features
        X = np.stack(elec_datum.embeddings).astype("float32")
        Y = build_Y(
            elec_signal.reshape(-1, 1),
            elec_datum.adjusted_onset.values,
            np.array(args.lags),
            args.window_size,
        ).astype("float32")

        elec_df = elec_datum.copy()
        elec_df["embedding"] = list(X)
        elec_df["neural"]    = list(Y)

        # --- compute spaces (no per-electrode save)
        for mod, feat_col in [('embedding','embedding'), ('neural','neural')]:
            feat  = np.stack(elec_df[feat_col].values)
            space = compute_space(elec_df, feat, min_occ=args.min_occ, remove_global_mean=False)
            df_row = report_space(space, f"{elec_name}_{mod}", outdir=None)
            df_row = df_row.assign(scope='per_electrode',
                                   sid=sid,
                                   elec_id=elec_id,
                                   electrode=str(elec_name),
                                   modality=mod)
            all_rows.append(df_row)

            if space.words.size:
                pooled[mod].append((space.words, space.start_vecs, space.end_vecs))

    # --- pooled/global
    for mod in ['embedding','neural']:
        if pooled[mod]:
            gspace = pool_aligned(pooled[mod], how='intersection')
            gdf = report_space(gspace, f"global_{mod}", outdir=None)
            gdf = gdf.assign(scope='global',
                             sid=np.nan,
                             elec_id=np.nan,
                             electrode='GLOBAL',
                             modality=mod)
            all_rows.append(gdf)

    # --- save one big CSV
    big = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    outpath = os.path.join(args.output_dir, "all_spaces_summary.csv")
    big.to_csv(outpath, index=False)
    print(f"[ALL] consolidated summary saved → {outpath}")

    return big

@main_timer
def main():
    args, yml_args = parse_arguments()
    args = setup_environ(args)
    write_config(args, yml_args)
    stitch_index = return_stitch_index(args)
    datum = read_datum(args, stitch_index)
    electrode_info = process_electrodes(args)
    run_all_electrodes(args, electrode_info, datum, stitch_index)
    return


if __name__ == "__main__":
    main()
