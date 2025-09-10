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
from utils import load_pickle, main_timer, get_git_hash
from reporting import report_space
import json, hashlib
# ---- NEW: subject-level parallel runner ----
from multiprocessing import Pool, cpu_count
import argparse
import yaml
import copy

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
    ds = load_pickle(args.electrode_file_path)
    df = pd.DataFrame(ds)
    df["subject"] = df["subject"].astype("int64")
    print(f"[electrodes] total rows in file: {len(df)}; unique subjects={sorted(df['subject'].unique().tolist())}")
    # --- load prod/comp sig files if provided ---
    whitelist = set()
    if getattr(args, "sig_elec_file_prod", None):
        sig_prod = pd.read_csv(args.sig_elec_file_prod).rename(columns={"electrode": "electrode_name"})
        whitelist |= set(sig_prod["electrode_name"].tolist())
    if getattr(args, "sig_elec_file_comp", None):
        sig_comp = pd.read_csv(args.sig_elec_file_comp).rename(columns={"electrode": "electrode_name"})
        whitelist |= set(sig_comp["electrode_name"].tolist())

    if whitelist:
        before = len(df)
        df = df[df["electrode_name"].isin(whitelist)]
        print(f"[electrodes] whitelist active (n={len(whitelist)}). kept {len(df)}/{before}")
    if getattr(args, "sid", None) is not None:
        before_sid = len(df)
        df = df[df["subject"] == int(args.sid)]
        print(f"[electrodes] sid={args.sid} -> kept {len(df)}/{before_sid} rows")
    electrode_info = {
        (int(row["subject"]), int(row["electrode_id"])): row["electrode_name"]
        for _, row in df.iterrows()
    }
    if not electrode_info:
        print(f"[electrodes] sid={getattr(args,'sid',None)} -> NO ELECTRODES after filters")
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



def _assign_df(df, mapping):
    import numpy as np, pandas as pd
    merged = {}
    for k, v in mapping.items():
        if isinstance(v, str):
            merged[k] = v
            continue
        # numpy/pandas scalar
        if hasattr(v, "item") and np.ndim(v) == 0:
            merged[k] = v.item()
            continue
        # list/array/Series -> squeeze; accept length-1; else stringify to avoid length mismatch
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(v).squeeze()
            if arr.ndim == 0:
                merged[k] = arr.item()
            elif arr.size == 1:
                x = np.ravel(arr)[0]
                merged[k] = x.item() if hasattr(x, "item") else x
            else:
                merged[k] = str(arr.tolist())
        else:
            merged[k] = v
    return df.assign(**merged)



def run_all_electrodes(args, electrode_info, datum, stitch_index):
    # pooled collectors
    pooled = {
        'embedding': {'production': [], 'comprehension': []},
        'neural':    {'production': [], 'comprehension': []},
    }
    all_rows = []

    # dirs
    per_dir    = os.path.join(args.output_dir, "per_electrode")
    pooled_dir = os.path.join(args.output_dir, "pooled")
    plots_dir  = os.path.join(args.output_dir, "plots")
    os.makedirs(per_dir, exist_ok=True)
    os.makedirs(pooled_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    for (sid, elec_id), elec_name in electrode_info.items():
        if elec_name is None:
            print(f"Electrode ID {elec_id} does not exist")
            continue

        # load signals & filter datum
        elec_signal, missing_convos = load_electrode_data(args, elec_id, stitch_index)
        elec_datum = datum.loc[~datum["conversation_name"].isin(missing_convos)] if missing_convos else datum
        if len(elec_datum) == 0:
            print(f"{sid} {elec_name} No Signal")
            continue

        # features
        X = np.stack(elec_datum.embeddings).astype("float32")
        Y = build_Y(
            elec_signal.reshape(-1, 1),
            elec_datum.adjusted_onset.values,
            np.array(args.lags),
            args.window_size,
        ).astype("float32")

        # masks: Speaker1 -> production
        prod_mask = (elec_datum.speaker == "Speaker1").to_numpy()
        comp_mask = ~prod_mask

        # subsets with per-occurrence features
        df_prod = elec_datum.loc[prod_mask].copy()
        df_comp = elec_datum.loc[comp_mask].copy()
        df_prod["embedding"] = list(X[prod_mask, :])
        df_prod["neural"]    = list(Y[prod_mask, :])
        df_comp["embedding"] = list(X[comp_mask, :])
        df_comp["neural"]    = list(Y[comp_mask, :])

        def do_space(df_sub, task_label, skip=False):
            if df_sub.empty:
                return
            # currently only 'neural' modality (intentional)
            for mod, feat_col in [('neural', 'neural')]:
                if len(df_sub[feat_col]) == 0:
                    continue
                feat = np.stack(df_sub[feat_col].values)

                # build space
                space = compute_space(
                    df_sub, feat,
                    min_occ=args.min_occ, remove_global_mean=False,
                    normalize=('unit' if mod == 'embedding' else None)
                )
                # skip if not enough words survived
                if getattr(space, "words", np.array([])).size == 0:
                    return

                if not skip:
                    label = f"{elec_name}_{mod}_{task_label}"
                    df_row = report_space(
                        space, label, outdir=None,
                        B_perm_cols=getattr(args, "B_perm_cols", 49),
                        B_mantel=getattr(args, "B_mantel", 0),
                        seed=getattr(args, "seed", 42),
                    )
                    df_row = df_row.assign(
                        scope='per_electrode',
                        sid=sid,
                        elec_id=elec_id,
                        electrode=str(elec_name),
                        feature_modality=mod,
                        task_modality=task_label,
                    )
                    all_rows.append(df_row)

                # for pooling (word-aligned, averaged). include timing info if present
                pooled_tuple = (space.words, space.start_vecs, space.end_vecs,
                                 getattr(space, 'start_times', None), getattr(space, 'end_times', None))
                pooled[mod][task_label].append(pooled_tuple)

        # run both tasks (skip per-electrode reporting by default)
        do_space(df_prod, "production",  skip=True)
        do_space(df_comp, "comprehension", skip=True)

    # pooled/global per (mod, task)
    for mod in ['neural']:
        for task in ['production', 'comprehension']:
            gspace = pool_aligned(pooled[mod][task], how='intersection')
            glabel = f"global_{mod}_{task}"
            gdf = report_space(
                gspace, glabel, outdir=pooled_dir,   # set outdir=None if you truly don't want files
                B_perm_cols=5000,
                B_mantel=10000,
                seed=getattr(args, "seed", 42),
            )
            gdf = gdf.assign(
                scope='global',
                sid=getattr(args, 'sid', None),
                elec_id=np.nan, electrode='GLOBAL',
                feature_modality=mod, task_modality=task,
            )
            all_rows.append(gdf)

    # return the master table (and optionally write it)
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        # write CSV if you want; or remove this to fully comply with the header comment
        tag = f"sid{getattr(args,'sid','ALL')}"
        all_csv = os.path.join(args.output_dir, f"all_spaces_summary_{tag}.csv")
        all_df.to_csv(all_csv, index=False)
        print(f"[MASTER] Wrote {len(all_df)} rows → {all_csv}")
        return all_df
    else:
        print("[MASTER] No rows to write (no spaces produced).")
        return pd.DataFrame()



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
