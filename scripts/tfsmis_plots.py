import argparse
import csv
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# TODO: this file is work in progress
plt.rcParams.update({"text.usetex": True})


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe
    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py
    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(file, "rb") as fh:
        datum = pickle.load(fh)

    return datum


def extract_correlations(args, directory_list, file_str=None):
    """[summary]

    Args:
        directory_list ([type]): [description]
        file_str ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    # Load the subject's electrode file
    electrode_list = load_pickle(args.electrode_file)["electrode_name"]
    if len(args.electrodes):
        electrode_list = [electrode_list[i - 1] for i in args.electrodes]

    if args.sig_elec_file is not None:
        elec_file = os.path.join(os.path.join(os.getcwd(), "data", args.sig_elec_file))
        elecs = pd.read_csv(elec_file)
        electrode_list = elecs.subject.astype(str) + "_" + elecs.electrode
        electrode_list = elecs.electrode  # NOTE for 247

    all_corrs = []
    for dir in directory_list:
        dir_corrs = []
        for electrode in electrode_list:
            file = os.path.join(dir, electrode + "_" + file_str + ".csv")
            try:
                with open(file, "r") as csv_file:
                    ha = list(map(float, csv_file.readline().strip().split(",")))
                dir_corrs.append(ha)
            except FileNotFoundError:
                file = os.path.join(dir, "*" + electrode + "_" + file_str + ".csv")
                files = glob.glob(file)
                if len(files):
                    file = files[0]
                    with open(file, "r") as csv_file:
                        ha = list(map(float, csv_file.readline().strip().split(",")))
                    dir_corrs.append(ha)
                elif len(args.electrodes) or args.sig_elec_file is not None:
                    print(f"{electrode} not found for {file_str} under {dir}")

        all_corrs.append(dir_corrs)

    all_corrs = np.stack(all_corrs)
    mean_corr = np.mean(all_corrs, axis=1)

    if not all_corrs.size:
        print("[WARN] no results found!")

    return all_corrs, mean_corr, electrode_list


def save_max_correlations(args, prod_max, comp_max, prod_list):
    """[summary]

    Args:
        args ([type]): [description]
        prod_max ([type]): [description]
        comp_max ([type]): [description]
        prod_list ([type]): [description]
    """
    df = pd.DataFrame(prod_max, columns=["production"])
    df["comprehension"] = comp_max
    df["electrode"] = [int(item.strip("elec")) for item in prod_list]
    df = df[["electrode", "production", "comprehension"]]
    df.to_csv(args.max_corr_csv, index=False)
    return


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="test")
    parser.add_argument("--input-directory", nargs="*", type=str, default=None)
    parser.add_argument("--labels", nargs="*", type=str, default=None)
    parser.add_argument("--embedding-type", type=str, default=None)
    parser.add_argument("--electrodes", nargs="*", type=int, default=[])
    parser.add_argument("--lags", nargs="*", type=int, default=[])
    parser.add_argument("--output-file-name", type=str, default=None)

    # group = parser.add_mutually_exclusive_group()
    parser.add_argument("--sid", nargs="?", type=int, default=None)
    parser.add_argument("--sig-elec-file", nargs="?", type=str, default=None)

    args = parser.parse_args()

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    return args


def initial_setup(args):
    assert len(args.input_directory) == len(args.labels), "Unequal number of"

    full_input_dir = os.path.join(os.getcwd(), "Results", args.input_directory)
    args.output_pdf = os.path.join(
        os.getcwd(), "_".join([str(args.sid), args.embedding_type, "encoding.pdf"])
    )
    args.max_corr_csv = os.path.join(
        full_input_dir,
        "_".join([str(args.sid), args.embedding_type, "maxCorrelations.csv"]),
    )
    return


def set_plot_styles(args):
    linestyles = ["-", "--"]
    color = ["b", "r", "g"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    linestyles = np.repeat(linestyles[0 : len(args.labels)], len(args.labels))
    colors = colors[0 : len(args.labels)] * 2

    # hack?
    if len(linestyles) == 1:
        linestyles = ["-", "--"]

    return (colors, linestyles)


def set_legend_labels(args):
    legend_labels = []

    for item in ["production", "comprehension"]:
        for label in args.labels:
            legend_labels.append(r"\textit{" + "-".join([label, item]) + "}")
    return legend_labels


def plot_data(args, data, pp, title=None, asstr=True):
    lags = np.arange(-2000, 2001, 25, dtype=int)
    # lags = np.array([-60000] + lags.tolist() + [60000], dtype=int)
    # lags = np.asarray(args.lags)
    lags = lags / 1000

    # fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    # ax = axes[0]
    fig, ax = plt.subplots()
    color, linestyles = set_plot_styles(args)
    ax.set_prop_cycle(color=color, linestyle=linestyles)
    ax.plot(lags, data.T, linewidth=0.75)

    ax.legend(set_legend_labels(args), frameon=False, loc="upper left")

    # df = pd.read_csv('avigail.txt', sep='\t')
    # ax.plot(df.lags, df.means, label='avigail', c='g')
    # print(df.describe())
    # breakpoint()

    ax.set(xlabel=r"\textit{lag (s)}", ylabel=r"\textit{correlation}", title=title)
    ax.set_ylim(-0.05, 0.35)
    ax.vlines(0, -0.05, 0.50, linestyles="dashed", linewidth=0.75)

    elecdir = f"/projects/HASSON/247/data/elecimg/{args.sid}/"
    elecname = (
        title.replace("EEG", "").replace("REF", "").replace("\\", "").replace("_", "")
    )
    elecname = elecname.replace("GR", "G")
    imname = elecdir + f"{args.sid}_{elecname}.png"
    imname = elecdir + f"thumb_{elecname}.png"
    if os.path.isfile(imname):
        arr_image = plt.imread(imname, format="png")
        fig.figimage(
            arr_image,
            fig.bbox.xmax - arr_image.shape[1],
            fig.bbox.ymax - arr_image.shape[0],
            zorder=5,
        )

    else:
        print("Missing", imname)

    pp.savefig(fig)
    plt.close()


def plot_average_correlations_multiple(pp, prod_corr_mean, comp_corr_mean, args):
    data = np.vstack([prod_corr_mean, comp_corr_mean])
    plot_data(args, data, pp, r"\textit{Average Correlation (all electrodes)}")


def plot_individual_correlation_multiple(pp, prod_corr, comp_corr, prod_list, args):
    prod_list = [item.replace("_", "\_") for item in prod_list]
    prod_corr = np.moveaxis(prod_corr, [0, 1, 2], [1, 0, 2])
    comp_corr = np.moveaxis(comp_corr, [0, 1, 2], [1, 0, 2])

    for prod_row, comp_row, electrode_id in zip(prod_corr, comp_corr, prod_list):
        data = np.vstack([prod_row, comp_row])
        plot_data(args, data, pp, electrode_id)


if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()

    args.output_pdf = os.path.join(
        os.getcwd(), "results", "figures", args.output_file_name + ".pdf"
    )
    args.electrode_file = os.path.join(
        os.getcwd(),
        "data",
        args.project_id,
        str(args.sid),
        "pickles",
        str(args.sid) + "_electrode_names.pkl",
    )

    assert len(args.input_directory) <= len(args.labels), "Unequal number of"

    # Results folders to be plotted
    results_dirs = []
    for directory in args.input_directory:
        fn = os.path.join(
            os.getcwd(), "results", args.project_id, directory, str(args.sid)
        )
        matches = glob.glob(fn)
        if len(matches) > 0:
            results_dirs += matches
        else:
            print(f"No results found under {directory}")

    # NOTE
    prod_corr_mean = None
    prod_corr, prod_corr_mean, prod_list = extract_correlations(
        args, results_dirs, "prod"
    )

    comp_corr, comp_corr_mean, comp_list = extract_correlations(
        args, results_dirs, "comp"
    )

    print("Production", prod_corr.shape)
    print("Comprehension", comp_corr.shape)
    # assert prod_corr.size > 0 or comp_corr.size > 0, 'Results not found'

    # Find maximum correlations (across all lags)
    # prod_max = np.max(prod_corr, axis=-1)  # .reshape(-1, 1)
    # comp_max = np.max(comp_corr, axis=-1)  # .reshape(-1, 1)

    # save correlations to a file
    # save_max_correlations(args, prod_max, comp_max, prod_list)

    pp = PdfPages(args.output_pdf)

    plot_average_correlations_multiple(pp, prod_corr_mean, comp_corr_mean, args)

    plot_individual_correlation_multiple(pp, prod_corr, comp_corr, prod_list, args)

    pp.close()
