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
    with open(file, 'rb') as fh:
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
    electrode_list = load_pickle(args.electrode_file)['electrode_name']

    all_corrs = []
    for dir in directory_list:

        dir_corrs = []
        for electrode in electrode_list:
            file = os.path.join(dir, electrode + '_' + file_str + '.csv')
            try:
                with open(file, 'r') as csv_file:
                    ha = list(map(float, csv_file.readline().strip().split(',')))
            except FileNotFoundError:
                print(f'{electrode} not Found')
            dir_corrs.append(ha)

        all_corrs.append(dir_corrs)

    # all_corrs.shape = [len(directory_list), len(electrode_list), num_lags]
    all_corrs = np.stack(all_corrs)

    # all_corrs.shape = [len(directory_list), 1, num_lags]
    mean_corr = np.mean(all_corrs, axis=1)

    return all_corrs, mean_corr, electrode_list


def save_max_correlations(args, prod_max, comp_max, prod_list):
    """[summary]

    Args:
        args ([type]): [description]
        prod_max ([type]): [description]
        comp_max ([type]): [description]
        prod_list ([type]): [description]
    """
    df = pd.DataFrame(prod_max, columns=['production'])
    df['comprehension'] = comp_max
    df['electrode'] = [int(item.strip('elec')) for item in prod_list]
    df = df[['electrode', 'production', 'comprehension']]
    df.to_csv(args.max_corr_csv, index=False)
    return


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--project-id', type=str, default=None)
    parser.add_argument('--output-prefix', type=str, default='test')
    parser.add_argument('--input-directory', nargs='*', type=str, default=None)
    parser.add_argument('--labels', nargs='*', type=str, default=None)
    parser.add_argument('--embedding-type', type=str, default=None)
    parser.add_argument('--electrodes', nargs='*', type=int)
    parser.add_argument('--output-file-name', type=str, default=None)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sid', nargs='?', type=int, default=None)
    group.add_argument('--sig-elec-file', nargs='?', type=str, default=None)

    args = parser.parse_args()

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    return args


def initial_setup(args):
    assert len(args.input_directory) == len(args.labels), "Unequal number of"

    full_input_dir = os.path.join(os.getcwd(), 'Results', args.input_directory)
    args.output_pdf = os.path.join(
        os.getcwd(),
        '_'.join([str(args.sid), args.embedding_type, 'encoding.pdf']))
    args.max_corr_csv = os.path.join(
        full_input_dir,
        '_'.join([str(args.sid), args.embedding_type, 'maxCorrelations.csv']))
    return


def set_plot_styles(args):
    linestyles = ['-', '--', ':']
    color = ['b', 'r']

    # linestyles = linestyles[0:len(args.labels)] * 2
    # color = np.repeat(color[0:len(args.labels)], len(args.labels))
    linestyles = ['-', '-']
    color = ['b', 'r']

    return (color, linestyles)


def set_legend_labels(args):
    legend_labels = []

    for item in ['production', 'comprehension']:
        for label in args.labels:
            legend_labels.append(r'\textit{' + '-'.join([label, item]) + '}')
    return legend_labels


def plot_data(args, data, pp, title=None):
    lags = np.arange(-2000, 2001, 25)

    fig, ax = plt.subplots()
    color, linestyles = set_plot_styles(args)
    ax.set_prop_cycle(color=color, linestyle=linestyles)
    ax.plot(lags, data.T, linewidth=0.75)
    ax.legend(set_legend_labels(args), frameon=False)
    ax.set(xlabel=r'\textit{lag (s)}',
           ylabel=r'\textit{correlation}',
           title=title)
    ax.set_ylim(-0.05, 0.35)
    ax.vlines(0, -0.05, 0.50, linestyles='dashed', linewidth=.75)

    pp.savefig(fig)
    plt.close()


def plot_average_correlations_multiple(pp, prod_corr_mean, comp_corr_mean,
                                       args):

    data = np.vstack([prod_corr_mean, comp_corr_mean])
    plot_data(args, data, pp, r'\textit{Average Correlation (all electrodes)}')


def plot_individual_correlation_multiple(pp, prod_corr, comp_corr, prod_list,
                                         args):
    prod_list = [item.replace('_', '\_') for item in prod_list]
    prod_corr = np.moveaxis(prod_corr, [0, 1, 2], [1, 0, 2])
    comp_corr = np.moveaxis(comp_corr, [0, 1, 2], [1, 0, 2])

    for prod_row, comp_row, electrode_id in zip(prod_corr, comp_corr,
                                                prod_list):
        data = np.vstack([prod_row, comp_row])
        plot_data(args, data, pp, electrode_id)


if __name__ == '__main__':
    # Parse input arguments
    args = parse_arguments()

    args.output_pdf = os.path.join(os.getcwd(), 'results', 'figures',
                                   args.output_file_name + '.pdf')
    args.electrode_file = os.path.join(os.getcwd(), 'data', args.project_id,
                                       str(args.sid), 'pickles',
                                       str(args.sid) + '_electrode_names.pkl')

    assert len(args.input_directory) == len(args.labels), "Unequal number of"

    # Results folders to be plotted
    results_dirs = [
        glob.glob(
            os.path.join(os.getcwd(), 'results', args.project_id, directory,
                         str(args.sid)))[0]
        for directory in args.input_directory
    ]

    prod_corr, prod_corr_mean, prod_list = extract_correlations(
        args, results_dirs, 'prod')

    comp_corr, comp_corr_mean, _ = extract_correlations(
        args, results_dirs, 'comp')

    # Find maximum correlations (across all lags)
    prod_max = np.max(prod_corr, axis=-1)  # .reshape(-1, 1)
    comp_max = np.max(comp_corr, axis=-1)  # .reshape(-1, 1)

    # save correlations to a file
    # save_max_correlations(args, prod_max, comp_max, prod_list)

    pp = PdfPages(args.output_pdf)

    plot_average_correlations_multiple(pp, prod_corr_mean, comp_corr_mean,
                                       args)

    plot_individual_correlation_multiple(pp, prod_corr, comp_corr, prod_list,
                                         args)

    pp.close()
