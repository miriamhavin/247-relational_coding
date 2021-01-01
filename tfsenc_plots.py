import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# TODO: this file is work in progress

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Helvetica"],
})


def extract_correlations(directory_list, file_str=None):
    all_corrs = []
    for dir in directory_list:
        file_list = sorted(
            glob.glob(os.path.join(dir, '*' + file_str + '.csv')))
        electrode_list = [
            os.path.split(item)[1].split('_')[0] for item in file_list
        ]
        for file in file_list:
            with open(file, 'r') as csv_file:
                ha = list(map(float, csv_file.readline().strip().split(',')))
            all_corrs.append(ha)

    hat = np.stack(all_corrs)
    mean_corr = np.mean(hat, axis=0)
    return hat, mean_corr, electrode_list


def save_max_correlations(args, prod_max, comp_max, prod_list):
    df = pd.DataFrame(prod_max, columns=['production'])
    df['comprehension'] = comp_max
    df['electrode'] = [int(item.strip('elec')) for item in prod_list]
    df = df[['electrode', 'production', 'comprehension']]
    df.to_csv(args.max_corr_csv, index=False)
    return


def extract_electrode_list():
    pass


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-prefix', type=str, default='test')
    parser.add_argument('--datum-emb-fn',
                        type=str,
                        default='podcast-datum-glove-50d.csv')
    parser.add_argument('--gpt2', type=int, default=1)
    parser.add_argument('--bert', type=int, default=None)
    parser.add_argument('--bart', type=int, default=None)
    parser.add_argument('--glove', type=int, default=1)
    parser.add_argument('--electrodes', nargs='*', type=int)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sid', nargs='?', type=int, default=None)
    group.add_argument('--sig-elec-file', nargs='?', type=str, default=None)

    args = parser.parse_args()

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    return args


def initial_setup(args):
    args.output_pdf = str(args.sid) + '_glove_encoding.pdf'
    args.max_corr_csv = str(args.sid) + '_glove50_maxCorrelations.csv"
    return


def plot_average_correlations(pp, prod_corr_mean, comp_corr_mean):
    fig, ax = plt.subplots()
    lags = np.arange(-2000, 2001, 25)

    ax.plot(lags, prod_corr_mean, 'k', label='production')
    ax.plot(lags, comp_corr_mean, 'r', label='comprehension')
    ax.legend()
    ax.set(xlabel='lag (s)',
           ylabel='correlation',
           title='Average Correlation (all electrodes')
    ax.grid()

    pp.savefig(fig)
    plt.close()


def plot_individual_correlations(pp, prod_corr, comp_corr, prod_list):
    for prod_row, comp_row, electrode_id in zip(prod_corr, comp_corr,
                                                prod_list):
        fig, ax = plt.subplots()
        lags = np.arange(-2000, 2001, 25)
        ax.plot(lags, prod_row, 'k', label='production')
        ax.plot(lags, comp_row, 'r', label='comprehension')
        ax.legend(frameon=False)
        ax.set(xlabel='lag (s)', ylabel='correlation', title=electrode_id)
        ax.grid()

        pp.savefig(fig)
        plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    initial_setup()

    directory_string = '20201231-hg-200ms-all--625'

    results_dirs = glob.glob(
        os.path.join(os.getcwd(), 'Results', directory_string))

    prod_corr, prod_corr_mean, prod_list = extract_correlations(
        results_dirs, 'prod')

    comp_corr, comp_corr_mean, _ = extract_correlations(results_dirs, 'comp')

    # Find maximum correlations (across all lags)
    prod_max = np.max(prod_corr, axis=1).reshape(-1, 1)
    comp_max = np.max(comp_corr, axis=1).reshape(-1, 1)

    electrode_list = extract_electrode_list()

    # save correlations to a file
    save_max_correlations(args, prod_max, comp_max, prod_list)

    pp = PdfPages(args.output_pdf)

    plot_average_correlations(pp, prod_corr_mean, comp_corr_mean)
    plot_individual_correlations(pp, prod_corr, comp_corr, prod_list)

    pp.close()
