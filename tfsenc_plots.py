import glob
import os

import matplotlib.pyplot as plt
import numpy as np

# TODO: this file is work in progress


def extract_correlations(directory_list, file_str=None):
    all_corrs = []
    for dir in directory_list:
        file_list = glob.glob(os.path.join(dir, '*' + file_str + '.csv'))
        for file in file_list:
            with open(file, 'r') as csv_file:
                ha = list(map(float, csv_file.readline().strip().split(',')))
            all_corrs.append(ha)

    hat = np.stack(all_corrs)
    mean_corr = np.mean(hat, axis=0)
    return mean_corr


python_dir_list = glob.glob(os.path.join(os.getcwd(), 'Results', '2020*'))
prod_mean_corr = extract_correlations(python_dir_list, 'prod')
comp_mean_corr = extract_correlations(python_dir_list, 'comp')

fig, ax = plt.subplots()
lags = np.arange(-2000, 2001, 100)
ax.plot(lags, prod_mean_corr, 'k', label='python')
ax.plot(lags, comp_mean_corr, 'r', label='matlab')
ax.legend()
ax.set(xlabel='lag (s)', ylabel='correlation', title='Here it is')
ax.grid()

fig.savefig("Production_vs_Comprehension.png")
plt.show()
