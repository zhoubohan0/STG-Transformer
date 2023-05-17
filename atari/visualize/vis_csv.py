import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def draw_excelcolumns(csv_root, is_index_col=False, columns='all', n_data=-1):
    '''column 0: index   column 1~(n-1): values'''
    search = glob(os.path.join(csv_root, '*.csv'))
    if len(search) == 0:
        return
    f = search[0]
    filext = os.path.splitext(f)[1]
    index_col = 0 if is_index_col else None
    if filext in ['.csv']:
        df = pd.read_csv(f, header=0, index_col=index_col)
    if filext in ['.xlsx']:
        df = pd.read_excel(f, header=0, index_col=index_col)
    if n_data == -1: n_data = df.shape[0]
    df = df[:n_data]
    xlabel = df.columns[0]
    ylabels = df.columns[1:] if columns == 'all' else columns
    row, col = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 2), (1, 5), (2, 3), ][len(ylabels)]
    fig, axs = plt.subplots(row, col, figsize=(col * 5, row * 5))  # , sharex='all'
    for i, ylabel in enumerate(ylabels):
        idx = i if row <= 1 else divmod(i, col)
        cur_ax = axs[idx] if row * col != 1 else axs
        sns.lineplot(x=xlabel, y=ylabel, data=df, ax=cur_ax)
        cur_ax.grid()
        # axs[idx].legend(loc='best', ncol=1)
        cur_ax.set_xlabel(xlabel)
        cur_ax.set_ylabel(ylabel)
        plt.tight_layout()
    plt.savefig(os.path.join(csv_root, f'{os.path.split(csv_root)[1]}.png'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    draw_excelcolumns(args.root, is_index_col=False, columns='all')
