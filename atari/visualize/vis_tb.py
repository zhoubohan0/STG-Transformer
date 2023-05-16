import argparse
import pathlib
import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from glob import glob
from tensorboard.backend.event_processing import event_accumulator
from scipy.interpolate import interp1d

params = {
    "font.size": 31,
    'font.family': 'STIXGeneral',
    "figure.subplot.wspace": 0.2,
    "figure.subplot.hspace": 0.4,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.titlesize": 31,
    "axes.labelsize": 31,
    "legend.fontsize": 25,
    'legend.handlelength':3,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "axes.grid": True,
    "grid.linestyle": "--"
}
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
red,blue,green,purple,orange,yellow,brown,pink,gray=[palette(i) for i in range(9)]
shallow_blue = np.array([130,200,230])/255.
deep_blue = np.array([20,180,200])/255.
shallow_red = np.array([0.9,0.11,0.11,0.9])
black = np.array([0,0,0,0.9])
plt.rcParams.update(params)
plt.rcParams['text.usetex'] = True

def tensorBoard2dataFrame(tbdir):
    ea = event_accumulator.EventAccumulator(tbdir)
    ea.Reload()
    #pd.DataFrame(ea.scalars.Items(ea.scalars.Keys()[0])).drop(columns=['wall_time']).plot(x='step',y='value'),plt.grid(),plt.show()
    print(ea.scalars.Keys())
    which = 'charts/episodic_return'
    raw_data = ea.scalars.Items(which)
    which = 'wall_time'
    data = pd.DataFrame(raw_data)
    data = data.drop(columns=[which])  # TODO
    print(f'Tensorboard data has been converted to dataFrame.')
    return data


def tb2csv(tbfiles, outfiles):
    for num, (each_tb, each_csv) in enumerate(zip(tbfiles, outfiles), 1):
        print(num)
        if os.path.exists(each_csv): continue
        ea = event_accumulator.EventAccumulator(each_tb)
        ea.Reload()
        which = 'charts/episodic_return'
        raw_data = ea.scalars.Items(which)
        which = 'wall_time'
        data = pd.DataFrame(raw_data)
        data = data.drop(columns=[which])
        if not os.path.exists(os.path.split(each_csv)[0]):
            os.makedirs(os.path.split(each_csv)[0])
        data.to_csv(each_csv)
        print(f'{num} tensorboard data has been converted to csv file.')


def smooth(arr, weight=0.9):
    if not isinstance(arr, list):
        arr = list(arr)
    last = arr[0]
    smoothed = []
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)



def get_column(f, length, xkey=None, ykey=None, index_col=0, issmooth=True):  # 返回dataframe
    if not isinstance(f, str):  # f is DataFrame
        data = f
    else:  # f is filename
        data = tensorBoard2dataFrame(f) if os.path.split(f)[1][:6] == 'events' else \
            pd.read_csv(f, header=0, index_col=index_col)
    if len(data.columns) == 2:
        xlabel, ylabel = data.columns
    elif xkey is not None and ykey is not None:
        xlabel, ylabel = xkey, ykey
    else:
        xlabel, ylabel = data.columns[0], data.columns[1]
    if length is not None:
        data = data[:length]
    target_column = data[ylabel]
    if issmooth:
        target_column = smooth(target_column)
    column = pd.DataFrame(np.c_[data[xlabel].values.astype(int), target_column], columns=[xlabel, ylabel])
    return column


def single_draw(f, length=None, index_col=None, axislabel=('Transition', 'Episodic Return'), save=True):
    plt.figure(figsize=(8, 6), dpi=300)
    target_column = get_column(f, length, index_col=index_col)
    xlabel, ylabel = target_column.columns
    target_column.plot(x=xlabel, y=ylabel)
    plt.xlabel(xlabel if not axislabel else axislabel[0]), plt.ylabel(ylabel if not axislabel else axislabel[1])
    plt.grid()
    if save: plt.savefig(os.path.join(os.path.split(f)[0], ylabel + '.jpg'), bbox_inches='tight', dpi=300)
    plt.show()


def muti_draw(files, length=None, xkey=None, ykey=None, labels=None, axislabel=('Transition', 'Episodic Return'),
              index_col=0, save=False):
    plt.figure(figsize=(8, 6), dpi=300)
    for i, f in enumerate(files):
        target_column = get_column(f, length, xkey, ykey, index_col=index_col)
        xlabel, ylabel = target_column.columns
        plt.plot(target_column[xlabel], target_column[ylabel], label=i if labels is None else labels[i])
    plt.xlabel(xlabel if not axislabel else axislabel[0]), plt.ylabel(ylabel if not axislabel else axislabel[1])
    plt.legend(loc='best', ncol=1)
    plt.grid()
    if save: plt.savefig(os.path.join(os.path.split(files[0])[0], ylabel + '_compare.jpg'), bbox_inches='tight', dpi=300)
    plt.show()


def all_draw_head(inputfiles, length=None, label=None,color=None,linestyle='-',way=1):
    '''sns.lineplot：https://zhuanlan.zhihu.com/p/111775661'''
    xlabel, ylabel = get_column(inputfiles[0], length).columns
    if way == 0:
        data = get_column(inputfiles[0], length)
        for i in range(1, len(inputfiles)):
            data = get_column(inputfiles[i], length).append(data)
        sns.lineplot(data=data, x=xlabel, y=ylabel, label=label, alpha=0.7)
    if way == 1:  
        ys = [get_column(each, length)[ylabel].values.clip(max=55000) for each in inputfiles]
        xs = [get_column(each, length)[xlabel].values for each in inputfiles]
        x = np.linspace(np.concatenate(xs).min(), np.concatenate(xs).max(), num=100)
        y = [np.interp(x, xs[i], ys[i]) for i in range(len(inputfiles))]
        avg = np.mean(y, axis=0)
        std = np.std(y, axis=0)
        if 'Breakout' not in inputfiles[0]:
            std = std / 2
        r1 = list(map(lambda x: x[0] - x[1], zip(avg, std/2)))
        r2 = list(map(lambda x: x[0] + x[1], zip(avg, std/2)))
        plt.plot(x, avg, color=color, label=label, linewidth=3.0, ls=linestyle)
        plt.fill_between(x, r1, r2, color=color, alpha=0.2)


def remove_outlier(df, adjust):
    xlabel, ylabel = df.columns
    mean = df[ylabel].mean()
    std = df[ylabel].std()
    threshold = mean + adjust * std
    is_outlier = df[ylabel] > threshold
    new_df = df[~is_outlier]
    return new_df


def all_draw(inputseries, length=None, axislabel=('Transition', 'Smoothed Episodic Return'), labels=None,colors=None,save=None):
    plt.figure(figsize=(8, 6), dpi=300)
    if labels is None: labels = [None]*len(inputseries)
    if colors is None:
        colors=[orange] if len(inputseries)==1 else [palette(i) for i in range(len(inputseries))]
    else:
        assert len(colors) == len(inputseries)
    for each_sery, each_label,each_color in zip(inputseries, labels,colors):
        all_draw_head(each_sery, length, each_label,each_color)
    plt.tick_params(axis='both', which='both', width=2)
    plt.legend(loc='best', ncol=1,prop={'weight': 'bold'},borderaxespad=0.5,framealpha=0.8, fancybox=False)
    # plt.grid(color='grey', linestyle='--', linewidth=1)
    if axislabel: plt.xlabel(axislabel[0])#, plt.ylabel(axislabel[1])
    plt.xticks(weight='bold'),plt.yticks(weight='bold')
    ax = plt.gca()
    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gcf().subplots_adjust(bottom=0.2)
    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)
        plt.savefig(os.path.join(save, f'{game}.pdf'), bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--STG_data", nargs='*', default=[])
    # args = parser.parse_args()
    labels = ['STG','ELE','GAIfO']
    for game in ['Breakout','Freeway','Qbert','SpaceInvaders']:
        src_list = glob(f'{game}/*/*')
        tgt_list = [os.path.join(s,s[s.find('seed'):]+'.csv') for s in src_list]
        tb2csv(src_list, tgt_list)
        all_draw(
            inputseries=[
                *[glob(f'ppo-exp/{l}*{game}*/events.out.tfevents.*') for l in labels]
            ],
            labels=labels,
            save='vis-res',
        )

