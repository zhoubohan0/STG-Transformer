#import imageio
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def plot(file):

    save_dir = os.path.dirname(file) + '/vis'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    #print(x)
    for i in range(0, lines.shape[1]):
        yname = lines[0,i]
        xname = 'steps * 1000'
        plt.xlabel(xname)
        plt.ylabel(yname)
        y = lines[1:, i]
        if y[1] == '':
            for j in range(1, len(y)):
                if y[j] == '':
                    y[j] = y[j-1]
        plt.plot(x, y.astype(np.float))
        plt.savefig(os.path.join(save_dir, yname+'.png'))
        plt.cla()


# smooth the curves
def smooth(arr, weight=0.98): #weight是平滑度，tensorboard 默认0.6
    last = 0
    smoothed = []
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def plot_smooth(file):
    save_dir = os.path.dirname(file) + '/vis_smooth'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(0, lines.shape[1]):
        yname = lines[0,i]
        xname = 'steps * 1000'
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.plot(x, smooth(lines[1:, i].astype(np.float)))
        plt.savefig(os.path.join(save_dir, yname+'.png'))
        plt.cla()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True) # discount

    args = parser.parse_args()


    with open(args.file, 'r') as f:
        lines = f.read().splitlines()

    for i,l in enumerate(lines):
        lines[i] = l.split('\t')

    i_x = lines[0].index('Epoch')
    lines = np.array(lines)

    x = lines[1:, i_x].astype(np.int)
    # x = list(range(1000))

    plot(args.file)
    plot_smooth(args.file)