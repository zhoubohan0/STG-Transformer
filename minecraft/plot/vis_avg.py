import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

# theme
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')

# title and name
title = 'Harvest Wool'
save_name = 'wool'

save_path = 'images'

xname = 'Epochs'
yname = 'Average Success Rate'

# num of data
num_horizon = 1000
x = list(range(num_horizon))

# Motion clip
f1 = '/home/ps/Plan4MC/data/ppo_wool_-10.0ds_lbmda0.8_seed0/ppo_wool_-10.0ds_lbmda0.8_seed0_s7/ppo_wool_-10.0ds_lbmda0.8_seed0'
f2 = '/home/ps/Plan4MC/data/ppo_wool_-10.0ds_lbmda0.9_seed128/ppo_wool_-10.0ds_lbmda0.9_seed128_s7/ppo_wool_-10.0ds_lbmda0.9_seed128'
f3 = 'motionclip/flower/3.txt'
f4 = 'motionclip/flower/4.txt'
f5 = 'motionclip/flower/5.txt'


group = [[f1, f2]]

label = ['STG', # ours v2
        'MineCLIP + Env', # baseline
        ]

# smooth the curves
def smooth(arr, weight=0.98): #weight是平滑度，tensorboard 默认0.6
    last = 0 # last = arr[0]
    smoothed = []
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# initialize data array
alldata = []
for i in range(len(group)):
    alldata.append([])
    for j in range(len(group[i])):
        with open(group[i][j], 'r') as f:
            lines = f.read().splitlines()

        for k,l in enumerate(lines):
            lines[k] = l.split('\t')

        i_y = lines[0].index('AverageEpSuccess')
        lines = np.array(lines)
        y = np.array(lines[1:num_horizon+1, i_y])
        
        y = y.astype(np.float)
        alldata[i].append(smooth(y))
    alldata[i] = np.array(alldata[i])


for i in range(len(alldata)):
    color=palette(i+2)#算法1颜色
    avg=np.mean(alldata[i],axis=0)
    std=np.std(alldata[i],axis=0)
    
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std/2))) #上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std/2))) #下方差
    plt.plot(x, avg, color=color,label=label[i],linewidth=3.0)
    plt.fill_between(x, r1, r2, color=color, alpha=0.2)


if not os.path.exists(save_path):
    os.mkdir(save_path)

plt.legend(loc='lower right')
plt.xlabel(xname)
plt.ylabel(yname)


plt.savefig(os.path.join(save_path, save_name+'.pdf'))
plt.savefig(os.path.join(save_path, save_name+'.png'))