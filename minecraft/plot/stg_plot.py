# 读取 .csv 格式的文件 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd

def plot_stg(src, file):
    df = pd.DataFrame()
    print('start readling csv file: {}'.format(file))
    chunksize = 5e3    
    for chunk in pd.read_csv(file, chunksize=chunksize):
        df = df.append(chunk)

    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 15))

    if 'ele' not in src:
        df.Total_loss.plot(ax=axes[0,0],title="loss")
        df.L2_loss.plot(ax=axes[0,1], title="L2_loss")
        df.G_loss.plot(ax=axes[0,2], title="G_loss")
        df.D_loss.plot(ax=axes[1,0], title="D_loss")
        
    df.TDR_loss.plot(ax=axes[1,1], title="TDR_loss")


    plt.savefig(f'{src}/stg_loss.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True) # discount

    args = parser.parse_args()
    src = os.path.dirname(args.file)
    plot_stg(src, args.file)