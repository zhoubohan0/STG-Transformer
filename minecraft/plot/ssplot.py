# 读取 .csv 格式的文件 
import numpy as np
import matplotlib.pyplot as plt
game = 'sunflower'
sstype = 'TDR'
# src = f'/home/like/rl-expert/STGTransformer/ss-exp/{game}_{sstype}_atariCNN_class5_seed666'
src = f'/home/ps/Plan4MC/ablation-model/flower_TDR/'

import pandas as pd
df = pd.DataFrame()

print('start readling csv file')
chunksize = 5e3    #这个数字设置多少有待考察
for chunk in pd.read_csv(f'{src}/trainss.csv', chunksize=chunksize):
    df = df.append(chunk)


titles = df.columns # shape=(5,)
print(titles)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 15))

if 'ele' not in src:
    df.Total_loss.plot(ax=axes[0,0],title="loss")
    df.L2_loss.plot(ax=axes[0,1], title="L2_loss")
    df.G_loss.plot(ax=axes[0,2], title="G_loss")
    df.D_loss.plot(ax=axes[1,0], title="D_loss")
    
df.TDR_loss.plot(ax=axes[1,1], title="TDR_loss")


plt.savefig(f'{src}/train_loss.png')

# x = data[1:,0]
# total_loss = data[1:,1]
# L2_loss = data[1:,2]
# D_loss = data[1:,3]
# TDC_loss = data[1:,4]
# y = [total_loss, L2_loss, D_loss, TDC_loss]

# # 用matplotlib画4个折线图
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.plot(x, y[i])
#     plt.title(title[i])
#     plt.xlabel('epoch')
#     plt.ylabel(title[i])
#     plt.grid(True)

# # 保存
# plt.savefig(f'{src}/train.png')
