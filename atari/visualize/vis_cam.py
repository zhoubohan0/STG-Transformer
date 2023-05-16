import argparse
import os
import pickle
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.manifold import TSNE
from glob import glob
from network import STGTransformer
from config import STGConfig

plt.rcParams['text.usetex'] = True


class Cal_CAM(nn.Module):
    def __init__(self, model_file, target_layer="0"):
        super(Cal_CAM, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = STGTransformer(STGConfig()).to(self.device)
        self.model.load(model_file)
        self.encoder = self.model.encoder._modules['feature']
        # 要求梯度的层
        self.feature_layer = target_layer
        # 记录梯度
        self.gradient = []
        # 记录输出的特征图
        self.output = []

    def save_grad(self, grad):
        self.gradient.append(grad)

    def get_grad(self):
        return self.gradient[-1].cpu().data

    def get_feature(self):
        return self.output[-1][0]

    # 计算最后一个卷积层的梯度，输出梯度和最后一个卷积层的特征图
    def getGrad(self, input1, input2):
        input1, input2 = torch.Tensor(input1).to(self.device).requires_grad_(True), torch.Tensor(input2).to(
            self.device).requires_grad_(True)
        x = input1
        for num, (name, module) in enumerate(self.encoder._modules.items()):
            # 待提取特征图的层
            if name == self.feature_layer:
                x = module(x)
                x.register_hook(self.save_grad)
                self.output.append([x])
            # 其他层跳过
            else:
                x = module(x)

        y = self.encoder(input2)
        tdr = self.model.tdr(x[0], y[0])
        self.model.zero_grad()
        # 反向传播获取梯度
        tdr.backward(retain_graph=True)
        # 获取特征图的梯度
        grad_val = self.get_grad()
        feature = self.get_feature()
        return grad_val, feature, input1.grad

    # 计算CAM
    def getCam(self, grad_val, feature):
        # 对特征图的每个通道进行全局池化
        alpha = torch.mean(grad_val, dim=(2, 3)).cpu()
        feature = feature.cpu()
        # 将池化后的结果和相应通道特征图相乘
        cam = torch.zeros(*feature.shape[2:4])
        for idx in range(alpha.shape[1]):
            cam = cam + alpha[0][idx] * feature[0][idx]
        # 进行ReLU操作
        cam = np.maximum(cam.detach().numpy(), 0)
        # plt.imshow(cam)
        # plt.colorbar()
        # plt.savefig("cam.jpg")

        # 将cam区域放大到输入图片大小
        cam_ = cv2.resize(cam, (160, 210))
        # 归一化
        cam_ = cam_ - np.min(cam_)
        cam_ = cam_ / np.max(cam_)
        return cam_

    def saveImg(self, img, filepath):
        fig = plt.figure(dpi=300)
        plt.imshow(img)
        # axes[0].set_title('Saliency Map (SM)')
        plt.gca().set_xticks(np.arange(0, 161, 40))
        plt.gca().set_yticks(np.arange(0, 211, 30))
        if filepath[-1] == 'l': plt.colorbar()
        plt.tight_layout()
        plt.savefig(filepath + '.pdf', bbox_inches='tight', dpi=300)
        # plt.savefig(filepath+'.png', bbox_inches='tight',dpi=300)

    def show_img(self, cam_, img, save):
        params = {
            "font.size": 17,  # 全局字号
            'font.family': 'STIXGeneral',  # 全局字体，微软雅黑(Microsoft YaHei)可显示中文
            "figure.subplot.wspace": 0.2,  # 图-子图-宽度百分比
            "figure.subplot.hspace": 0.4,  # 图-子图-高度百分比
            "axes.spines.right": True,  # 坐标系-右侧线
            "axes.spines.top": True,  # 坐标系-上侧线
            "axes.titlesize": 17,  # 坐标系-标题-字号
            "axes.labelsize": 17,  # 坐标系-标签-字号
            "legend.fontsize": 17,  # 图例-字号
            "xtick.labelsize": 16,  # 刻度-标签-字号
            "ytick.labelsize": 16,  # 刻度-标签-字号
            "xtick.direction": 'out',  # 刻度-方向
            "ytick.direction": 'out',  # 刻度-方向
            # "axes.grid": True,  # 坐标系-网格
            # "grid.linestyle": "--"  # 网格-线型
        }
        plt.rcParams.update(params)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        self.saveImg(cam_, save + '_l')

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_), cv2.COLORMAP_JET)
        if img.max() <= 1: img = (img * 255).astype(np.uint8)
        cam_img = 0.25 * heatmap + 0.9 * img
        img_with_mask = cv2.cvtColor(cam_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        self.saveImg(img_with_mask, save + '_r')

    def forward(self, img1, img2, vis_img, save):
        grad_val, feature, input_grad = self.getGrad(img1, img2)
        cam_ = self.getCam(grad_val, feature)
        return self.show_img(cam_, vis_img, save)


def visualize_cnn(model_file, sampled_states, target, save):
    if not os.path.exists(os.path.split(save)[0]):
        os.mkdir(os.path.split(save)[0])
    cal_cam = Cal_CAM(model_file, target_layer=target)
    state, next_state = sampled_states / 255.
    cal_cam.forward(state.reshape(1, *state.shape), next_state.reshape(1, *state.shape), (state*255).astype(np.uint8), save)


def visualize_embedding_cmp(csvs):
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30.0, n_iter=1000, verbose=0)
    for csv, label in zip(csvs, ['$STG$', '$STG^{-}$']):
        df = pd.read_csv(csv, header=0, index_col=0)
        index = [str(i) for i, each in enumerate(df.iloc[0].mean().values) if abs(each) > 1e-5]
        vis = tsne.fit_transform(df[index])
        plt.scatter(vis[:, 0], vis[:, 1], marker='o', s=8, alpha=0.99, label=label)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Breakout')
    parser.add_argument("--algo", type=str, default='STG')
    args = parser.parse_args()

    sampled_traj = pickle.load(open(random.choice(glob(f'dataset/{args.env}/*.pkl')),'rb'))['states']
    i = random.choice(range(1,sampled_traj.shape[0]-1))
    sampled_states = sampled_traj[i:i+2]
    for layer in ['0', '2', '4']:
        visualize_cnn(
            model_file=glob(f'pretrain-exp/{args.env}_{args.algo}/*.pth')[-1],
            sampled_states=sampled_states,
            target=layer,
            save=f'vis-res/{args.env}_{args.algo}_{layer}'
        )

