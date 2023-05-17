import argparse
import os
import pickle
import random
import sys
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch import nn

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
plt.rcParams['text.usetex'] = True


class Cal_CAM(nn.Module):
    def __init__(self, model, target_layer="0"):
        super(Cal_CAM, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.encoder = self.model.encoder._modules['feature']
        # target layer of network
        self.feature_layer = target_layer
        # gradient recorder
        self.gradient = []
        # feature-map recorder
        self.output = []

    def save_grad(self, grad):
        self.gradient.append(grad)

    def get_grad(self):
        return self.gradient[-1].cpu().data

    def get_feature(self):
        return self.output[-1][0]

    def getGrad(self, input1, input2):
        # output gradient and feature map of the target layer
        input1, input2 = torch.Tensor(input1).to(self.device).requires_grad_(True), torch.Tensor(input2).to(
            self.device).requires_grad_(True)
        x = input1
        for num, (name, module) in enumerate(self.encoder._modules.items()):
            if name == self.feature_layer:
                x = module(x)
                x.register_hook(self.save_grad)
                self.output.append([x])
            else:
                x = module(x)

        y = self.encoder(input2)
        tdr = self.model.tdr(x[0], y[0])
        self.model.zero_grad()
        # backpropagation to obtain gradient
        tdr.backward(retain_graph=True)
        # get gradient
        grad_val = self.get_grad()
        feature = self.get_feature()
        return grad_val, feature, input1.grad

    # calculate CAM
    def getCam(self, grad_val, feature):
        # global average pooling of each channel in the feature map -> weights
        alpha = torch.mean(grad_val, dim=(2, 3)).cpu()
        feature = feature.cpu()
        # weighted feature map
        cam = torch.zeros(*feature.shape[2:4])
        for idx in range(alpha.shape[1]):
            cam = cam + alpha[0][idx] * feature[0][idx]
        # ReLU
        cam = np.maximum(cam.detach().numpy(), 0)
        # plt.imshow(cam)
        # plt.colorbar()
        # plt.savefig("cam.jpg")

        # resize and normalize for visualization
        cam_ = cv2.resize(cam, (84, 84))
        cam_ = cam_ - np.min(cam_)
        cam_ = cam_ / np.max(cam_)
        return cam_

    def saveImg(self, img, filepath):
        fig = plt.figure(dpi=300)
        plt.imshow(img)
        # axes[0].set_title('Saliency Map (SM)')
        plt.gca().set_xticks(np.arange(0, 85, 15))
        plt.gca().set_yticks(np.arange(0, 85, 21))
        if filepath[-1] == 'l': plt.colorbar()
        plt.tight_layout()
        plt.savefig(filepath + '.pdf', bbox_inches='tight', dpi=300)
        # plt.savefig(filepath+'.png', bbox_inches='tight',dpi=300)

    def show_img(self, cam_, img, save):
        params = {
            "font.size": 17,
            'font.family': 'STIXGeneral',
            "figure.subplot.wspace": 0.2,
            "figure.subplot.hspace": 0.4,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.titlesize": 17,
            "axes.labelsize": 17,
            "legend.fontsize": 17,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.direction": 'out',
            "ytick.direction": 'out',
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


def visualize_cnn(model, sampled_states, target, save):
    if not os.path.exists(os.path.split(save)[0]):
        os.mkdir(os.path.split(save)[0])
    cal_cam = Cal_CAM(model, target_layer=target)
    state, next_state = sampled_states / 255.
    cal_cam.forward(state.reshape(1, *state.shape), next_state.reshape(1, *state.shape),
                    (state * 255).astype(np.uint8).transpose(1, 2, 0)[..., :-1],save)


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
    parser.add_argument("--model", type=str, default='')
    args = parser.parse_args()
    if not args.model:
        args.model = f'./pretrain-exp/{args.env}_{args.algo}/{args.env}_{args.algo}.pt'
    sampled_traj = pickle.load(open(random.choice(glob(f'dataset/{args.env}/*.pkl')), 'rb'))['states']
    i = random.choice(range(1, sampled_traj.shape[0] - 1))
    sampled_states = sampled_traj[i:i + 2]
    for layer in ['0', '2', '4']:
        visualize_cnn(
            model=torch.load(args.model),
            sampled_states=sampled_states,
            target=layer,
            save=f'vis-res/{args.env}_{args.algo}_{layer}'
        )
