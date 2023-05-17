import random
import argparse
import time
from collections import defaultdict
from csv import DictWriter
from glob import glob
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import seaborn as sns
import cv2, os
# import pandas as pd
import numpy as np
# import tensorflow as tf
# from torchvision.models import resnet18
import torch, math, pickle
import torch.nn as nn
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.nn.utils import spectral_norm
import imageio.v2 as imageio


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Logger:
    def __init__(self, save_dir, filename, fieldnames, timing=False):
        self.timing = timing
        self.fieldnames = fieldnames
        if timing:
            self.fieldnames += ["time"]
            self.previous_log_time = time.time()
            self.start_log_time = self.previous_log_time
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        self.csv_file = open(os.path.join(save_dir, filename), 'w', encoding='utf8')
        self.logger = DictWriter(self.csv_file, fieldnames=(self.fieldnames))
        self.logger.writeheader()
        self.csv_file.flush()

    def update(self, fieldvalues):
        epinfo = defaultdict(str)
        if self.timing:
            current_log_time = time.time()
            delta_time = current_log_time - self.previous_log_time
            self.previous_log_time = current_log_time
            epinfo["time"] = str(delta_time)
        if isinstance(fieldvalues, list):
            for filedname, filedvalue in zip(self.fieldnames, fieldvalues):
                epinfo.update({filedname: filedvalue})
        if isinstance(fieldvalues, dict):
            epinfo.update(fieldvalues)
        self.logger.writerow(epinfo)
        self.csv_file.flush()


class Config:
    """ base GPT config, params common to all GPT versions """
    # 模型超参数
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    block_size = 200
    n_layer = 3
    n_head = 4
    n_embd = 512
    max_timestep = 10000
    # 训练超参数
    weight_decay = 0.1
    learning_rate = 1e-4
    betas = (0.9, 0.95)
    '''损失系数'''
    l2_coff = 0.5
    g_coff = 0.05
    d_coff = 0.5  # 不要改，大10倍避免loss爆炸
    tdr_coff = 0.1

    fe = 'MCCNN'  # ResCNN
    n_epoch = 300
    batch_size = 4

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Self_Attn1D(nn.Module):
    def __init__(self, in_dim, k=8):
        super(Self_Attn1D, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1, )
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // k, kernel_size=1, )
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps(B X C X T)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*T)
        """
        B, C = x.size()
        T = 1
        x = x.view(B, C, T)

        # B X C X (N)
        proj_query = self.query_conv(x).view(B, -1, T).permute(0, 2, 1)
        # B X C x (W*H)
        proj_key = self.key_conv(x).view(B, -1, T)
        energy = torch.bmm(proj_query, proj_key)
        # B X (N) X (N)
        attention = self.softmax(energy)
        # B X C X N
        proj_value = self.value_conv(x).view(B, -1, T)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, T)

        out = self.gamma * out + x
        out = out.squeeze(2)

        return out  # , attention


# Temporal Distance Regression
class TDR(nn.Module):
    def __init__(self, in_dim, emb_dim, out_dim):
        super().__init__()
        self.att = Self_Attn1D(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(),
            nn.Linear(emb_dim // 2, emb_dim // 4), nn.ReLU(),
            nn.Linear(emb_dim // 4, out_dim)
        )

    def symlog(self, i, j):
        delta = j - i
        return delta.sign() * torch.log(1 + delta.abs())

    def symexp(self, x):
        return x.sign() * (x.abs().exp() - 1)

    def forward(self, x, y):
        mix = self.att((y - x).view(-1, x.shape[-1])).reshape(x.shape)  # torch.cat((x,y),-1)
        return self.net(mix)


class Conv2dLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
                 normalization='batch', nonlinear='relu'):

        if padding is None:
            padding = (kernel_size - 1) // 2

        bias = (normalization is None or normalization is False)

        module = [nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )]

        if normalization is not None and normalization is not False:
            if normalization == 'batch':
                module.append(nn.BatchNorm2d(num_features=out_channels))
            else:
                raise NotImplementedError('unsupported normalization layer: {0}'.format(normalization))

        if nonlinear is not None and nonlinear is not False:
            if nonlinear == 'relu':
                module.append(nn.ReLU(inplace=True))
            elif nonlinear == 'leakyrelu':
                module.append(nn.LeakyReLU(inplace=True))
            elif nonlinear == 'tanh':
                module.append(nn.Tanh())
            else:
                raise NotImplementedError('unsupported nonlinear activation: {0}'.format(nonlinear))

        super(Conv2dLayer, self).__init__(*module)


class MCCNN(nn.Module):
    '''CNN for Atari (160,256) image'''

    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.conv1 = Conv2dLayer(in_dim, 32, 3, stride=2, normalization='batch', nonlinear='leakyrelu')
        self.conv2 = Conv2dLayer(32, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')
        self.conv3 = Conv2dLayer(64, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')
        self.conv4 = Conv2dLayer(64, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')
        self.conv5 = Conv2dLayer(64, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')
        self.conv6 = Conv2dLayer(64, 128, 3, stride=2, normalization='batch', nonlinear='leakyrelu')
        self.residual1 = ResBlock(32, 32)
        self.residual2 = ResBlock(64, 64)
        self.residual3 = ResBlock(64, 64)
        self.residual4 = ResBlock(64, 64)
        self.residual5 = ResBlock(64, 64)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 4, emb_dim),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):  # (bs,c,84,84)
        feat1 = self.conv1(img)
        feat1 = feat1 + self.residual1(feat1)
        feat2 = self.conv2(feat1)
        feat2 = feat2 + self.residual2(feat2)
        feat3 = self.conv3(feat2)
        feat3 = feat3 + self.residual3(feat3)
        feat4 = self.conv4(feat3)
        feat4 = feat4 + self.residual4(feat4)
        feat5 = self.conv5(feat4)
        feat5 = feat5 + self.residual5(feat5)
        feat6 = self.conv6(feat5)
        x = self.fc(feat6)
        return x


class CNN(nn.Module):
    '''CNN for Atari (84,84) image'''

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.feature = nn.Sequential(
            self.layer_init(nn.Conv2d(in_dim, 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 7 * 7, emb_dim)),
            nn.ReLU(),
        )

    def forward(self, img):  # (bs,c,84,84)
        return self.feature(img)


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResidualCNN(nn.Module):
    '''CNN for Atari (84,84) image'''

    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True)
        )
        self.residual1 = nn.Sequential(
            ResBlock(32, 32),
            nn.Dropout(p=0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True)
        )
        self.residual2 = nn.Sequential(
            ResBlock(64, 64),
            nn.Dropout(p=0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.residual3 = nn.Sequential(
            ResBlock(64, 64),
            nn.Dropout(p=0.1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, emb_dim),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):  # (bs,c,84,84)
        feat1 = self.conv1(img)
        feat1 = feat1 + self.residual1(feat1)
        feat2 = self.conv2(feat1)
        feat2 = feat2 + self.residual2(feat2)
        feat3 = self.conv3(feat2)
        feat3 = feat3 + self.residual3(feat3)
        x = self.fc(feat3)
        return x


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1)).view(1, 1,
                                                                                                               config.block_size + 1,
                                                                                                               config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CSABlock(nn.Module):  # CausalSelfAttention Block
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Discriminator(nn.Module):  # GAN的鉴别器可以加频谱归一化函数nn.utils.spectral_norm
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd * 2, 512),
            nn.GELU(),
            spectral_norm(nn.Linear(512, 256)),
            nn.GELU(),
            spectral_norm(nn.Linear(256, 128)),
            nn.GELU(),
            spectral_norm(nn.Linear(128, 64)),
            nn.GELU(),
            spectral_norm(nn.Linear(64, 32)),
            nn.GELU(),
            spectral_norm(nn.Linear(32, 1)),
        )
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=config.learning_rate)  # , betas=config.betas
        self.clip_value = 0.03

    def forward(self, pred, true):
        return self.net(torch.cat((pred, true), -1))

    def update(self, x, pred, true):
        D_loss = (self.forward(x, pred) - self.forward(x, true)).mean()
        loss = config.d_coff * D_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 权值裁剪
        for p in self.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)
        return D_loss


class SSTransformer(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def load(self, stgmodel_path):
        if stgmodel_path.endswith('.pth'):
            self.load_state_dict(torch.load(stgmodel_path))
            print('load model from {}'.format(stgmodel_path))
        if stgmodel_path.endswith('.pt'):
            self = torch.load(stgmodel_path)
            print('load model from {}'.format(stgmodel_path))

    def __init__(self, config):
        super().__init__()
        self.initialized = False
        self.config = config
        # input embedding
        self.encoder = MCCNN(3, config.n_embd)
        # '''nn.Parameter()进入模型可更新参数中'''
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        # self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))
        # self.drop = nn.Dropout(config.embd_pdrop)
        # # transformer block
        # self.blocks = nn.Sequential(*[CSABlock(config) for _ in range(config.n_layer)])
        # # decoder head
        # self.ln_f = nn.LayerNorm(config.n_embd)
        # final projection(regression)
        # 二分类器
        # self.discriminator = Discriminator(config)
        self.out = nn.Linear(config.n_embd, config.n_embd)
        # 时序距离预测器
        self.tdr = TDR(config.n_embd, config.n_embd, 1)
        # 对所有网络初始化参数
        self.apply(self._init_weights)
        # 优化器
        self.optimizer = self.configure_optimizers()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')
        # no_decay.add('global_pos_emb')

        # 参数必将仅能处于decay/no_decay的集合中
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        # assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=self.config.betas)
        return optimizer

    def update(self, data):
        '''
        data: (batch_size, block_size, (channel, width, height))
        timesteps: (batch_size, 1, 1)
        return: (batch_size, block_size, emb_dim)
        '''
        batch_size, block_size, c, w, h = data.shape
        # 输入特征
        input_embedding = self.encoder(data.view(-1, c, w, h)).view(batch_size, block_size,self.config.n_embd)  # 先展平再reshape回
        idx1 = np.arange(block_size)
        idx2 = np.random.permutation(block_size)
        idx1, idx2 = torch.LongTensor(idx1).to(device), torch.LongTensor(idx2).to(device)
        time_dis = self.tdr.symlog(idx1, idx2).repeat(batch_size)
        time_pred = self.tdr(input_embedding[:, idx1, :], input_embedding[:, idx2, :])
        tdr_loss = F.mse_loss(time_pred.view(time_dis.shape), time_dis)
        # 更新总损失
        loss = self.config.tdr_coff * tdr_loss  # L1 + L2 + Adversary
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return tdr_loss.item()

    def trainss(self, src_root_list, threshold):
        if not self.initialized:
            for src_root in src_root_list:
                gif2pkl(src_root, threshold)
            self.initialized = True

        # 训练
        save_dir = f'elemodel-exp/{task}_TDR/'
        exp_logger = Logger(save_dir, f'trainss.csv',fieldnames=['update', 'TDR_loss'])
        self.train()  # 测试时model.eval()关闭dropout和batchnorm
        num = 0
        for i_epoch in range(self.config.n_epoch):
            for each in glob(os.path.join(src_root,'*success1.pkl')):
                pkl_name = os.path.splitext(os.path.basename(each))[0]
                ret_value = int(pkl_name.split('_')[1][3:])
                if threshold is None or ret_value > threshold:
                    states = pickle.load(open(each,'rb'))
                    for i in range(4):
                        if states.shape[0]<=config.block_size:
                            batch = torch.Tensor(states/ 255.).unsqueeze(0)
                        else:
                            b = states.shape[0] - self.config.block_size
                            step = torch.randint(b, (min(config.batch_size,b), 1, 1),dtype=torch.long)  # (config.batch_size,1,1)
                            batch = torch.Tensor(states[np.array([list(range(id, id + self.config.block_size + 1)) for id in step])]) / 255.0  # (config.batch_size, config.block_size+1, 4, 84, 84)
                        tdr_loss = self.update(batch.to(device))
                        exp_logger.update(fieldvalues=[num, tdr_loss])
                        num += 1
                        if num % 5000 == 0: torch.save(self.state_dict(), f'{save_dir}/{num}.pth')
                        if num % 100 == 0: print(f'update:{num:05d} | TDR_loss:{tdr_loss:.6f}')
        # 保存
        torch.save(self.state_dict(), f'{save_dir}/{num}.pth')
        print()

    def cal_intrinsic_s2e(self, states_ls,device=torch.device('cuda')):
        '''
        states:(block_size+1,4,84,84)
        step:标量
        '''
        self.eval()
        with torch.no_grad():
            progress_span = torch.Tensor(0).to(device)
            for i in range(0,states_ls.shape[1],self.config.block_size):
                states = states_ls[:,i:i+self.config.block_size+1,...].to(device)
                embeddings = self.encoder(states[0]).unsqueeze(0)  # (1, block_size+1, n_embd)
                if embeddings.shape[1] <= 1: continue
                cur_embedding, next_embedding = embeddings[:, :-1, :], embeddings[:, 1:,:]  # (1, block_size+1, n_embd)
                # limit = torch.log(1 + torch.Tensor([horizen]).to(device))
                ps = self.tdr.symexp(self.tdr(cur_embedding, next_embedding).view(-1))#.clamp(-limit,limit)
                progress_span = torch.cat((progress_span, ps), dim=0)
            intrinsic = progress_span.detach().cpu().numpy()
            return intrinsic,intrinsic,intrinsic


def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def gif2pkl(dataset_path, threshold=None):
    src = glob(os.path.join(dataset_path, '*success1.gif'))
    count = 0

    for i, gif_path in enumerate(src, 1):
        gif_name = os.path.splitext(os.path.basename(gif_path))[0]
        ret_value = int(gif_name.split('_')[1][3:])

        if threshold is None or ret_value > threshold:
            count += 1
            frames = []
            gif = cv2.VideoCapture(gif_path)
            while True:
                ret, frame = gif.read()
                if not ret: break
                frames.append(frame)
            frames = np.array(frames, np.uint8).transpose(0, 3, 1, 2)
            pkl_path = os.path.join(dataset_path, f'{gif_name}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(frames, f)

    if threshold is not None:
        print(f'Total number of files with ret > {threshold}: {count}')
    else:
        print(f'Total number of files processed: {count}')

if __name__ == '__main__':
    task = 'ele-flower'
    src_root = ["/home/ps/Plan4MC/checkpoint/Expert_ppo_double_plant_re-1.0_lbmda0.8_seed42",
                ]
    threshold = 100 # default: None (process all the success gif files without filtering the return value)


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--block_size', type=int, default=200)
    args = parser.parse_args()
    seed = args.seed
    setseed(seed)
    config = Config(block_size=args.block_size)

    # 模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ssmodel = SSTransformer(config).to(device)
    
    # 训练以及测试
    ssmodel.trainss(src_root, threshold)