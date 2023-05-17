import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# 1D Attention for TDR
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


# CNN for Encoder
class AtariCNN(nn.Module):
    '''CNN for Atari (84,84) image'''

    def layer_init(self, layer, std=math.sqrt(2), bias_const=0.0):
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


# Causal Self-Attention
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


# Discriminator
class Discriminator(nn.Module):
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
        self.clip_value = 0.01

    def forward(self, pred, true):
        return self.net(torch.cat((pred, true), -1))

    def update(self, x, pred, true, d_coff):
        D_loss = (self.forward(x, pred) - self.forward(x, true)).mean()
        loss = d_coff * D_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 权值裁剪
        for p in self.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)
        return D_loss
