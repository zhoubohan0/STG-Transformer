import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

from util import Logger
from .networks import AtariCNN, TDR, CSABlock, Discriminator


class STGTransformer(nn.Module):
    def load(self, stgmodel_path):
        if stgmodel_path.endswith('.pth'):
            ckpt = self.state_dict()
            load_ckpt = torch.load(stgmodel_path)
            ckpt.update(load_ckpt)
            self.load_state_dict(ckpt)
            print('load model from {}'.format(stgmodel_path))
        if stgmodel_path.endswith('.pt'):
            self = torch.load(stgmodel_path)
            print('load model from {}'.format(stgmodel_path))

    def __init__(self, config):
        super().__init__()
        self.config = config
        # input embedding
        self.encoder = AtariCNN(4, config.n_embd)
        # positional encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer block
        self.blocks = nn.Sequential(*[CSABlock(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        # decoder
        self.out = nn.Linear(config.n_embd, config.n_embd)
        # discriminator
        self.discriminator = Discriminator(config)
        # temporal distance regressor
        self.tdr = TDR(config.n_embd, config.n_embd, 1)
        # optimizer
        self.optimizer = self.configure_optimizers()
        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
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
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

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

    def update(self, data, timesteps):
        '''
        data: (batch_size, block_size, (channel, width, height))
        timesteps: (batch_size, 1, 1)
        return: (batch_size, block_size, emb_dim)
        '''
        batch_size, block_size, c, w, h = data.shape
        embeddings = self.encoder(data.view(-1, c, w, h)).view(batch_size, block_size, self.config.n_embd)
        input_embedding, target_embedding = embeddings[:, :-1, :], embeddings[:, 1:, :]
        # positional encoding
        position_encodings = self.pos_emb[:, :input_embedding.shape[1], :]
        if timesteps is not None:
            timesteps = timesteps.clip(max=self.config.max_timestep - 1)
            position_encodings = position_encodings + self.global_pos_emb.repeat_interleave(input_embedding.shape[0],
                                                                                            dim=0).gather(1,
                                                                                                          timesteps.repeat_interleave(
                                                                                                              self.config.n_embd,
                                                                                                              dim=-1))
        # CSA
        x = self.ln_f(self.blocks(self.drop(input_embedding + position_encodings)))  # (bs,block_size,n_emb)
        # decode
        delta_embedding = self.out(x)
        expert_embedding = delta_embedding + input_embedding
        # update discriminator
        D_loss = self.discriminator.update(input_embedding.detach(), expert_embedding.detach(),
                                           target_embedding.detach(), self.config.d_coff)
        # calculate adversarial loss and MSE loss
        L2_loss = F.mse_loss(delta_embedding, target_embedding - input_embedding)
        G_loss = -self.discriminator(input_embedding, expert_embedding).mean()
        # calculate temporal distance
        idx1 = np.arange(self.config.block_size)
        idx2 = np.random.permutation(self.config.block_size)
        timesteps = timesteps.repeat_interleave(self.config.block_size).reshape(-1, self.config.block_size)
        idx1, idx2 = torch.LongTensor(idx1).to(self.config.device), torch.LongTensor(idx2).to(self.config.device)
        time_dis = self.tdr.symlog(idx1 + timesteps, idx2 + timesteps)
        time_pred = self.tdr(target_embedding[:, idx1, :], target_embedding[:, idx2, :])
        tdr_loss = F.mse_loss(time_pred.view(time_dis.shape), time_dis)
        # total loss
        loss = self.config.l2_coff * L2_loss + self.config.g_coff * G_loss + self.config.tdr_coff * tdr_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return L2_loss.item(), G_loss.item(), D_loss.item(), tdr_loss.item(), loss.item()

    def trainSTG(self, src, game):
        root = './pretrain-exp'
        if not os.path.exists(root):
            os.mkdir(root)
        mode = 'STG' if self.config.tdr_coff != 0 else 'STG-'
        save_dir = os.path.join(root, f"{game}_{mode}")
        exp_logger = Logger(save_dir, f'trainstg.csv',
                            fieldnames=['update', 'Total_loss', 'L2_loss', 'G_loss', 'D_loss', 'TDR_loss'])
        self.train()
        cnt = 0
        for i_epoch in range(self.config.n_epoch):
            random.shuffle(src)
            src_len = len(src)
            for i_src in range(src_len):
                states = pickle.load(open(os.path.join('.', src[i_src]), 'rb'))['states']
                minibatch = states.shape[0] // self.config.block_size
                for i in range(minibatch):
                    step = torch.randint(states.shape[0] - self.config.block_size, (self.config.batch_size, 1, 1),
                                         dtype=torch.long)  # (config.batch_size,1,1)
                    batch = torch.Tensor(states[np.array([list(range(id, id + self.config.block_size + 1)) for id in
                                                          step])]) / 255.0  # (config.batch_size, config.block_size+1, 4, 84, 84)
                    L2_loss, G_loss, D_loss, tdr_loss, tot_loss = self.update(batch.to(self.config.device),
                                                                              step.to(self.config.device))
                    exp_logger.update(fieldvalues=[cnt, tot_loss, L2_loss, G_loss, D_loss, tdr_loss])
                    cnt += 1
                    if cnt % 5000 == 0:
                        torch.save(self.state_dict(), f'{save_dir}/{game}_{mode}_{cnt}.pth')
                    if cnt % 100 == 0:
                        print(
                            f'update:{cnt:05d} | loss:{tot_loss:.6f} | L2_loss:{L2_loss:.6f} | G_loss:{G_loss:.6f} | D_loss:{D_loss:.6f} | TDR_loss:{tdr_loss:.6f}')
        torch.save(self.state_dict(), f'{save_dir}/{game}_{mode}_{cnt}.pth')
        torch.save(self, f'{save_dir}/{game}_{mode}.pt')

    def compare_embedding(self, data_path, ckpt_paths, labels=['$STG$', '$STG^{-}$']):
        colors = [np.array([60, 220, 215]) / 255., np.array([205, 255, 120]) / 255.]
        tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30.0, n_iter=1000, verbose=0)

        self.eval()
        selected_traj = random.choice(glob(data_path))
        states = pickle.load(open(selected_traj, 'rb'))['states']
        step = torch.arange(0, states.shape[0] - self.config.block_size - 1, self.config.block_size).long().view(-1, 1,
                                                                                                                 1)
        batch = torch.Tensor(
            states[np.array([list(range(id, id + self.config.block_size + 1)) for id in step])]) / 255.0
        step, batch = step.to(self.config.device), batch.to(self.config.device)
        batch_size, block_size, c, w, h = batch.shape
        for ckpt, label, color in zip(ckpt_paths, labels, colors):
            self.load(ckpt)
            input_embedding = self.encoder(batch.view(-1, c, w, h)).view(batch_size, block_size, self.config.n_embd)
            vis_in = tsne.fit_transform(input_embedding.reshape(-1, self.config.n_embd).detach().cpu().numpy())
            plt.scatter(vis_in[:, 0], vis_in[:, 1], c=color, marker='o', s=8, alpha=0.99, label=label)  #
        plt.legend(loc='best')
        plt.savefig(os.path.splitext(ckpt_paths[0])[0] + '.pdf')
        # plt.savefig(os.path.splitext(ckpt_paths[0])[0]+'.png')
        print()

    def cal_intrinsic_s2e(self, states_ls, steps_ls):
        '''
        states:(block_size+1,4,84,84)
        step:(1,)
        '''
        device = states_ls[0][0].device
        self.eval()
        with torch.no_grad():
            discrimination_score = torch.Tensor(0).to(device)
            progress_span = torch.Tensor(0).to(device)
            for i in range(len(steps_ls)):
                step = steps_ls[i]
                states = torch.stack(states_ls[i])
                if len(states) <= 1:
                    continue
                with torch.no_grad():
                    embeddings = self.encoder(states)
                    embeddings = embeddings.unsqueeze(0)  # (1, block_size+1, n_embd)
                    cur_embedding, next_embedding = embeddings[:, :-1, :], embeddings[:, 1:,
                                                                           :]  # (1, block_size+1, n_embd)
                    # Positional Embedding
                    step = step.clip(max=self.config.max_timestep - 1)
                    position_encodings = self.pos_emb[:, :cur_embedding.shape[1], :]
                    position_encodings = position_encodings + self.global_pos_emb.repeat_interleave(
                        cur_embedding.shape[0], dim=0) \
                        .gather(1, step.repeat_interleave(self.config.n_embd, dim=-1))
                    delta_embedding = self.out(self.ln_f(
                        self.blocks(self.drop(cur_embedding + position_encodings))))  # (1, block_size, n_embd)
                    pred_embedding = cur_embedding + delta_embedding
                    # Intrinsic Reward
                    '''
                    pred_emb - next_emb = [Δ1, Δ2, ..., Δn], intrinsic:
                    1. Δ1^2 + Δ2^2 + ... + Δn^2                     L2^2
                    2. √(Δ1^2 + Δ2^2 + ... + Δn^2)                  L2
                    3. |Δ1| + |Δ2| + ... + |Δn|                     L1
                    4. D(cur_emb,pred_emb) - D(cur_emb,next_emb)    Adversary
                    '''
                    # pe = F.mse_loss(pred_embedding, next_embedding, reduction='none').sum(-1).view(-1)
                    ds = (self.discriminator(cur_embedding, pred_embedding) - self.discriminator(cur_embedding,
                                                                                                 next_embedding)).view(
                        -1)
                    ps = self.tdr.symexp(self.tdr(cur_embedding, pred_embedding).view(-1))
                    discrimination_score = torch.cat((discrimination_score, ds), dim=0)
                    progress_span = torch.cat((progress_span, ps), dim=0)
            return {'ADV': discrimination_score, 'TD': progress_span}
