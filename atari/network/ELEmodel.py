import random
import os
import numpy as np
import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
from utils import Logger
from networks import AtariCNN,TDR


class ELE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # input embedding
        self.encoder = AtariCNN(4,config.n_embd)
        # final projection
        self.out = nn.Linear(config.n_embd, config.n_embd)
        # temporal distance regressor
        self.tdr = TDR(config.n_embd, config.n_embd,1)
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
        # no_decay.add('pos_emb')
        # no_decay.add('global_pos_emb')

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
        input_embedding = self.encoder(data.view(-1, c, w, h)).view(batch_size, block_size, self.config.n_embd)
        # calculate temporal distance
        idx1 = np.arange(self.config.block_size)
        idx2 = np.random.permutation(idx1)
        timesteps = timesteps.repeat(block_size).reshape(-1,block_size)
        idx1,idx2,timesteps = torch.LongTensor(idx1).to(self.config.device),torch.LongTensor(idx2).to(self.config.device),torch.LongTensor(timesteps).to(self.config.device)
        time_dis = self.tdr.symlog(idx1 + timesteps, idx2 + timesteps)
        time_pred = self.tdr(input_embedding[:,idx1,:],input_embedding[:,idx2,:])
        tdr_loss = F.mse_loss(time_pred.view(time_dis.shape), time_dis)
        # total loss
        loss = self.config.tdr_coff * tdr_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return tdr_loss.item()

    def trainELE(self,src):
        root = './pretrain-exp'
        if not os.path.exists(root):
            os.mkdir(root)
        save_dir = os.path.join(root, f"{self.config.game}_ELE")
        exp_logger = Logger(save_dir, f'trainele.csv',fieldnames=['update', 'TDR_loss'])
        self.train()
        src_len = len(src)
        for i_epoch in range(self.config.n_epoch):
            random.shuffle(src)
            for i_src, d in enumerate(src):
                states = pickle.load(open(os.path.join('.', d), 'rb'))['states']
                step = np.random.choice(states.shape[0] - self.config.block_size, size=self.config.batch_size, replace=False)
                batch = torch.Tensor(states[np.array([list(range(id, id + self.config.block_size)) for id in step])]) / 255.0  # (config.batch_size, config.block_size+1, 4, 84, 84)
                tdr_loss = self.update(batch.to(self.config.device), step)
                num = i_epoch * src_len + i_src + 1
                exp_logger.update(fieldvalues=[num, tdr_loss])
                if num % 5000 == 0:
                    torch.save(self, f'{save_dir}/ELE_{num}.pt')
            print(f'update:{num:05d} | TDR_loss:{tdr_loss:.6f}')
        num_update = src_len * self.config.n_epoch
        torch.save(self, f'{save_dir}/ELE_{num_update}.pt')

    def cal_intrinsic_s2e(self,states_ls,steps_ls):
        '''
        states:(block_size+1,4,84,84)
        step:(1,)
        '''
        self.eval()
        with torch.no_grad():
            intrinsic = torch.Tensor(0).to(steps_ls[0].device)
            for i in range(len(steps_ls)):
                states = torch.stack(states_ls[i])
                with torch.no_grad():
                    embeddings = self.encoder(states).unsqueeze(0)  # (1, block_size+1, n_embd)
                    cur_embedding,next_embedding = embeddings[:, :-1, :],embeddings[:, 1:, :] # (1, block_size+1, n_embd)
                    intrinsic = self.tdr.symexp(self.tdr(cur_embedding,next_embedding).view(-1))
                    intrinsic = torch.cat((intrinsic, intrinsic), dim=0)
            return {'TD':intrinsic}
