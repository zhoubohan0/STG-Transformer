import pickle
import time
import os
from collections import defaultdict
from csv import DictWriter
import random

import cv2
import numpy as np
import torch
from torch import nn


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


class ReplayBuffer(object):
    def __init__(self, obs_shape, action_dim, batch_size, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.batch_size = batch_size
        self.state = np.zeros((max_size, *obs_shape), dtype='uint8')
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, *obs_shape), dtype='uint8')
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.LongTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]).view(-1, 1),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.done[ind]).view(-1, 1)
        )


class BatchCollecter:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.ep_length = 0
        self.ep_return = 0

    def add(self, s, a, r, ):
        '''r的索引范围从0~T-1，表示R[t]=1'''
        self.states.append(s)
        self.actions.append(a)
        if r: self.rewards.append(r)

    def save(self, n_ep, length, rtn, root):
        self.ep_length = length
        self.ep_return = rtn
        self.states = np.stack(self.states)  # .astype(np.uint8)
        self.actions = np.stack(self.actions)
        self.rewards = np.array(self.rewards)
        pickle.dump({
            'length': self.ep_length,
            'return': self.ep_return,
            'states': self.states,
            'actions': self.actions,
             'rewards': self.rewards
            },
            open(os.path.join(root, f'epi{n_ep}_len{self.ep_length}_rtn{int(self.ep_return)}.pkl'), 'wb')
        )


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def show(img, winname='atari'):
    cv2.namedWindow(winname, 0)
    cv2.resizeWindow(winname, 600, 600)
    cv2.imshow(winname, img)
    cv2.waitKey(0)


def checkdata(imgs, s2e=None, winname='atari'):  # imgs是(n,w,h,c)的np.uint8
    cv2.namedWindow(winname, 0)
    cv2.resizeWindow(winname, 600, 600)
    if s2e is None:
        s2e = (0,imgs.shape[0])
    for i in range(*s2e):
        cv2.imshow(winname,imgs[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


