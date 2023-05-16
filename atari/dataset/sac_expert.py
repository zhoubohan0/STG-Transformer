import argparse
import copy, pickle
import glob
import os
import random
import time
import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter
from utils import make_env,ReplayBuffer, BatchCollecter, setseed, layer_init
from args import sac_parser
class SoftQNetwork(nn.Module):
    def __init__(self, n_action, obs_shape):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, n_action))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, n_action, obs_shape):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, n_action))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x):
        if x.ndim < 4: x = x.unsqueeze(0)
        logits = self.forward(x / 255.)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class SAC(nn.Module):
    def __init__(self, args):
        super(SAC, self).__init__()
        self.args = args
        self.env = make_env(args.env_id, args.seed, args.capture_video, args.run_name, args.clipframe)
        self.n_action, self.obs_shape = self.env.action_space.n, (4, 84, 84)
        # 网络组件
        self.actor = Actor(self.n_action, self.obs_shape)
        self.qf1 = SoftQNetwork(self.n_action, self.obs_shape)
        self.qf2 = SoftQNetwork(self.n_action, self.obs_shape)
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr, eps=1e-4)
        if args.autotune:  # Automatic entropy tuning
            self.target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(self.n_action))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr, eps=1e-4)
        else:
            self.alpha = args.alpha
        self.replayBuffer = ReplayBuffer(self.obs_shape, 1, args.batch_size, max_size=args.buffer_size)
        self.episode_returns = []

    def __str__(self):
        return 'SAC'

    def train_sac(self):
        setseed(self.args.seed)
        writer = SummaryWriter(os.path.join(root,"sac-exp",self.args.run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )
        # 开始训练
        start_time, obs = time.time(), self.env.reset()
        for global_step in range(self.args.total_timesteps):
            if global_step < self.args.learning_starts:
                actions = np.array(
                    self.env.action_space.sample())  # np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions = self.actor.get_action(torch.Tensor(obs).to(self.args.device))[0].detach().cpu().numpy()
            next_obs, rewards, dones, info = self.env.step(actions)
            self.replayBuffer.add(obs, actions, rewards, next_obs, dones)
            obs = next_obs  # 步进
            for each_key in info:
                if "episode" in each_key:
                    # 记录轨迹长度和回报
                    ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                    self.episode_returns.append(ep_return)
                    n_epi = len(self.episode_returns)
                    if n_epi % self.args.console_log_freq == 0:
                        print(f"episode:{n_epi} | global_step:{global_step} | episodic_return:{ep_return}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    # 保存参数
                    if n_epi % self.args.save_freq == 0:
                        torch.save(self.state_dict(), os.path.join(root,'sac-exp',self.args.run_name,f'{str(self)}_epi{n_epi}_rtn{np.mean(self.episode_returns[-self.args.save_freq:]):.2f}.pth'))
                    # 重置环境
                    obs = self.env.reset()
                    break
            # 更新
            if global_step > self.args.learning_starts:
                if global_step % self.args.update_frequency == 0:
                    b_obs, b_action, b_reward, b_obs_, b_done = self.replayBuffer.sample()
                    b_obs, b_action, b_reward, b_obs_, b_done = b_obs.to(self.args.device), b_action.to(self.args.device), b_reward.to(
                        self.args.device), b_obs_.to(self.args.device), b_done.to(self.args.device)
                    with torch.no_grad():
                        _, next_state_log_pi, next_state_action_probs = self.actor.get_action(b_obs_)
                        qf1_next_target = self.qf1_target(b_obs_)
                        qf2_next_target = self.qf2_target(b_obs_)
                        # use the action probabilities instead of MC sampling to estimate the expectation
                        min_qf_next_target = next_state_action_probs * (
                                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi)
                        # adapt Q-target for discrete Q-function
                        min_qf_next_target = min_qf_next_target.sum(dim=1)
                        next_q_value = b_reward.flatten() + (1 - b_done.flatten()) * self.args.gamma * (min_qf_next_target)
                    # use Q-values only for the taken actions
                    qf1_values = self.qf1(b_obs)
                    qf2_values = self.qf2(b_obs)
                    qf1_a_values = qf1_values.gather(1, b_action).view(-1)
                    qf2_a_values = qf2_values.gather(1, b_action).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    self.q_optimizer.zero_grad()
                    qf_loss.backward()
                    self.q_optimizer.step()

                    # ACTOR training
                    _, log_pi, action_probs = self.actor.get_action(b_obs)
                    with torch.no_grad():
                        qf1_values = self.qf1(b_obs)
                        qf2_values = self.qf2(b_obs)
                        min_qf_values = torch.min(qf1_values, qf2_values)
                    # no need for reparameterization, the expectation can be calculated for discrete actions
                    actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.args.autotune:
                        # re-use action probabilities for temperature loss
                        alpha_loss = (action_probs.detach() * (
                                    -self.log_alpha * (log_pi + self.target_entropy).detach())).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()
                # update the target networks
                if global_step % self.args.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                # 记录
                if global_step % self.args.writer_log_freq == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", self.alpha, global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/step_per_second", int(global_step / (time.time() - start_time)),global_step)
                    if self.args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
        torch.save(self.state_dict(), os.path.join(root,'sac-exp',self.args.run_name,f'{str(self)}_epi{len(self.episode_returns)}_rtn{np.mean(self.episode_returns[-self.args.save_freq:]):.2f}.pth'))
        self.env.close()
        writer.close()


class SACTrainer:
    def __init__(self, args, load=False):
        self.SACagent = SAC(args).to(args.device)
        if load: self.load()

    def train(self):
        self.SACagent.train_sac()

    def load(self):
        ckpts = glob.glob(os.path.join(root,f'checkpoints/{self.args.run_name}/*.pth',))
        assert len(ckpts) > 0, 'No pretrained model found'
        self.SACagent.load_state_dict(torch.load(ckpts[-1]))

    def test(self, render=True):
        setseed(random.randint(0, 10000))
        state = torch.Tensor(self.env.reset()).to(self.args.device)
        while True:
            if render: self.env.render()
            action = self.SACagent.actor.get_action(state)[0]
            next_state, reward, done, info = self.env.step(action.cpu().numpy())
            for each_key in info:
                if "episode" in each_key:
                    ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                    print(f'Trajectory length:{ep_length} | Trajectory return:{ep_return} ')
                    return ep_return
            state = torch.Tensor(next_state).to(self.args.device)
            time.sleep(0.01)


class SACTester:
    def __init__(self, args):
        self.SACagent = SAC(args).to(args.device)

    def load(self, ckpt):
        self.SACagent.load_state_dict(torch.load(ckpt))

    def test_one_episode(self):
        state = torch.Tensor(self.env.reset()).to(self.args.device)
        while True:
            self.env.render()
            action = self.SACagent.actor.get_action(state)[0].detach().cpu().numpy()
            next_state, reward, done, info = self.env.step(action)
            for each_key in info:
                if "episode" in each_key:
                    ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                    print(f'Trajectory length:{ep_length} | Trajectory return:{ep_return} ')
                    return ep_return
            state = torch.Tensor(next_state).to(self.args.device)
            time.sleep(0.01)

    def collect(self, ckpts, save_dir, num, resume=0):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for n_ep in range(resume, resume + num):
            setseed(n_ep)
            self.load(random.choice(ckpts))
            # begin to collect
            collecter = BatchCollecter()
            state, terminal = self.env.reset(), False
            while not terminal:
                action = self.SACagent.actor.get_action(torch.Tensor(state.astype(np.float32)).to(self.args.device))[0].detach().cpu().numpy()
                next_state, reward, done, info = self.env.step(action)
                store_state = (state.copy()*255).astype(np.uint8) if state.max() < 1 else state.copy()
                collecter.add(store_state, action, reward)
                for each_key in info:
                    if "episode" in each_key:
                        ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                        print(f'Episode:{n_ep} | Trajectory length:{ep_length} | Trajectory return:{ep_return}')
                        collecter.save(n_ep, ep_length, ep_return, save_dir)
                        terminal = True
                        break
                state = next_state

    def checkdata(self, states, s2e=None, winname='atari'):
        if states.shape[1] < 10: imgs = states.transpose(0, 2, 3, 1)
        cv2.namedWindow(winname, 0)
        cv2.resizeWindow(winname, 600, 600)
        if s2e is None:
            s2e = (0, imgs.shape[0])
        for i in range(*s2e):
            cv2.imshow(winname, imgs[i])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    args = sac_parser()
    root = os.getcwd()
    if args.mode == 'train':
        trainer = SACTrainer(args)
        trainer.train()
    if args.mode == 'test':
        save_dir = os.path.join(root,f'{args.env_id[: -14]}')
        tester = SACTester(args)
        tester.collect(args.test_checkpoints, save_dir, 50)

