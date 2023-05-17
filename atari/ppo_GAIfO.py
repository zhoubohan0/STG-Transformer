import argparse
import itertools
import os
import pickle
import random
import time
import warnings
from glob import glob

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn.utils import spectral_norm

from util import make_env, setseed, layer_init

warnings.filterwarnings('ignore')


class ACAgent(nn.Module):
    def __init__(self, n_action, emb_dim=512):
        super().__init__()
        self.feature = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(), nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_action), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.discriminator = nn.Sequential(
            nn.Linear(emb_dim * 2, 512),
            nn.GELU(),
            spectral_norm(nn.Linear(512, 256)),
            nn.GELU(),
            spectral_norm(nn.Linear(256, 128)),
            nn.GELU(),
            spectral_norm(nn.Linear(128, 64)),
            nn.GELU(),
            spectral_norm(nn.Linear(64, 32)),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.dis_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.95))

    def get_value(self, x):
        if x.ndim < 4: x = x.unsqueeze(0)
        return self.critic(self.feature(x))

    def get_action_and_value(self, x, action=None):
        if x.ndim < 4: x = x.unsqueeze(0)
        hidden = self.feature(x)
        probs = Categorical(logits=self.actor(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def update_cal_intrinsic(self, states_ls, expert_state_pool):
        device, Dloss_ls = states_ls[0][0].device, []
        GAIfO_intrinsic = torch.Tensor(0).to(device)
        for i in range(len(states_ls)):
            interaction_states = torch.stack(states_ls[i])
            b = random.choice(range(len(expert_state_pool) - len(interaction_states)))
            expert_states = torch.Tensor(expert_state_pool[b:b + len(interaction_states)]).to(device)
            # update
            interaction_score = self.discriminator(
                torch.cat((self.feature(interaction_states[:-1]), self.feature(interaction_states[1:])), -1)).squeeze(
                -1)  # (block_size,1)
            expert_score = self.discriminator(
                torch.cat((self.feature(expert_states[:-1]), self.feature(expert_states[1:])), -1)).squeeze(
                -1)  # (block_size,1)
            D_loss = self.bce(interaction_score, torch.ones_like(interaction_score)) + self.bce(expert_score,
                                                                                                torch.zeros_like(
                                                                                                    expert_score))  # 1:non-expert;0:expert
            self.dis_optimizer.zero_grad()
            D_loss.backward()
            self.dis_optimizer.step()
            Dloss_ls.append(D_loss.item())
            with torch.no_grad():
                D = torch.sigmoid(interaction_score)
                intri = torch.log(D + 1e-5)  # -torch.log(1+1e-5-D)##torch.log(D/(1-D)+1e-5)
                GAIfO_intrinsic = torch.cat((GAIfO_intrinsic, -intri), dim=0)
        return GAIfO_intrinsic, np.mean(Dloss_ls)


class PPO(nn.Module):
    def __init__(self, args):
        super(PPO, self).__init__()
        self.args = args
        self.env = make_env(self.args.env_name, self.args.seed, self.args.record_video, self.args.run_name,
                            self.args.clipframe)
        self.n_action, self.obs_shape = self.env.action_space.n, (4, 84, 84)
        self.ACagent = ACAgent(self.n_action)
        self.ac_optimizer = optim.Adam(
            itertools.chain(self.ACagent.feature.parameters(), self.ACagent.actor.parameters(),
                            self.ACagent.critic.parameters()), lr=self.args.lr, eps=1e-5)  # trick
        self.episode_returns = []
        self.intrinsic_rewards = []

    def __str__(self):
        return 'GAIfO'

    def train_ppo(self):
        setseed(self.args.seed)
        writer = SummaryWriter(f"ppo-exp/{self.args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )
        observations = torch.zeros(self.args.num_steps, *self.obs_shape).to(self.args.device)
        actions = torch.zeros(self.args.num_steps).to(self.args.device)
        logprobs = torch.zeros(self.args.num_steps).to(self.args.device)
        # rewards = torch.zeros(args.num_steps).to(device)
        dones = torch.zeros(self.args.num_steps).to(self.args.device)
        values = torch.zeros(self.args.num_steps).to(self.args.device)
        # to train
        global_step, obs, done, ep_step = 0, torch.Tensor(self.env.reset()).to(self.args.device), False, 0
        start_time = time.time()
        num_updates = self.args.total_timesteps // self.args.num_steps
        for i_update in range(num_updates):
            if self.args.anneal_lr:
                frac = 1.0 - i_update / num_updates  # np.power(0.999,i_update /10000.)
                self.ac_optimizer.param_groups[0]["lr"] = frac * self.args.lr
            collect_states, ep_steps = [[]], [torch.LongTensor([ep_step]).view(1, 1, 1).to(self.args.device)]
            for step in range(self.args.num_steps):
                ep_step += 1
                global_step += 1
                observations[step] = obs
                collect_states[-1].append(obs)
                dones[step] = done

                with torch.no_grad():
                    action, logprob, _, value = self.ACagent.get_action_and_value(obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                obs, reward, done, info = self.env.step(action.cpu().numpy())
                obs = torch.Tensor(obs).to(self.args.device)
                # rewards[step] = torch.tensor(reward).to(device).view(-1)

                for each_key in info:
                    if "episode" in each_key:
                        ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                        self.episode_returns.append(ep_return)
                        n_epi = len(self.episode_returns)
                        if n_epi % self.args.log_freq == 0:
                            print(
                                f"episode:{n_epi} | global_step:{global_step} | episodic_return:{np.mean(self.episode_returns[-self.args.log_freq:]):.2f}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        if n_epi % self.args.save_freq == 0:
                            torch.save(self.state_dict(), os.path.join(f'ppo-exp/{self.args.run_name}',
                                                                       f'{self.__str__()}_epi{n_epi}_rtn{np.mean(self.episode_returns[-self.args.save_freq:]):.2f}.pth'))
                        ep_step = 0
                        collect_states[-1].append(obs), collect_states.append([]), ep_steps.append(
                            torch.LongTensor([ep_step]).view(1, 1, 1).to(self.args.device))
                        obs, done = torch.Tensor(self.env.reset()).to(self.args.device), False
                        break
            collect_states[-1].append(obs)
            # intrinsic
            if i_update % 20 == 0:
                expert_obs_pool = pickle.load(open(random.choice(src), 'rb'))['states']  # (n,4,84,84)
            intrinsic, dloss = self.ACagent.update_cal_intrinsic(collect_states, expert_obs_pool)
            rewards = self.args.intrinsic_coef * intrinsic
            self.intrinsic_rewards.extend(intrinsic.tolist())
            # calculate loss of actor & critic
            with torch.no_grad():
                next_value = self.ACagent.get_value(obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.args.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[
                        t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            b_obs = observations.reshape(-1, *self.obs_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,)).long()
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_inds, clipfracs = np.arange(self.args.num_steps), []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.num_steps, self.args.minibatch_size):
                    mb_inds = b_inds[start:start + self.args.minibatch_size]
                    _, newlogprob, entropy, newvalue = self.ACagent.get_action_and_value(b_obs[mb_inds],
                                                                                         b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv: mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8)
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.vloss_clip:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    # total loss
                    loss = pg_loss \
                           - self.args.entropy_coef * entropy_loss \
                           + self.args.vf_coef * v_loss
                    self.ac_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        itertools.chain(self.ACagent.feature.parameters(), self.ACagent.actor.parameters(),
                                        self.ACagent.critic.parameters()), 0.5)
                    self.ac_optimizer.step()

                if self.args.target_kl is not None:
                    if approx_kl > self.args.target_kl:
                        break
            # value variance
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            # record
            writer.add_scalar("charts/learning_rate", self.ac_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/Discriminator_loss", dloss, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/step_per_second", int(global_step / (time.time() - start_time)), global_step)
        # save
        torch.save(self.state_dict(), os.path.join(f'ppo-exp/{self.args.run_name}',
                                                   f'{self.__str__()}_epi{len(self.episode_returns)}_rtn{np.mean(self.episode_returns[-self.args.save_freq:]):.2f}.pth'))
        # pickle.dump(self.intrinsic_rewards,open(f'ppo-exp/{self.args.run_name}/intrinsic_rewards.pkl','wb'))
        self.env.close()
        writer.close()


class GAIfOTrainer:
    def __init__(self, args, load=False):
        self.args = args
        self.PPOagent = PPO(args).to(args.device)
        if load: self.load()

    def train(self):
        self.PPOagent.train_ppo()

    def load(self):
        ckpts = glob(f'ppo-exp/{self.args.run_name}/*.pth')
        assert len(ckpts) > 0, 'No pretrained model found'
        self.PPOagent.load_state_dict(torch.load(ckpts[-1]))


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, default='')
    parser.add_argument("--algo", type=str, default='GAIfO')
    parser.add_argument("--env_name", type=str, default=f'BreakoutNoFrameskip-v4')
    parser.add_argument("--clipframe", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--total_timesteps", type=int, default=int(1e7))
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--anneal_lr", type=bool, default=True)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--vloss_clip", type=bool, default=True)
    parser.add_argument("--norm_adv", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--record_video", type=bool, default=False)
    parser.add_argument("--intrinsic_coef", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.run_name = f'{args.algo}_{-args.intrinsic_coef}_lmbda{args.gae_lambda}_{args.env_name}_seed{args.seed}'

    if not args.src_root:
        args.src_root = f'dataset/{args.env_name[:-14]}'
    src = glob(os.path.join(args.src_root, '*.pkl'))

    trainer = GAIfOTrainer(args)
    trainer.train()
