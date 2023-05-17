import os
import time
import warnings
from glob import glob

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.distributions import Categorical

from args import ppo_parser
from util import make_env, setseed, layer_init

warnings.filterwarnings('ignore')


class ACAgent(nn.Module):
    def __init__(self, n_action):
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


class PPO(nn.Module):
    def __init__(self, args):
        super(PPO, self).__init__()
        self.args = args
        self.env = make_env(args.env_name, args.seed, args.record_video, args.run_name, args.clipframe)
        self.n_action, self.obs_shape = self.env.action_space.n, (4, 84, 84)
        self.ACagent = ACAgent(self.n_action).to(self.args.device)
        self.optimizer = optim.Adam(self.ACagent.parameters(), lr=self.args.lr, eps=1e-5)  # trick
        self.episode_returns = []

    def __str__(self):
        return 'PPO'

    def train_ppo(self, reward_model):
        setseed(self.args.seed)
        writer = SummaryWriter(f"ppo-exp/{self.args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )
        # storage
        observations = torch.zeros(self.args.num_steps, *self.obs_shape).to(self.args.device)
        actions = torch.zeros(self.args.num_steps).to(self.args.device)  # 动作是0维标量
        logprobs = torch.zeros(self.args.num_steps).to(self.args.device)
        rewards = torch.zeros(self.args.num_steps).to(self.args.device)
        dones = torch.zeros(self.args.num_steps).to(self.args.device)
        values = torch.zeros(self.args.num_steps).to(self.args.device)
        # training
        global_step, obs, done, ep_step = 0, torch.Tensor(self.env.reset()).to(self.args.device), False, 0
        start_time = time.time()
        num_updates = self.args.total_timesteps // self.args.num_steps
        for i_update in range(num_updates):
            if self.args.anneal_lr:
                frac = 1.0 - i_update / num_updates  # np.power(0.999,i_update /10000.)
                self.optimizer.param_groups[0]["lr"] = frac * self.args.lr
            # collect num_steps transitions
            ss_states, ep_steps = [[]], [torch.LongTensor([ep_step]).view(1, 1, 1).to(self.args.device)]
            for step in range(self.args.num_steps):
                ep_step += 1
                global_step += 1
                observations[step] = obs
                ss_states[-1].append(obs)
                dones[step] = done

                with torch.no_grad():
                    action, logprob, _, value = self.ACagent.get_action_and_value(obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                obs, reward, done, info = self.env.step(action.cpu().numpy())
                obs = torch.Tensor(obs).to(self.args.device)
                rewards[step] = torch.tensor(reward).to(self.args.device).view(-1)

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
                        # save
                        if n_epi % self.args.save_freq == 0:
                            torch.save(self.state_dict(), os.path.join(f'ppo-exp/{self.args.run_name}',
                                                                       f'{str(self)}_epi{n_epi}_rtn{np.mean(self.episode_returns[-self.args.save_freq:]):.2f}.pth'))
                        # reset
                        ep_step = 0
                        ss_states[-1].append(obs), ss_states.append([]), ep_steps.append(
                            torch.LongTensor([ep_step]).view(1, 1, 1).to(self.args.device))
                        obs, done = torch.Tensor(self.env.reset()).to(self.args.device), False
                        break
            ss_states[-1].append(obs)
            intrinsics = reward_model.cal_intrinsic_s2e(ss_states, ep_steps)
            if self.args.intrinsic == 'ds':
                ss_rewards = self.args.ds_coef * intrinsics['ADV']
            if self.args.intrinsic == 'ds_tanh':
                ss_rewards = self.args.ds_coef * (1.0 * intrinsics['ADV']).tanh()
            if self.args.intrinsic == 'ps':
                ss_rewards = self.args.ps_coef * intrinsics['TD']
            if self.args.intrinsic == 'dsps':
                ss_rewards = self.args.ds_coef * intrinsics['ADV'] + self.args.ps_coef * intrinsics['TD']
            rewards = rewards + ss_rewards if self.args.env_reward else ss_rewards
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
            b_advantages = advantages
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
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ACagent.parameters(), 0.5)
                    self.optimizer.step()

                if self.args.target_kl is not None:
                    if approx_kl > self.args.target_kl:
                        break
            # value variance
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            # record
            writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/step_per_second", int(global_step / (time.time() - start_time)), global_step)
        # save checkpoints
        torch.save(self.state_dict(), os.path.join(f'ppo-exp/{self.args.run_name}',
                                                   f'{self.__str__()}_epi{len(self.episode_returns)}_rtn{np.mean(self.episode_returns[-self.args.save_freq:]):.2f}.pth'))
        self.env.close()
        writer.close()


class STGTrainer:
    def __init__(self, args, ckpt=None):
        self.PPOagent = PPO(args).to(args.device)
        if ckpt is not None:
            self.load(ckpt)

    def train(self, reward_model):
        self.PPOagent.train_ppo(reward_model)

    def load(self, ckpt=''):
        if ckpt == '':
            ckpts = glob(f'ppo-exp/{self.args.run_name}/*.pth')
            assert len(ckpts) > 0, 'No pretrained model found'
            ckpt = ckpts[-1]
        self.PPOagent.load_state_dict(torch.load(ckpt))


if __name__ == '__main__':
    args = ppo_parser()

    # load reward model
    if args.algo in ['STG', 'stg', 'STG-', 'stg-']:
        from network import STGTransformer
        from config import STGConfig

        reward_model = STGTransformer(STGConfig()).to(args.device)
    if args.algo in ['ELE', 'ele']:
        from network import ELE
        from config import ELEConfig

        reward_model = ELE(ELEConfig()).to(args.device)
    if os.path.exists(args.pretrained_model):
        reward_model.load(args.pretrained_model)

    # begin training
    trainer = STGTrainer(args)
    trainer.train(reward_model)
