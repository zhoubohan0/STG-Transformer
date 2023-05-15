import argparse
import os
import pickle
import time
import random
import numpy as np
import gym
import warnings
import torch
from torch import nn, optim
from glob import glob
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from atari_env import make_env
from utils import setseed

warnings.filterwarnings('ignore')


class ACAgent(nn.Module):
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def __init__(self, n_action):
        super().__init__()
        self.feature = nn.Sequential(
            self.layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(), nn.Flatten(),
            self.layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = self.layer_init(nn.Linear(512, n_action), std=0.01)
        self.critic = self.layer_init(nn.Linear(512, 1), std=1)

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
    def __init__(self, env):
        super(PPO, self).__init__()
        self.env = env
        self.n_action, self.obs_shape = env.action_space.n, (4, 84, 84)
        self.ACagent = ACAgent(self.n_action).to(device)
        self.optimizer = optim.Adam(self.ACagent.parameters(), lr=args.lr, eps=1e-5)  # trick
        self.episode_returns = []
        self.ss_rewards = []

    def __str__(self):
        return 'PPO'

    def train_ppo(self):
        writer = SummaryWriter(f"ppo-exp/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        # storage
        observations = torch.zeros(args.num_steps, *self.obs_shape).to(device)
        actions = torch.zeros(args.num_steps).to(device)  # 动作是0维标量
        logprobs = torch.zeros(args.num_steps).to(device)
        rewards = torch.zeros(args.num_steps).to(device)
        dones = torch.zeros(args.num_steps).to(device)
        values = torch.zeros(args.num_steps).to(device)
        # training
        global_step, obs, done, ep_step = 0, torch.Tensor(env.reset()).to(device), False, 0
        start_time = time.time()
        num_updates = args.total_timesteps // args.num_steps
        for i_update in range(num_updates):
            if args.anneal_lr:
                frac = 1.0 - i_update / num_updates  # np.power(0.999,i_update /10000.)
                self.optimizer.param_groups[0]["lr"] = frac * args.lr
            # collect num_steps transitions
            ss_states, ep_steps = [[]], [torch.LongTensor([ep_step]).view(1, 1, 1).to(device)]
            for step in range(args.num_steps):
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
                obs, reward, done, info = env.step(action.cpu().numpy())
                obs = torch.Tensor(obs).to(device)
                rewards[step] = torch.tensor(reward).to(device).view(-1)

                for each_key in info:
                    if "episode" in each_key:
                        ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                        self.episode_returns.append(ep_return)
                        n_epi = len(self.episode_returns)
                        if n_epi % args.log_freq == 0:
                            print(
                                f"episode:{n_epi} | global_step:{global_step} | episodic_return:{np.mean(self.episode_returns[-args.log_freq:]):.2f}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        # save
                        if n_epi % args.save_freq == 0:
                            torch.save(self.state_dict(), os.path.join(f'ppo-exp/{args.run_name}',
                                                                       f'{self.__str__()}_epi{n_epi}_rtn{np.mean(self.episode_returns[-args.save_freq:]):.2f}.pth'))
                        # reset
                        ep_step = 0
                        ss_states[-1].append(obs), ss_states.append([]), ep_steps.append(
                            torch.LongTensor([ep_step]).view(1, 1, 1).to(device))
                        obs, done = torch.Tensor(env.reset()).to(device), False
                        break
            ss_states[-1].append(obs)
            intrinsics = reward_model.cal_intrinsic_s2e(ss_states, ep_steps)
            if args.intrinsic == 'ds':
                ss_rewards = args.ds_coef * intrinsics['ADV']
            if args.intrinsic == 'ds_tanh':
                ss_rewards = args.ds_coef * (1.0 * intrinsics['ADV']).tanh()
            if args.intrinsic == 'ps':
                ss_rewards = args.ps_coef * intrinsics['TD']
            if args.intrinsic == 'dsps':
                ss_rewards = args.ds_coef * intrinsics['ADV'] + args.ps_coef * intrinsics['TD']
            rewards = rewards + ss_rewards if args.env_reward else ss_rewards
            self.ss_rewards.extend(ss_rewards.tolist())
            # calculate loss of actor & critic
            with torch.no_grad():
                next_value = self.ACagent.get_value(obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            b_obs = observations.reshape(-1, *self.obs_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,)).long()
            b_advantages = advantages
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_inds, clipfracs = np.arange(args.num_steps), []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.num_steps, args.minibatch_size):
                    mb_inds = b_inds[start:start + args.minibatch_size]
                    _, newlogprob, entropy, newvalue = self.ACagent.get_action_and_value(b_obs[mb_inds],
                                                                                         b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv: mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.vloss_clip:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    # total loss
                    loss = pg_loss \
                           - args.entropy_coef * entropy_loss \
                           + args.vf_coef * v_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ACagent.parameters(), 0.5)
                    self.optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
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
        torch.save(self.state_dict(), os.path.join(f'ppo-exp/{args.run_name}',
                                                   f'{self.__str__()}_epi{len(self.episode_returns)}_rtn{np.mean(self.episode_returns[-args.save_freq:]):.2f}.pth'))
        pickle.dump(self.ss_rewards, open(f'ppo-exp/{args.run_name}/ss_rewards.pkl', 'wb'))
        env.close()
        writer.close()


class Trainer:
    def __init__(self, env, ckpt=None):  # ckpt=None从头训练，ckpt=''用最好的参数，ckpt='*.pth'载入指定参数
        self.env = env
        self.PPOagent = PPO(env).to(device)
        if ckpt is not None: self.load(ckpt)

    def train(self):
        self.PPOagent.train_ppo()

    def load(self, ckpt=''):
        if ckpt == '':
            ckpts = glob(f'../ppo-exp/{args.run_name}/*.pth')
            assert len(ckpts) > 0, 'No pretrained model found'
            ckpt = ckpts[-1]
        self.PPOagent.load_state_dict(torch.load(ckpt))

    def test(self, render=True, save_state=False):
        state = env.reset()
        if save_state: save_states = np.zeros((0, 4, 84, 84))
        state = torch.Tensor(state).to(device)
        while True:
            if render: env.render()
            action = self.PPOagent.ACagent.get_action_and_value(state)[0]
            next_state, reward, done, info = env.step(action.cpu().numpy())
            for each_key in info:
                if "episode" in each_key:
                    ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                    print(f'Trajectory length:{ep_length} | Trajectory return:{ep_return} ')
                    if save_state: np.save(f'_len{ep_length}_rtn{ep_return}.npy', save_states)
                    return ep_return
            if save_state: save_states = np.concatenate((save_states, np.expand_dims(next_state, 0)), axis=0)
            state = torch.Tensor(next_state).to(device)
            time.sleep(0.01)


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default=f'Breakout;p;pNoFrameskip-v4')
    parser.add_argument("--clipframe", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--total_timesteps", type=int, default=int(1e7))
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--anneal_lr", type=int, default=1)
    parser.add_argument("--update_epochs", type=int, default=4)  # PPO updates each time
    parser.add_argument("--clip_coef", type=float, default=0.1)  # surrogate clipping coefficient
    parser.add_argument("--vloss_clip", type=bool, default=True)  # use a clipped loss for the value function
    parser.add_argument("--norm_adv", type=bool, default=True)  # if apply normalization to advantage
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)  # target KL divergence threshold
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--record_video", type=int, default=0)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--pe_coef", type=float, default=-0.5)
    parser.add_argument("--ds_coef", type=float, default=-1)
    parser.add_argument("--ps_coef", type=float, default=0.1)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--intrinsic", type=str, default='ps')
    parser.add_argument("--algo", type=str, default='STG', help="choose from [STG,STG-,ELE]")
    parser.add_argument("--env_reward", type=int, default=0)  # whether to use environmental rewards
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.intrinsic == 'ds' or args.intrinsic == 'ds_tanh':
        coeffs = f'{args.ds_coef}'
    if args.intrinsic == 'ps':
        coeffs = f'{args.ps_coef}'
    if args.intrinsic == 'dsps':
        coeffs = f'{args.ds_coef}_{args.ps_coef}'
    args.run_name = f"{args.algo}_{coeffs}{args.intrinsic}_lambda{args.gae_lambda}_{args.env_name}_seed{args.seed}"
    env = make_env(args.env_id, args.seed, args.capture_video, args.run_name, args.clipframe)
    setseed(args.seed)
    # load reward model
    if 'STG' in args.algo:
        from STGmodel import STGTransformer, STGConfig

        reward_model = STGTransformer(STGConfig()).to(device)
    if 'ELE' == args.algo:
        from ELEmodel import ELE, ELEConfig

        reward_model = ELE(ELEConfig()).to(device)
    reward_model.load(args.pretrained_model)
    # begin training
    trainer = Trainer(env)
    trainer.train()
