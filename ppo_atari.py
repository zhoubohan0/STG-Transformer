import argparse
import glob
import os
import pickle
import time
import random
import numpy as np
import gym
import cv2
import warnings
import torch
from torch import nn, optim
from collections import deque
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')


# 环境准备
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ProcessFrame84Env(gym.Wrapper):
    def _process_frame84(self, frame):
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        if self.clipframe:
            x_t = cv2.resize(img, (84, 110), interpolation=cv2.INTER_LINEAR)[18:102, :]
        else:
            x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        # x_t = np.reshape(x_t, [84, 84])  # , 1
        return x_t.astype(np.uint8)

    def __init__(self, env=None,clipframe=True):
        super(ProcessFrame84Env, self).__init__(env)
        self.clipframe = clipframe
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_frame84(obs), reward, done, info

    def reset(self):
        return self._process_frame84(self.env.reset())


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FrameStackEnv(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        # assert len(self.frames) == self.k
        # return LazyFrames(list(self.frames))
        return np.stack(list(self.frames))


class ScaledFloatEnv(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


def wrap_deepmind(args):
    """Configure environment for DeepMind-style Atari."""
    # assert 'NoFrameskip' in env.spec.id
    env = gym.make(args.env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.record_vido: env = gym.wrappers.Monitor(env, f"videos/{args.run_name}")
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings(): env = FireResetEnv(env)
    env = ProcessFrame84Env(env,args.clipframe)
    env = ClipRewardEnv(env)
    env = FrameStackEnv(env, k=4)
    env = ScaledFloatEnv(env)
    return env


def show(img, winname='atari'):
    cv2.namedWindow(winname, 0)
    cv2.resizeWindow(winname, 600, 600)
    cv2.imshow(winname, img)
    cv2.waitKey(0)


# 网络
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
            nn.ReLU(),nn.Flatten(),
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

    def __str__(self):
        return 'PPO'

    def train_ppo(self):
        writer = SummaryWriter(f"ppo-exp/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        # 存储（不断被覆盖更新）
        observations = torch.zeros(args.num_steps, *self.obs_shape).to(device)
        actions = torch.zeros(args.num_steps).to(device)  # 动作是0维标量
        logprobs = torch.zeros(args.num_steps).to(device)
        rewards = torch.zeros(args.num_steps).to(device)
        dones = torch.zeros(args.num_steps).to(device)
        values = torch.zeros(args.num_steps).to(device)
        # 开始训练
        global_step, obs, done = 0, torch.Tensor(env.reset()).to(device), False
        start_time = time.time()
        num_updates = args.total_timesteps // args.num_steps
        for i_update in range(1, num_updates + 1):
            if args.anneal_lr:
                frac = 1.0 - (i_update - 1.0) / num_updates
                self.optimizer.param_groups[0]["lr"] = frac * args.lr
            # 采集num_steps个交互
            for step in range(args.num_steps):
                global_step += 1
                observations[step] = obs
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
                        # 记录轨迹长度和回报
                        ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                        self.episode_returns.append(ep_return)
                        n_epi = len(self.episode_returns)
                        print(f"episode:{n_epi} | global_step:{global_step} | episodic_return:{ep_return}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        # 保存参数
                        if n_epi % args.save_freq == 0:
                            torch.save(self.state_dict(), os.path.join(f'ppo-exp/{args.run_name}',f'{self.__str__()}_epi{n_epi}_rtn{np.mean(self.episode_returns[-args.save_freq:]):.2f}.pth'))
                        # 重置环境
                        obs, done = torch.Tensor(env.reset()).to(device), False
                        break
            # 计算actor和critic损失
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
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            # 更新update_epochs次
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
                    # 总loss并优化
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
            # 体现更新value方差变化
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            # 记录
            writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/step_per_second", int(global_step / (time.time() - start_time)), global_step)
        torch.save(self.state_dict(), os.path.join(f'ppo-exp/{args.run_name}',f'{self.__str__()}_epi{len(self.episode_returns)}_rtn{np.mean(self.episode_returns[-args.save_freq:]):.2f}.pth'))
        pickle.dump(self.episode_returns,open(f'ppo-exp/{args.run_name}/return.pkl','wb'))
        env.close()
        writer.close()


class Trainer:
    def __init__(self, env, load=False):
        self.env = env
        self.PPOagent = PPO(env).to(device)
        if load: self.load()

    def train(self):
        self.PPOagent.train_ppo()

    def load(self):
        ckpts = glob.glob(f'ppo-exp/{args.run_name}/*.pth')
        assert len(ckpts) > 0,'No pretrained model found'
        self.PPOagent.load_state_dict(torch.load(ckpts[-1]))

    def test(self, render=True):
        state = torch.Tensor(env.reset()).to(device)
        while True:
            if render: env.render()
            action = self.PPOagent.ACagent.get_action_and_value(state)[0]
            next_state, reward, done, info = env.step(action.cpu().numpy())
            for each_key in info:
                if "episode" in each_key:
                    ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                    print(f'Trajectory length:{ep_length} | Trajectory return:{ep_return} ')
                    return ep_return
            state = torch.Tensor(next_state).to(device)
            time.sleep(0.03)


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='PongNoFrameskip-v4')  # SeaquestNoFrameskip-v4 BreakoutNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 FreewayNoFrameskip-v4 MontezumaRevengeNoFrameskip-v4 PongNoFrameskip-v4 MsPacmanNoFrameskip-v4
    parser.add_argument("--clipframe", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--total_timesteps", type=int, default=int(1e7))
    parser.add_argument("--minibatch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--anneal_lr", type=bool, default=True)
    parser.add_argument("--update_epochs", type=int, default=4)  # PPO策略更新次数
    parser.add_argument("--clip_coef", type=float, default=0.1)  # surrogate clipping coefficient
    parser.add_argument("--vloss_clip", type=bool, default=True)  # use a clipped loss for the value function
    parser.add_argument("--norm_adv", type=bool, default=True)  # 对adv是否标准化
    parser.add_argument("--entropy_coef", type=float, default=0.01)  # 总loss中entropy系数
    parser.add_argument("--vf_coef", type=float, default=0.5)  # 总loss中value loss系数
    parser.add_argument("--target-kl", type=float, default=None)  # target KL divergence threshold
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--record_vido", type=bool, default=False)

    args = parser.parse_args()
    args.run_name = f'PPO_{args.env_name}_seed{args.seed}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 环境
    env = wrap_deepmind(args)
    # 保证随机种子
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # 训练
    trainer = Trainer(env)
    trainer.train()
    test_return = trainer.test(False)
    print(test_return)
