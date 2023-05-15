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
from atari_env import make_env
from utils import ReplayBuffer, BatchCollecter, setseed, layer_init

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
    def __init__(self, env):
        super(SAC, self).__init__()
        self.n_action, self.obs_shape = env.action_space.n, (4, 84, 84)
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
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr, eps=1e-4)
        else:
            self.alpha = args.alpha
        self.replayBuffer = ReplayBuffer(self.obs_shape, 1, args.batch_size, max_size=args.buffer_size)
        self.episode_returns = []

    def __str__(self):
        return 'SAC'

    def train_sac(self):
        writer = SummaryWriter(f"sac-exp/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        # 开始训练
        start_time, obs = time.time(), env.reset()
        for global_step in range(args.total_timesteps):
            if global_step < args.learning_starts:
                actions = np.array(
                    env.action_space.sample())  # np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions = self.actor.get_action(torch.Tensor(obs).to(device))[0].detach().cpu().numpy()
            next_obs, rewards, dones, info = env.step(actions)
            self.replayBuffer.add(obs, actions, rewards, next_obs, dones)
            obs = next_obs  # 步进
            for each_key in info:
                if "episode" in each_key:
                    # 记录轨迹长度和回报
                    ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                    self.episode_returns.append(ep_return)
                    n_epi = len(self.episode_returns)
                    if n_epi % args.console_log_freq == 0:
                        print(f"episode:{n_epi} | global_step:{global_step} | episodic_return:{ep_return}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    # 保存参数
                    if n_epi % args.save_freq == 0:
                        torch.save(self.state_dict(), os.path.join(f'sac-exp/{args.run_name}',
                                                                   f'{str(self)}_epi{n_epi}_rtn{np.mean(self.episode_returns[-args.save_freq:]):.2f}.pth'))
                    # 重置环境
                    obs = env.reset()
                    break
            # 更新
            if global_step > args.learning_starts:
                if global_step % args.update_frequency == 0:
                    b_obs, b_action, b_reward, b_obs_, b_done = self.replayBuffer.sample()
                    b_obs, b_action, b_reward, b_obs_, b_done = b_obs.to(device), b_action.to(device), b_reward.to(
                        device), b_obs_.to(device), b_done.to(device)
                    with torch.no_grad():
                        _, next_state_log_pi, next_state_action_probs = self.actor.get_action(b_obs_)
                        qf1_next_target = self.qf1_target(b_obs_)
                        qf2_next_target = self.qf2_target(b_obs_)
                        # use the action probabilities instead of MC sampling to estimate the expectation
                        min_qf_next_target = next_state_action_probs * (
                                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi)
                        # adapt Q-target for discrete Q-function
                        min_qf_next_target = min_qf_next_target.sum(dim=1)
                        next_q_value = b_reward.flatten() + (1 - b_done.flatten()) * args.gamma * (min_qf_next_target)
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

                    if args.autotune:
                        # re-use action probabilities for temperature loss
                        alpha_loss = (action_probs.detach() * (
                                    -self.log_alpha * (log_pi + self.target_entropy).detach())).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()
                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                # 记录
                if global_step % args.writer_log_freq == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", self.alpha, global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/step_per_second", int(global_step / (time.time() - start_time)),
                                      global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
        torch.save(self.state_dict(), os.path.join(f'sac-exp/{args.run_name}',
                                                   f'{str(self)}_epi{len(self.episode_returns)}_rtn{np.mean(self.episode_returns[-args.save_freq:]):.2f}.pth'))
        env.close()
        writer.close()


class Trainer:
    def __init__(self, env, load=False):
        self.SACagent = SAC(env).to(device)
        if load: self.load()

    def train(self):
        self.SACagent.train_sac()

    def load(self):
        ckpts = glob.glob(f'checkpoints/{args.run_name}/*.pth')
        assert len(ckpts) > 0, 'No pretrained model found'
        self.SACagent.load_state_dict(torch.load(ckpts[-1]))

    def test(self, render=True):
        setseed(random.randint(0, 10000))
        state = torch.Tensor(env.reset()).to(device)
        while True:
            if render: env.render()
            action = self.SACagent.actor.get_action(state)[0]
            next_state, reward, done, info = env.step(action.cpu().numpy())
            for each_key in info:
                if "episode" in each_key:
                    ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                    print(f'Trajectory length:{ep_length} | Trajectory return:{ep_return} ')
                    return ep_return
            state = torch.Tensor(next_state).to(device)
            time.sleep(0.01)


class Tester:
    def __init__(self, env):
        self.env = env
        self.SACagent = SAC(env).to(device)

    def load(self, ckpt):
        self.SACagent.load_state_dict(torch.load(ckpt))

    def test_one_episode(self):
        state = torch.Tensor(env.reset()).to(device)
        while True:
            env.render()
            action = self.SACagent.actor.get_action(state)[0].detach().cpu().numpy()
            next_state, reward, done, info = env.step(action)
            for each_key in info:
                if "episode" in each_key:
                    ep_return, ep_length = info['episode']['r'], info["episode"]["l"]
                    print(f'Trajectory length:{ep_length} | Trajectory return:{ep_return} ')
                    return ep_return
            state = torch.Tensor(next_state).to(device)
            time.sleep(0.01)

    def collect(self, ckpts, save_dir, num, resume=0):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for n_ep in range(resume, resume + num):
            setseed(n_ep)
            self.load(random.choice(ckpts))
            # 开始采集
            collecter = BatchCollecter()
            state, terminal = env.reset(), False  # 注意状态是[0,1]还是[0,255]
            while not terminal:
                action = self.SACagent.actor.get_action(torch.Tensor(state.astype(np.float32)).to(device))[
                    0].detach().cpu().numpy()
                next_state, reward, done, info = env.step(action)
                collecter.add(state, action, reward)
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
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=666, help="seed of the experiment")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb", type=bool, default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb_project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--capture_video", type=int, default=0, nargs="?",
                        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--writer_log_freq", type=int, default=100)
    parser.add_argument("--console_log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=500)
    # Algorithm specific arguments
    parser.add_argument("--env_id", type=str, default="BreakoutNoFrameskip-v4",
                        help="the id of the environment")  # BeamRider
    parser.add_argument("--clipframe", type=int, default=1, help='直接将原图resize/选取18:102高度再resize(84,84)')
    parser.add_argument("--total_timesteps", type=int, default=int(5e6), help="total timesteps of the experiments")
    parser.add_argument("--buffer_size", type=int, default=int(1e6),
                        help="the replay memory buffer size")  # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="target smoothing coefficient (default: 1)")  # Default is 1 to perform replacement update
    parser.add_argument("--batch_size", type=int, default=64, help="the batch size of sample from the reply memory")
    parser.add_argument("--learning_starts", type=int, default=2e4, help="timestep to start learning")
    parser.add_argument("--policy_lr", type=float, default=3e-4,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q_lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--update_frequency", type=int, default=4, help="the frequency of training updates")
    parser.add_argument("--target_network_frequency", type=int, default=8000,
                        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=bool, default=True, nargs="?", const=True,
                        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--target_entropy_scale", type=float, default=0.89,
                        help="coefficient for scaling the autotune entropy target")
    parser.add_argument("--test_checkpoints", nargs='*', default=[])
    parser.add_argument("--mode", type=str, default='train')
    args = parser.parse_args()
    args.run_name = f'SAC_{args.env_id}_seed{args.seed}'
    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setseed(args.seed)
    env = make_env(args.env_id, args.seed, args.capture_video, args.run_name, args.clipframe)
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    if args.mode == 'train':
        trainer = Trainer(env)
        trainer.train()
    if args.mode == 'test':
        save_dir = f'expert/{args.env_id[: -14]}'
        tester = Tester(env)
        tester.collect(args.test_checkpoints, save_dir, 50)

