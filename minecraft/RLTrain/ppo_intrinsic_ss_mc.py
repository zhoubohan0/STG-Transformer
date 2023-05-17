import mineagent.common
import os
import time
import torch
import numpy as np
from mineagent.common.logx import EpochLogger
import imageio
from mineagent.official import torch_normalize
from RLTrain.minecraft import MinecraftEnv, preprocess_obs, transform_action
from mineagent.batch import Batch
from mineagent import features, SimpleFeatureFusion, MineAgent, MultiCategoricalActor, Critic
import copy
import pickle
from minedojo.sim import InventoryItem


# PPO buffer, but the observation is stored with Batch
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, act_dim, size=1000, gamma=0.99, lam=0.95, agent_model='mineagent', obs_dim=None):
        self.agent_model = agent_model
        if agent_model == 'mineagent':
            self.obs_buf = [Batch() for i in range(size)]
        else:
            self.obs_buf = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(size, act_dim), dtype=np.int)
        # self.act2_buf = np.zeros(utils.combined_shape(size, act2_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        # self.act2_buf[self.ptr] = act[1]
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def modify_trajectory_rewards(self, rews):
        """
        modify the recently saved rewards with a numpy array: rews.
        should be called after many store(), and before finish_path().
        """
        assert self.ptr - self.path_start_idx == len(rews)
        self.rew_buf[self.path_start_idx: self.ptr] = rews

    def finish_path(self, rew, last_val=0,lmbda=0.1):
        # rew:last reward of one trajectory (gamma=1 return)
        # last_val: v(s_last)
        # lmbda: GAE 
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        # vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        # deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.adv_buf[path_slice] = rews[:-1] + rew[-1]  # 加上最后一个sparse reward(gamma=1下的return)        #utils.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = utils.discount_cumsum(rews, self.gamma*lmbda)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)  # mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        if self.agent_model == 'mineagent':
            data = dict(act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
            rtn = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
            rtn['obs'] = Batch.cat(self.obs_buf)
        else:
            data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
            rtn = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        return rtn


# self-imitation learning buffer
class SelfImitationBuffer:
    def __init__(self, act_dim, size=500, imitate_success_only=True, agent_model='mineagent'):
        '''
        each saved item is a trajectory: act_buf [[len, act_dim], ...]
        '''
        self.obs_buf = []
        self.act_buf = []  # np.zeros(utils.combined_shape(size, act_dim), dtype=np.int)
        self.ret_buf = []  # returns
        self.success_buf = []
        self.cur_size, self.max_size = 0, size
        self.baseline = -100.  # return with SS model can be negative
        self.success_rate = 0.
        self.avg_return = 0.
        self.imitate_success_only = imitate_success_only

        # self.rgb_buf = []
        self.i_saved_traj = 0
        self.agent_model = agent_model

    # eval the trajectory performance and decide to store
    def eval_and_store(self, obs, act, ret, success, rgb=None, save_dir=None):
        '''
        store if success or episode return >= baseline
        if the buffer is full, first-in-first-out
        '''
        if self.cur_size > 0:
            self.baseline = np.mean(self.ret_buf) + 2 * np.std(self.ret_buf)
        if success or ((not self.imitate_success_only) and (ret >= self.baseline)):
            assert self.cur_size <= self.max_size
            self.obs_buf.append(obs)
            self.act_buf.append(act)
            self.ret_buf.append(ret)
            self.success_buf.append(success)
            self.success_rate = np.mean(self.success_buf)
            self.avg_return = np.mean(self.ret_buf)
            # self.rgb_buf.append(rgb)

            if self.cur_size < self.max_size:
                self.cur_size += 1
            else:  # FIFO
                del (self.obs_buf[0])
                del (self.act_buf[0])
                del (self.ret_buf[0])
                del (self.success_buf[0])
                # del(self.rgb_buf[0])

            # save the expert trajectory
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                pth = os.path.join(save_dir, 'traj_{}.pth'.format(self.i_saved_traj))
                pickle.dump([obs, act, ret, success, rgb], open(pth, 'wb'))
                self.i_saved_traj += 1

        # print(self.cur_size, len(self.obs_buf), self.baseline, self.success_rate,
        #    obs.shape, act.shape, ret)

    # get all the data for training. 
    # convert the trajectory list [N * [len, dim]] to transition array [N', dim]
    def get(self):
        assert self.cur_size > 0
        act_ = np.concatenate(self.act_buf)
        if self.agent_model == 'mineagent':
            obs_ = Batch.cat(self.obs_buf)
            rtn = {
                'act': torch.as_tensor(act_, dtype=torch.long),
                'obs': obs_
            }
        else:
            obs_ = np.concatenate(self.obs_buf)
            rtn = {
                'act': torch.as_tensor(act_, dtype=torch.long),
                'obs': torch.as_tensor(obs_, dtype=torch.float32)
            }
        return rtn


'''
PPO algorithm implementation:
for every epoch, first play the game to collect trajectories
until the buffer is full, then update the actor and the critic for sevaral steps using the buffer.
'''


def ppo_selfimitate_ss(args, ss_reward_model,seed=0, device=None,
                       steps_per_epoch=400, epochs=500, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4, vf_lr=1e-4,
                       train_pi_iters=80, train_v_iters=80, lam=0.95, max_ep_len=1000,
                       target_kl=0.01, save_freq=5, logger_kwargs=dict(), save_path='checkpoint',
                       agent_config_path='', n=0):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        device: cpu or cuda gpu device for training NN

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    # setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger_kwargs.update({'output_fname': args.exp_name})
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    # seed += 10000 * proc_id()
    torch.manual_seed(args.expseed)
    np.random.seed(args.expseed)

    # Instantiate environment
    env = MinecraftEnv(
        task_id=args.task,
        image_size=(160, 256),
        max_step=args.horizon,
        device=device,
        seed=seed,
        dense_reward=bool(args.use_dense),
        dis=args.dis,
        biome=args.biome,
    )
    obs_dim = env.observation_size
    agent_act_dim = len(args.actor_out_dim)
    print('Task prompt:', env.task_prompt)
    # logger.log('env: obs {}, act {}'.format(env.observation_space, env.action_space))

    # Create actor-critic agent
    if args.agent_model == 'mineagent':
        agent_config = utils.get_yaml_data(agent_config_path)
        feature_net_kwargs = agent_config['feature_net_kwargs']
        feature_net = {}
        for k, v in feature_net_kwargs.items():
            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(features, cls)
            feature_net[k] = cls(**v, device=device)
        feature_fusion_kwargs = agent_config['feature_fusion']
        feature_net = SimpleFeatureFusion(
            feature_net, **feature_fusion_kwargs, device=device
        )
        feature_net_v = copy.deepcopy(feature_net)  # actor and critic do not share
        actor = MultiCategoricalActor(
            feature_net,
            action_dim=args.actor_out_dim,  # [3, 3, 4, 25, 25, 8],
            device=device,
            **agent_config['actor'],
            activation='tanh',
        )
        critic = Critic(
            feature_net_v,
            action_dim=None,
            device=device,
            **agent_config['actor'],
            activation='tanh'
        )
        mine_agent = MineAgent(
            actor=actor,
            critic=critic,
            deterministic_eval=False
        ).to(device)  # use the same stochastic policy in training and test
        mine_agent.eval()
    elif args.agent_model == 'cnn':
        mine_agent = utils.CNNActorCritic(
            action_dim=args.actor_out_dim,
            deterministic_eval=False
        ).to(device)
        mine_agent.eval()
    else:
        raise NotImplementedError
    if n != 0:
        mine_agent.load_state_dict(torch.load(f'{save_path}/model_{n}.pth'))
    mine_agent.eval()

    # Sync params across processes
    # sync_params(ac)

    # Set up experience buffer
    buf = PPOBuffer(agent_act_dim, steps_per_epoch, gamma, lam, args.agent_model, obs_dim)

    # set up imitation buffer
    imitation_buf = SelfImitationBuffer(agent_act_dim, args.imitate_buf_size, args.imitate_success_only,
                                        args.agent_model)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, beta=0.1):
        obs, act, adv, logp_old = data['obs'], data['act'].to(device), data['adv'].to(device), data['logp'].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
        else:
            obs = obs.to(device)

        # Policy loss
        pi = mine_agent(obs).dist
        logp = pi.log_prob(act)
        # print('logp, logp_old = ', logp, logp_old)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() + beta*logp.mean()   

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret'].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
            obs_ = obs.obs
        else:
            obs_ = obs.to(device)
        return ((mine_agent.critic(obs_) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = torch.optim.Adam(mine_agent.actor.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(mine_agent.critic.parameters(), lr=vf_lr)

    # optimizer = torch.optim.Adam(mine_agent.parameters(), lr=lr)

    # a training epoch
    def update():
        mine_agent.train()

        data = buf.get()  # dict

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']  # mpi_avg(pi_info['kl'])
            # logger.log('kl={}'.format(kl))
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl, kl=%f.' % (i, kl))
                break
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # set up function for computing self-imitation loss
    # use the batch indexed by idxs in data
    def compute_loss_imitation(data, idxs):
        obs, act = data['obs'][idxs], data['act'][idxs].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
        else:
            obs = obs.to(device)
        pi = mine_agent(obs).dist
        loss_imitation = pi.imitation_loss(act)
        return loss_imitation

    # training step for self-imitation learning
    def update_imitation():
        mine_agent.train()
        data = imitation_buf.get()
        n_data = data['act'].shape[0]
        n_iter = max(int(n_data / args.imitate_batch_size), 1)  # iterations to train for 1 epoch
        # print('data', data, n_iter, n_data)
        for i in range(n_iter):
            pi_optimizer.zero_grad()
            idxs = np.random.randint(0, n_data, size=args.imitate_batch_size)
            # print('training batch', data['act'][idxs], data['obs'][idxs])
            loss_imitation = compute_loss_imitation(data, idxs)
            loss_imitation.backward()
            pi_optimizer.step()
        logger.store(LossImitation=loss_imitation.item(), NumItersImitation=n_iter)

    start_time = time.time()

    

    task_need_shears = ["harvest_1_tallgrass", 
                        "harvest_1_leaves",
                        "harvest_1_double_plant",
        ]
    
    task_need_sword = ["combat_cow_plains_diamond_armors_diamond_sword_shield",
                       "combat_sheep_plains_diamond_armors_diamond_sword_shield",
                       "combat_pig_plains_diamond_armors_diamond_sword_shield"
                       ]

    for epoch in range((n != 0) * (n + 1), n + epochs):

        # Save model and test

        
        logger.log('start epoch {}'.format(epoch))
        o, ep_ret, ep_len = env.reset(), 0, 0  # Prepare for interaction with environment
        ep_ret_ss, ep_success, ep_ret_dense = 0, 0, 0
        ep_rewards = []
        if args.agent_model == 'mineagent':
            # ep_state_embeddings = torch.as_tensor(o['rgb_emb']).squeeze(0)
        # else:
            ep_obs = torch.Tensor(np.asarray(o['rgb'], dtype=np.int).copy()).view(1, 1, *env.observation_size)
        rgb_list = []
        episode_in_epoch_cnt = 0

        if args.task in task_need_shears: 
            print("initialize the task {} with a shears".format(args.task))
            env.base_env.set_inventory([InventoryItem(slot="mainhand", name="shears", quantity=1)])
        if args.task in task_need_sword: 
            print("initialize the task {} with a diamond sword".format(args.task))
            env.base_env.set_inventory([InventoryItem(slot="mainhand", name="diamond_sword", quantity=1)])
            
        # rollout in the environment
        mine_agent.train()  # train mode to sample stochastic actions

        for t in range(steps_per_epoch):
            if args.save_raw_rgb:
                rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))

            if args.agent_model == 'mineagent':
                batch_o = preprocess_obs(o, device)
            else:
                batch_o = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1, *obs_dim)
                batch_o = torch.as_tensor(batch_o, dtype=torch.float32).to(device)
            batch_act = mine_agent.forward_actor_critic(batch_o)
            a, v, logp = batch_act.act, batch_act.val, batch_act.logp
            v = v[0]
            logp = logp[0]

            a_env = transform_action(a)
            next_o, r, d, _ = env.step(a_env)
            success = r

            r = r * args.reward_success
            ep_rewards.append(r)
            ep_obs = torch.cat((ep_obs, torch.Tensor(np.asarray(next_o['rgb'], dtype=np.uint8).copy()).view(1, 1,*env.observation_size)),1)

            ep_success += success
            if ep_success > 1: ep_success = 1
            ep_ret += r
            ep_len += 1

            # save and log
            if args.agent_model == 'mineagent':
                batch_o.to_numpy()  # less gpu mem
            else:
                batch_o = batch_o.cpu().numpy()
            buf.store(batch_o, a[0].cpu(), r, v,
                      logp)  # the stored reward will be modified at episode end, if use SS reward
            logger.store(VVals=v.detach().cpu().numpy())

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1

            # compute SS rewards for each step.
            # modify the trajectory rewards in the buffer
            if terminal or epoch_ended:
                
                if args.intri_type == 'pe':
                    ep_rewards_ss = ss_reward_model.cal_intrinsic_s2e(ep_obs/255.)[0]
                if args.intri_type == 'ds':
                    ep_rewards_ss = ss_reward_model.cal_intrinsic_s2e(ep_obs/255.)[1]
                if args.intri_type == 'ps':
                    ep_rewards_ss = ss_reward_model.cal_intrinsic_s2e(ep_obs/255.)[2]
                assert len(ep_rewards) == len(ep_rewards_ss)
                ep_rewards = args.env_reward * np.asarray(ep_rewards) + args.ss_coff * ep_rewards_ss
                # print the averaged ep_rewards_ss
                print("averaged ss rewards: ", np.mean(ep_rewards_ss))

                ep_ret_ss = np.sum(ep_rewards_ss)
                ep_ret += ep_ret_ss
                buf.modify_trajectory_rewards(ep_rewards)

                # check and add to imitation buffer if the trajectory ends
                if terminal:
                    if args.agent_model == 'mineagent':
                        obs_ = Batch.cat(buf.obs_buf[buf.path_start_idx: buf.ptr])
                    else:
                        obs_ = buf.obs_buf[buf.path_start_idx: buf.ptr].copy()
                    act_ = buf.act_buf[buf.path_start_idx: buf.ptr].copy()
                    if args.save_raw_rgb:
                        rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))
                        rgb_list = np.asarray(rgb_list)
                    imitation_buf.eval_and_store(obs_, act_, ep_ret_ss, int(ep_success), rgb_list, None)

                    # save the gif
                    if args.save_raw_rgb and ((epoch % save_freq == 0) or (epoch == epochs - 1)) and episode_in_epoch_cnt == 0:
                        pth = os.path.join(args.save_path, '{}_{}_ss{:.2f}_success{}.gif'.format(epoch, episode_in_epoch_cnt, float(ep_ret_ss), int(ep_success)))
                        imageio.mimsave(pth, [np.transpose(i_, [1,2,0]) for i_ in rgb_list], duration=0.1)
                        # logger.save_state({'env': env}, None)
                        if (epoch % 50) == 0:
                            torch.save(mine_agent.state_dict(), os.path.join(save_path, 'model_{}.pth'.format(epoch)))


                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    if args.agent_model == 'mineagent':
                        batch_o = preprocess_obs(o, device)
                    else:
                        batch_o = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1, *obs_dim)
                        batch_o = torch.as_tensor(batch_o, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        v = mine_agent.forward_actor_critic(batch_o).val
                    v = v[0].cpu().detach().numpy()
                else:
                    v = 0

                buf.finish_path(ep_rewards, v,args.lmbda)

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpRetSS=ep_ret_ss, EpSuccess=ep_success,
                                 EpRetDense=ep_ret_dense)

                o, ep_ret, ep_len = env.reset(), 0, 0
                env.base_env.clear_inventory()

                if args.task in task_need_shears: 
                    env.base_env.set_inventory([InventoryItem(slot="mainhand", name="shears", quantity=1)])
                if args.task in task_need_sword: 
                    env.base_env.set_inventory([InventoryItem(slot="mainhand", name="diamond_sword", quantity=1)])
                
                ep_ret_ss, ep_success, ep_ret_dense = 0, 0, 0
                ep_rewards = []
                ep_obs = torch.Tensor(np.asarray(o['rgb'], dtype=np.int).copy()).view(1, 1, *env.observation_size)
                rgb_list = []
                episode_in_epoch_cnt += 1


        # Perform PPO update!
        update()
        episode_in_epoch_cnt = 0

        # Perform self-imitation
        if imitation_buf.cur_size >= 1 and (epoch % args.imitate_freq == 0) and epoch > 0:
            for i_imitate in range(args.imitate_epoch):
                update_imitation()
            logger.store(ImitationBufferSuccess=imitation_buf.success_rate,
                         ImitationBufferReturn=imitation_buf.avg_return,
                         ImitationBufferAcceptReturn=imitation_buf.baseline,
                         ImitationBufferNumTraj=imitation_buf.cur_size)
            # Log info about imitation
            logger.log_tabular('LossImitation', average_only=True)
            logger.log_tabular('NumItersImitation', average_only=True)
            logger.log_tabular('ImitationBufferSuccess', average_only=True)
            logger.log_tabular('ImitationBufferReturn', average_only=True)
            logger.log_tabular('ImitationBufferAcceptReturn', average_only=True)
            logger.log_tabular('ImitationBufferNumTraj', average_only=True)
        elif epoch == 0:
            logger.log_tabular('LossImitation', 0)
            logger.log_tabular('NumItersImitation', 0)
            logger.log_tabular('ImitationBufferSuccess', 0)
            logger.log_tabular('ImitationBufferReturn', 0)
            logger.log_tabular('ImitationBufferAcceptReturn', 0)
            logger.log_tabular('ImitationBufferNumTraj', 0)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRetSS', with_min_and_max=True)
        logger.log_tabular('EpSuccess', with_min_and_max=True)
        logger.log_tabular('EpRetDense', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

        # to avoid destroying too many blocks, remake the environment

        if args.task in task_need_shears and (epoch % 10 == 0) and epoch>0:
            env.remake_env()
        elif (epoch % 50 == 0) and epoch > 0:
            env.remake_env()
            # save the imitation learning buffer
            # pth = os.path.join(save_path, 'buffer_{}.pth'.format(epoch))
            # pickle.dump(imitation_buf, open(pth, 'wb'))
