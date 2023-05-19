import argparse
#from spinup_utils.mpi_tools import mpi_fork
import os

import torch
import sys

from mineagent.common.run_utils import setup_logger_kwargs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99) # discount
    parser.add_argument('--target-kl', type=float, default=0.5) # kl upper bound for updating policy
    parser.add_argument('--seed', '-s', type=int, default=7) # random seed for both np, torch and env
    parser.add_argument('--cpu', type=int, default=1) # number of workers
    parser.add_argument('--gpu', default='0') # -1 if use cpu, otherwise select the gpu id
    parser.add_argument('--steps', type=int, required=True) # sample steps per epoch (buffer size * workers)
    parser.add_argument('--epochs', type=int, default=1000) # epoch number
    parser.add_argument('--save-path', type=str, default='checkpoint') # model save path
    parser.add_argument('--exp-name', type=str, required=True) # log name
    parser.add_argument('--task', type=str,required=True) # task_id
    parser.add_argument('--horizon', type=int, required=True) # task horizon. 500 in the current released code

    # CLIP model and agent model config
    parser.add_argument('--clip-config-path', type=str, default='mineagent/official/config.yml')
    parser.add_argument('--clip-model-path', type=str, default='mineagent/official/attn.pth')
    parser.add_argument('--agent-model', type=str, default='mineagent') # agent architecture: mineagent, cnn
    parser.add_argument('--agent-config-path', type=str, default='mineagent/conf.yaml') # for mineagent

    # reward weights
    parser.add_argument('--reward-success', type=float, default=200.)
    parser.add_argument('--reward-clip', type=float, default=1.)
    parser.add_argument('--ss_coff', type=float, default=-1.0)
    parser.add_argument('--clip-reward-mode', type=str, default='direct') # how to compute clip reward
    parser.add_argument('--reward-step', type=float, default=-1.) # per-step penalty
    parser.add_argument('--use-dense', type=int, default=0) # use dense reward
    parser.add_argument('--reward-dense', type=float, default=1.) # dense reward weight

    parser.add_argument('--actor-out-dim', type=int, nargs='+', default=[12,3])
    # actor output dimensions. mineagent official: [3,3,4,25,25,8]; my initial implement: [56,3]
    # mineagent with clipped camera space: [3,3,4,5,3] or [12,3]
    # should modify transform_action() in minecraft.py together with this arg

    # self-imitation learning
    parser.add_argument('--imitate-buf-size', type=int, default=500) # max num of traj to store
    parser.add_argument('--imitate-batch-size', type=int, default=1000) # batchsize for imitation learning
    parser.add_argument('--imitate-freq', type=int, default=100) # how many ppo epochs to run self-imitation
    parser.add_argument('--imitate-epoch', type=int, default=1) # how many self-imitation epochs
    parser.add_argument('--imitate-success-only', type=int, default=1) # save only success trajs into imitation buffer
    parser.add_argument('--add-ss-in-imitation', type=int, default=0)
    
    # arguments for other research works
    parser.add_argument('--save-raw-rgb', type=int, default=1) # save embeddings or rgb for Bohan's work?
    parser.add_argument('--use-ss-reward', type=int, default=1) # experiment for pretrained SS-transformer
    parser.add_argument('--ss-k', type=int, default=10) # prediction horizon for SS transformer
    parser.add_argument('--ss_model_path', type=str, required=True) # pretrained SS model path
    
    parser.add_argument('--continue-n', type=int, default=0)
    parser.add_argument('--env_reward', type=int, default=0)
    parser.add_argument('--intri_type', type=str, required=True)
    parser.add_argument('--dis', type=int, default=4)
    parser.add_argument('--expseed', type=int, default=0)
    parser.add_argument('--only-clip', type=int, default=0)
    parser.add_argument('--lmbda', type=float, default=0.8)
    parser.add_argument('--biome', type=str, default='plains')
    parser.add_argument('--stg', type=int, required=True)
    parser.add_argument('--algo', type=str, required=True)
    args = parser.parse_args()
    #print(args)
    
    
    task_abbr = [each for each in ['milk','wool','leaves', 'tallgrass','double_plant', 'combat_cow', 'combat_sheep', 'combat_pig'] if each in args.task][0]
    args.exp_name = f"{args.exp_name}_{task_abbr}_{'' if not args.env_reward else 're'}{args.ss_coff}{args.intri_type}_lbmda{args.lmbda}_seed{args.expseed}"
    if not os.path.exists(args.save_path): os.mkdir(args.save_path)
    args.save_path = os.path.join(args.save_path, args.exp_name)
    if not os.path.exists(args.save_path): os.mkdir(args.save_path)
    
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # set gpu device
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    print('Using device:', device)

    if args.algo in ['STG','stg']:
        from STGTrain.STG4MC import *
    if args.algo in ['ELE','ele']:
        from STGTrain.ELE4MC import *
    # initialize the SS reward model
    if args.stg:
        ss_reward_model = SSTransformer(Config()).to(device)
        ss_reward_model.load_state_dict(torch.load(args.ss_model_path))


    # from ppo_selfimitate_ss_ import ppo_selfimitate_ss
    from ppo_intrinsic_ss_mc import ppo_selfimitate_ss
    print('STG: Training ppo_selfimitate_SS.')
    ppo_selfimitate_ss(args, ss_reward_model=ss_reward_model,
        gamma=args.gamma, save_path=args.save_path, target_kl=args.target_kl,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, device=device, 
        clip_config_path=args.clip_config_path, clip_model_path=args.clip_model_path,  # mineclip_official/config.yml和adjust.pth
        agent_config_path=args.agent_config_path,n=args.continue_n)  # mineagent/conf.yaml