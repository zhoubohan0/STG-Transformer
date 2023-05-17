import argparse
import torch


def ppo_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default=f'BreakoutNoFrameskip-v4')
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
    parser.add_argument("--ds_coef", type=float, default=-1)
    parser.add_argument("--ps_coef", type=float, default=0.1)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--intrinsic", type=str, default='ds')
    parser.add_argument("--algo", type=str, default='STG', help="choose from [STG,STG-,ELE]")
    parser.add_argument("--env_reward", type=int, default=0)  # whether to use environmental rewards
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.intrinsic == 'ds' or args.intrinsic == 'ds_tanh':
        coeffs = f'{args.ds_coef}'
    if args.intrinsic == 'ps':
        coeffs = f'{args.ps_coef}'
    if args.intrinsic == 'dsps':
        coeffs = f'{args.ds_coef}_{args.ps_coef}'
    args.run_name = f"{args.algo}_{coeffs}{args.intrinsic}_lambda{args.gae_lambda}_{args.env_name}_seed{args.seed}"
    return args
