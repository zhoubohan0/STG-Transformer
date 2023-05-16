import argparse,os,torch

def sac_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=666, help="seed of the experiment")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.run_name = f'SAC_{args.env_id}_seed{args.seed}'

    return args


