import argparse
import torch


def pretrain_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='STG', help="choose from [STG,STG-,ELE]")
    parser.add_argument("--game", type=str, default='Breakout')
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--l2_coff", type=float, default=0.5)
    parser.add_argument("--g_coff", type=float, default=0.3)
    parser.add_argument("--d_coff", type=float, default=0.5)
    parser.add_argument("--tdr_coff", type=float, default=0.1)

    parser.add_argument("--src_root", type=str, default='')
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not args.src_root:
        args.src_root = f'dataset/{args.game}'
    if args.algo in ['STG-', 'stg-']:
        args.tdr_coff = 0
    if args.algo in ['ELE', 'ele']:
        args.l2_coff = args.g_coff = args.d_coff = 0
    return args
