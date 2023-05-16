import os
from glob import glob
from args import pretrain_parser
from network import ELE,STGTransformer
from config import ELEConfig,STGConfig
from utils import setseed


if __name__ == '__main__':
    args = pretrain_parser()

    game, seed, device = args.game, args.seed, args.device
    setseed(seed)

    src = glob(os.path.join(args.src_root, '*.pkl'))

    if args.algo in ['STG', 'stg','STG-', 'stg-']:
        stgconfig = STGConfig(l2_coff=args.l2_coff,d_coff=args.d_coff, g_coff=args.g_coff, tdr_coff=args.tdr_coff,device=device)
        model = STGTransformer(stgconfig).to(device)
        model.trainSTG(src)

    if args.algo in ['ELE', 'ele']:
        eleconfig = ELEConfig(tdr_coff=args.tdr_coff,device=device)
        model = ELE(eleconfig).to(device)
        model.trainELE(src)

