import argparse

import torch

from solver.solver import Solver
from utils.tool import fix_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Multimodal Time Series Foundation Model")

    parser.add_argument("--description", type=str, default="ChronoSteer-base")
    parser.add_argument("--only_test", default=True, action="store_true")

    parser.add_argument("--train_data", type=str,
                        default="./dataset/ChronoSteer-100K/data/ChronoSteer-100K.json")
    parser.add_argument("--test_data", type=str,
                        default="./dataset/MTSFBench-300/data/rev-test-reply.json")

    parser.add_argument("--tsfm_path", type=str, default="amazon/chronos-bolt-base")
    parser.add_argument("--embed_path", type=str, default="BAAI/bge-m3")
    parser.add_argument("--save_path", type=str, default="./log/")

    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--series_dim", type=int, default=768)
    parser.add_argument("--context_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=1024)

    parser.add_argument("--pretrain_epoch", type=int, default=100)
    parser.add_argument("--pretrain_patience", type=int, default=10)
    parser.add_argument("--pretrain_batch_size", type=int, default=32)
    parser.add_argument("--pretrain_lr", type=float, default=0.001)

    parser.add_argument("--finetune_epoch", type=int, default=100)
    parser.add_argument("--finetune_patience", type=int, default=10)
    parser.add_argument("--finetune_batch_size", type=int, default=256)
    parser.add_argument("--finetune_lr", type=float, default=0.001)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print("\n=====================Args========================")
    print(args)
    print("=================================================\n")

    fix_seed(args.seed)

    args.setting = "{0}".format(
        args.description,
    )

    print(f">>>>>>>>  initing : {args.setting}  <<<<<<<<\n")
    solver = Solver(args)

    if not args.only_test:
        print(f"\n>>>>>>>>  pretraining : {args.setting}  <<<<<<<<\n")
        solver.pretrain(test=True)

        print(f"\n>>>>>>>>  finetuning : {args.setting}  <<<<<<<<\n")
        solver.finetune(test=True)

    print(f"\n>>>>>>>>  testing : {args.setting}  <<<<<<<<\n")
    solver.test()

    print("\n=================================================\n")
    print("Done!")
