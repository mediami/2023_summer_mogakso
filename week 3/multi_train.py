import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '9,'
import wandb
import random

import sys
import argparse

from pathlib import Path
from train import run
from src.setup import clear
from src.args import get_args_parser



# setting_dict = dict(
#     cifar10="data --dataset_type CIFAR10 --train-size 32 32 --random-crop-pad 4 --test-size 32 32 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --epoch 300 --optimizer sgd --nesterov --lr .5 --min-lr 5e-5 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 2 --pin-memory --amp --channels-last",
#     cifar100="data --dataset_type CIFAR100 --train-size 32 32 --random-crop-pad 4 --test-size 32 32 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --epoch 200 --optimizer sgd --nesterov --lr .5 --min-lr 5e-5 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 2 --pin-memory --amp --channels-last",
# )
setting_dict = dict(
    cifar10="data --dataset_type CIFAR10 --train-size 32 32 --random-crop-pad 4 --test-size 32 32 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --epoch 300 --optimizer sgd --nesterov --lr .5 --min-lr 5e-5 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 2 --pin-memory --amp --channels-last",
    cifar100="data --dataset_type CIFAR100 --train-size 32 32 --random-crop-pad 4 --test-size 32 32 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 --epoch 200 --optimizer sgd --nesterov --lr .5 --min-lr 5e-5 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 256 -j 2 --pin-memory --amp --channels-last",
)

def get_multi_args_parser():
    parser = argparse.ArgumentParser(description='pytorch-cifar-examples', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('setup', type=str, nargs='+', choices=setting_dict.keys(), help='experiment setup')
    parser.add_argument('-m', '--model-name', type=str, nargs='+', default=['resnet50'], help='list of model names')
    parser.add_argument('-c', '--cuda', type=str, default='0', help='cuda device')
    parser.add_argument('-o', '--output-dir', type=str, default='log', help='log dir')
    parser.add_argument('-p', '--project-name', type=str, default='classification-tutorial', help='project name used for wandb logger')
    parser.add_argument('-w', '--who', type=str, default='your-name', help='enter your name')
    parser.add_argument('-r', '--resume', type=str, default=None, help='resume training from the specified checkpoint')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='use wandb')
    parser.add_argument('-s', '--save-checkpoint', action='store_true', default=False, help='save checkpoint')
    return parser

def pass_required_variable_from_previous_args(args, prev_args=None):
    if prev_args:
        required_vars = ['gpu', 'world_size', 'distributed', 'is_rank_zero', 'device']
        for var in required_vars:
            exec(f"args.{var} = prev_args.{var}")


def save_arguments(args, is_master):
    if is_master:
        print("Multiple Train Setting")
        print(f" - model (num={len(args.model_name)}): {', '.join(args.model_name)}")
        print(f" - setting (num={len(args.setup)}): {', '.join(args.setup)}")
        print(f" - cuda: {args.cuda}")
        print(f" - output dir: {args.output_dir}")

        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(args.output_dir, 'last_multi_args.txt'), 'wt') as f:
            f.write(" ".join(sys.argv))


if __name__ == '__main__':
    is_master = os.environ.get('LOCAL_RANK', None) is None or int(os.environ['LOCAL_RANK']) == 0
    multi_args_parser = get_multi_args_parser()
    multi_args = multi_args_parser.parse_args()
    save_arguments(multi_args, is_master)
    prev_args = None

    for setup in multi_args.setup:
        args_parser = get_args_parser()
        args = args_parser.parse_args(setting_dict[setup].split(' '))
        pass_required_variable_from_previous_args(args, prev_args)
        for model_name in multi_args.model_name:
            args.setup = setup
            args.exp_name = f"{model_name}_{setup}"
            args.model_name = model_name
            for option_name in ['cuda', 'project_name', 'who', 'use_wandb', 'resume', 'save_checkpoint']:
                exec(f"args.{option_name} = multi_args.{option_name}")
            run(args)
            clear(args)
        prev_args = args