# -*- coding: utf-8 -*-
# @Description:
#   argparse for program.

# - Package Imports - #
import argparse
import torch
from torch.distributed import init_process_group
import os
from datetime import datetime
from pathlib import Path


# - Coding Part - #
def _str2bool(input_str):
    if input_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Unsupported value encounter: {input_str}')


def _str2path(input_str):
    if input_str is None:
        return None
    return Path(input_str)


def parse_args():
    """Functions for arguments setting."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--argset',
                        help='Decide the worker for experiment.',
                        choices=['TIDE', 'Init'],
                        required=True,
                        type=str)
    parser.add_argument('--train_dir',
                        help='Folder for training data',
                        default=None,
                        type=_str2path)
    parser.add_argument('--out_dir',
                        help='The output directory',
                        default='./output',
                        type=_str2path)
    parser.add_argument('--test_dir',
                        help='Folder for evaluation',
                        default=None,
                        type=_str2path)
    parser.add_argument('--exp_type',
                        help='Training / testing / online testing.',
                        default='train', choices=['train', 'eval', 'online'], type=str)
    parser.add_argument('--loss_type',
                        help='Loss type used for training.',
                        default='pf', choices=['su', 'ph', 'pf'], type=str)
    parser.add_argument('--run_tag',
                        help='For output directory naming. Use timestamp by default.',
                        default='[Time]', type=str)

    # Debug mode. Useful for checking your program going well.
    parser.add_argument('--debug_mode',
                        help='Check program is correct or not.',
                        default='False', type=_str2bool)

    # For distributed
    parser.add_argument('--local_rank',
                        help='Node rank for distributed training.',
                        type=int, default=0)

    # Parameters for training
    parser.add_argument('--clip_len',
                        help='Clip length for training.',
                        default=8, type=int)
    parser.add_argument('--batch_num',
                        help='Batch size for training.',
                        default=1, type=int)
    parser.add_argument('--num_workers',
                        help='Workers for loading data. Will be set to 0 if under win systems.',
                        default=4, type=int)
    parser.add_argument('--epoch_start',
                        help='Start epoch for training. If > 0, pre-trained model is needed.',
                        default=0, type=int)
    parser.add_argument('--epoch_end',
                        help='End epoch for training.',
                        default=64, type=int)
    parser.add_argument('--lr',
                        help='Learning rate for model.',
                        default=1e-4, type=float)
    parser.add_argument('--lr_step',
                        help='Learning-rate decay step. 0 for no decay. '
                             'When negative, lr will increase for lr-searching.',
                        default=0, type=int)

    # For visualization & report
    parser.add_argument('--report_stone',
                        help='How many iters for real-time report to be printed.',
                        default=32, type=int)
    parser.add_argument('--img_stone',
                        help='How many iters for img visualization.',
                        default=256, type=int)

    # For saving
    parser.add_argument('--save_res',
                        help='Output result or not.',
                        default='False', type=_str2bool)
    parser.add_argument('--save_stone',
                        help='How many epochs for saving result. 0 for no saving.',
                        default=0, type=int)
    parser.add_argument('--remove_history',
                        help='Save old model.pt or not. Useful when your model is very large.',
                        default='False', type=_str2bool)

    args = parser.parse_args()

    #
    # Post processing
    #

    # Timestamp for run_tag
    if args.run_tag == '[Time]':
        args.run_tag = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())

    # Small fix for windows: num_worker -> 0
    if os.name == 'nt':
        args.num_workers = 0

    # For distributed when you have multiple GPU cards. Only works for linux.
    torch.cuda.set_device(args.local_rank)
    if os.name == 'nt':
        args.data_parallel = False
    else:
        init_process_group('nccl', init_method='env://')
        args.data_parallel = True
    args.device = torch.device(f'cuda:{args.local_rank}')

    return args
