# -*- coding: utf-8 -*-

# - Package Imports - #
import json

from utils.args import parse_args
from tide.exp_tide import ExpTIDEWorker
from tide.exp_init import ExpInitWorker


# - Coding Part - #
def get_worker(args):
    if args.argset == 'Init':
        return ExpInitWorker(args)
    elif args.argset == 'TIDE':
        return ExpTIDEWorker(args)
    else:
        raise RuntimeError(f'Invalid argset {args.argset}')


def save_cmd_args(out_dir, args):
    dict_save = {}
    for key, value in args.__dict__.items():
        if key in ['train_dir', 'out_dir', 'test_dir']:
            dict_save[key] = str(value)
        elif key in ['device', 'data_parallel']:
            continue
        else:
            dict_save[key] = value

    with open(str(out_dir / f'params.json'), 'w+') as file:
        json.dump(dict_save, file, indent=2)


def main():
    args = parse_args()

    worker = get_worker(args)
    worker.set_path(data_dir=args.train_dir,
                    res_dir=args.out_dir,
                    test_dir=args.test_dir)
    save_cmd_args(worker.res_dir, args)

    worker.init_dataset()
    worker.init_networks()
    worker.init_losses()
    worker.init_optimizers()

    worker.do()


if __name__ == '__main__':
    main()
