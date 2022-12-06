# -*- coding: utf-8 -*-

# - Package Imports - #
from utils.args import get_args, post_process
from tide.exp_tide import ExpTIDEWorker
from tide.exp_init import ExpInitWorker


# - Coding Part - #
def get_worker(args):
    worker_set = {
        'init': ExpInitWorker,
        'tide': ExpTIDEWorker,
    }
    assert args.argset in worker_set.keys()
    return worker_set[args.argset](args)


def main():
    args = get_args()
    post_process(args)

    worker = get_worker(args)
    worker.init_all()
    worker.do()


if __name__ == '__main__':
    main()
