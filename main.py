# -*- coding: utf-8 -*-

# - Package Imports - #
from utils.args import get_args, post_process
from tide.exp_tide import ExpTIDEWorker
from tide.exp_init import ExpInitWorker
from tide.exp_oade import ExpOADEWorker
from cmp.asn.exp_asn import ExpASNWorker
from cmp.ctd.exp_ctd import ExpCtdWorker


# - Coding Part - #
def get_worker(args):
    worker_set = {
        'init': ExpInitWorker,
        'tide': ExpTIDEWorker,
        'oade': ExpOADEWorker,
        'asn': ExpASNWorker,
        'ctd': ExpCtdWorker,
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
