import argparse
import configparser
import multiprocessing as mp

from rpc import RpcServer
from utils_rpc import *

"""
The main function launching rpc server
"""

argparser = argparse.ArgumentParser(description='configuration setting')
argparser.add_argument('--basic-config', type=str, default='./config.properties',
                       help='Path to Basic Configuration for RabbitMQ')
argparser.add_argument('--processes', type=int, default=4,
                       help='Num of Processes')
argparser.add_argument("--tune-times", type=int, default=5,
                       help="Num of hyperparameter tuning")
argparser.add_argument("--bandit", choices=["BanditUCB2", "BanditEXP3"], type=str, default=None,
                       help="bandit")
argparser.add_argument("--base-estimator", choices=["GP", "RF", "ET", "GBRT"], type=str, default="GP",
                       help="base estimator for bayesian")
argparser.add_argument("--acq-func", choices=["LCB", "EI", "PI", "gp_hedge"], type=str, default="gp_hedge",
                       help="Acquisition function")
args = argparser.parse_args()


def rpc_process(queue):
    conf = configparser.RawConfigParser()
    conf.read(args.basic_config)
    rpc = RpcServer(conf)
    rpc.server(rpc_callback, records=queue, tune_times=args.tune_times, base_estimator=args.base_estimator,
               acq_func=args.acq_func, bandit=args.bandit)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='myapp.log',
                        filemode='w')
    manager = mp.Manager()
    records = manager.dict()
    cpus = mp.cpu_count()

    if cpus - 1 <= args.processes:
        processes = cpus - 1
    else:
        processes = args.processes

    workers = [mp.Process(target=rpc_process, args=(records,)) for _ in range(processes)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
