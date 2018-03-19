import argparse
import os
from skopt import Optimizer
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from multiprocessing import Queue, Process
import itertools
from utils import *
from random_search import *
from bandit.ucb2 import *
from bandit.exp3 import *

parser = argparse.ArgumentParser(description="Method Comparison")
parser.add_argument("--train-set", type=str, required=True, help="path to train set")
parser.add_argument("--test-set", type=str, required=True, help="path to test set")
parser.add_argument("--output", type=str, required=True, help="path to output")
parser.add_argument("--times", type=int, default=50, help="num of tuning")
parser.add_argument("--num-processes", type=int, default=4, help="num of process")
parser.add_argument("--method", type=str, choices=["Bayesian", "Random", "BanditUCB2", "BanditEXP3"],
                    default="Bayesian",
                    help="Tuning Method")
args = parser.parse_args()


def async_func(queue):
    while True:
        if not queue.empty():
            num_leaves = Integer(20, 50)
            num_boost_round = Integer(100, 300)
            learning_rate = Real(0.1, 1)
            lambda_l1 = Real(0, 1)
            lambda_l2 = Real(0, 1)
            # Dimensions
            dimensions = [num_leaves, num_boost_round, learning_rate, lambda_l1, lambda_l2]
            # test_set name
            each = queue.get()
            # different methods
            base_estimator = ["GP", "RF", "ET", "GBRT"]
            acq_func = ["LCB", "EI", "PI", "gp_hedge"]
            with open(os.path.join(args.output, each[:each.find(".t")]), "w") as f:
                train_path = os.path.join(args.train_set, each[:each.find(".t")])
                test_path = os.path.join(args.test_set, each)
                X_train = lgb.Dataset(train_path)
                X_test, y_test = get_data(test_path)
                for (base, acq) in itertools.product(base_estimator, acq_func):
                    # Optimizer
                    opt = Optimizer(dimensions=dimensions, random_state=42, acq_func=acq, base_estimator=base)
                    best = dict()
                    best['eval'] = float('-inf')
                    auc_trace = []
                    for i in range(args.times):
                        params_v = opt.ask()
                        params = {'num_leaves': params_v[0],
                                  'num_boost_round': params_v[1],
                                  'learning_rate': params_v[2],
                                  'lambda_l1': params_v[3],
                                  'lambda_l2': params_v[4],
                                  'objective': 'binary',
                                  'num_thread': 1
                        }
                        clf = lgb.train(params, X_train)
                        y_pred = clf.predict(X_test)
                        auc = roc_auc_score(y_test > 0, y_pred)
                        auc_trace.append(auc)
                        opt.tell(params_v, -auc)
                        if auc > best['eval']:
                            best['eval'] = auc
                            best['params'] = params
                    record = dict()
                    record["base_estimator"] = base
                    record["acq_func"] = acq
                    record["best"] = best
                    record["auc_trace"] = auc_trace
                    json_string = json.dumps(record, cls=MyEncoder)
                    f.write(json_string)
                    f.write("\n")
        else:
            break


def async_func_rand(queue):
    while True:
        if not queue.empty():
            num_leaves = Integer(20, 50)
            num_boost_round = Integer(100, 300)
            learning_rate = Real(0.1, 1)
            lambda_l1 = Real(0, 1)
            lambda_l2 = Real(0, 1)
            # Dimensions
            dimensions = [num_leaves, num_boost_round, learning_rate, lambda_l1, lambda_l2]
            # test_set name
            each = queue.get()
            # different methods
            with open(os.path.join(args.output, each[:each.find(".t")]), "w") as f:
                train_path = os.path.join(args.train_set, each[:each.find(".t")])
                test_path = os.path.join(args.test_set, each)
                X_train = lgb.Dataset(train_path)
                X_test, y_test = get_data(test_path)
                # Optimizer
                opt = RandomGridSearch(dimensions=dimensions)
                best = dict()
                best['eval'] = float('-inf')
                auc_trace = []
                for i in range(args.times):
                    params_v = opt.ask()
                    params = {'num_leaves': params_v[0],
                              'num_boost_round': params_v[1],
                              'learning_rate': params_v[2],
                              'lambda_l1': params_v[3],
                              'lambda_l2': params_v[4],
                              'objective': 'binary',
                              'num_thread': 1
                              }
                    clf = lgb.train(params, X_train)
                    y_pred = clf.predict(X_test)
                    auc = roc_auc_score(y_test > 0, y_pred)
                    auc_trace.append(auc)
                    if auc > best['eval']:
                        best['eval'] = auc
                        best['params'] = params
                record = dict()
                record["best"] = best
                record["auc_trace"] = auc_trace
                json_string = json.dumps(record, cls=MyEncoder)
                f.write(json_string)
                f.write("\n")
        else:
            break


def async_func_bandit(queue):
    while True:
        if not queue.empty():
            num_leaves = Integer(20, 50)
            num_boost_round = Integer(100, 300)
            learning_rate = Real(0.1, 1)
            lambda_l1 = Real(0, 1)
            lambda_l2 = Real(0, 1)
            # Dimensions
            dimensions = [num_leaves, num_boost_round, learning_rate, lambda_l1, lambda_l2]
            # test_set name
            each = queue.get()
            # different methods
            with open(os.path.join(args.output, each[:each.find(".t")]), "w") as f:
                count = {0: 0,
                         1: 0}
                train_path = os.path.join(args.train_set, each[:each.find(".t")])
                test_path = os.path.join(args.test_set, each)
                X_train = lgb.Dataset(train_path)
                X_test, y_test = get_data(test_path)
                # Optimizer
                opt_bayesian = Optimizer(dimensions=dimensions)
                opt_random = RandomGridSearch(dimensions=dimensions)
                best = dict()
                best['eval'] = float('-inf')
                auc_trace = []
                if args.method == "BanditUCB2":
                    bandit = UCB2(0.001, [], [])
                else:
                    bandit = Exp3(0.01, [])
                bandit.initialize(2)
                for i in range(args.times):
                    index = bandit.select_arm()
                    count[index] += 1
                    if index == 0:
                        params_v = opt_bayesian.ask()
                    else:
                        params_v = opt_random.ask()
                    params = {'num_leaves': params_v[0],
                              'num_boost_round': params_v[1],
                              'learning_rate': params_v[2],
                              'lambda_l1': params_v[3],
                              'lambda_l2': params_v[4],
                              'objective': 'binary',
                              'num_thread': 1
                              }
                    clf = lgb.train(params, X_train)
                    y_pred = clf.predict(X_test)
                    auc = roc_auc_score(y_test > 0, y_pred)
                    auc_trace.append(auc)
                    opt_bayesian.tell(params_v, -auc)
                    bandit.update(index, auc)
                    if auc > best['eval']:
                        best['eval'] = auc
                        best['params'] = params
                record = dict()
                record["best"] = best
                record["auc_trace"] = auc_trace
                record["bayesian"] = count[0]
                record["random"] = count[1]
                json_string = json.dumps(record, cls=MyEncoder)
                f.write(json_string)
                f.write("\n")
        else:
            break


def main():
    queue = Queue()
    test_set = os.listdir(args.test_set)
    for each in test_set:
        queue.put(each)

    if args.method == "Bayesian":
        workers = [Process(target=async_func, args=(queue,)) for _ in range(args.num_processes)]
    elif args.method == "Random":
        workers = [Process(target=async_func_rand, args=(queue,)) for _ in range(args.num_processes)]
    else:
        workers = [Process(target=async_func_bandit, args=(queue,)) for _ in range(args.num_processes)]

    for w in workers:
        w.start()
    for w in workers:
        w.join()


if __name__ == '__main__':
    main()
