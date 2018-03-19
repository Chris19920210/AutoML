import multiprocessing as mp
from utils_rpc import Record
from bandit.ucb2 import *
from bayesian_research.random_search import *
import numpy as np
import sys
"""
Test for multi-processes
"""


def evaluate(record, arm, y):
    record.bandit.update(arm, y)
    x = record.models[0].ask()
    record.models[0].tell(x, y)
    record.tune_times += 1
    return record


def process(records, times, process_id, lock):
    for i in range(times):
        index = np.random.randint(2)
        lock.acquire()
        arm = records[index].bandit.select_arm()
        y = np.random.random(1)[0]
        record = evaluate(records[index], arm, y)
        records[index] = record
        lock.release()
        print("index：%d, Process:%d, tune times %d" % (index, process_id, records[index].tune_times))
        print("y values %s" % (",".join(map(str, records[index].models[0].y))))


def main():
    manager = mp.Manager()
    records = manager.dict()
    num_leaves = Integer(20, 50)
    num_boost_round = Integer(100, 300)
    learning_rate = Real(0.1, 1)
    lambda_l1 = Real(0, 1)
    lambda_l2 = Real(0, 1)
    # Dimensions
    dimensions = [num_leaves, num_boost_round, learning_rate, lambda_l1, lambda_l2]
    model1 = RandomGridSearch(dimensions)
    bandit1 = UCB2(0.001, [], [])
    bandit1.initialize(2)
    model2 = RandomGridSearch(dimensions)
    bandit2 = UCB2(0.001, [], [])
    bandit2.initialize(2)
    records[0] = Record(0, [model1], bandit1)
    records[1] = Record(0, [model2], bandit2)
    lock = manager.Lock()

    workers = [mp.Process(target=process, args=(records, 10, i, lock)) for i in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()


if __name__ == '__main__':
    print(sys.path)
    main()
