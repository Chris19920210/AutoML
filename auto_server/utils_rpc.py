import json
import logging
import sys

import numpy as np
from skopt import Optimizer
from skopt.space import Real, Categorical, Integer

from bandit.exp3 import *
from bandit.ucb2 import *
from bayesian_research.random_search import RandomGridSearch
from bayesian_research.utils import MyEncoder

"""
Utils function for rpc server
"""


class Record(object):
    def __init__(self, tune_times, models, arm=0, bandit=None):
        self.tune_times = tune_times
        self.models = models
        self.bandit = bandit
        self.arm = arm


def model_constructor(d, base_estimator, acq_func):
    dimensions = []
    for var in sorted(d["param_space"], key=lambda x: x["parameterName"]):
        if var["type"] == "CATEGORICAL":
            dimensions.append(Categorical(var["feasiblePoints"]))
        elif var["type"] == "DOUBLE":
            dimensions.append(Real(var["minValue"], var["maxValue"]))
        else:
            dimensions.append(Integer(var["minValue"], var["maxValue"]))
    return Optimizer(dimensions=dimensions,
                     random_state=1, base_estimator=base_estimator, acq_func=acq_func), RandomGridSearch(dimensions)


def data_generator(d):
    y = d['Eval']
    print("recv =====> ", json.dumps(d, ensure_ascii=False))
    x = [value for key, value in sorted(d["params"].items(), key=lambda item: item[0])]
    return x, y


def evaluate(x, y, record):
    if record.bandit is not None:
        record.bandit.update(record.arm, -y)
        index = record.bandit.select_arm()
        record.arm = index
    else:
        index = np.random.randint(len(record.models))
    record.tune_times += 1
    for model in record.models:
        model.tell(x, y)
    return record.models[index].ask(), record


def rpc_callback(d, **kwargs):
    if d['init'] == 'false' and not d["id"] in kwargs["records"]:
        d['end_flag'] = 'true'
        logging.info("%s has already done!" % d["id"])
        return d

    if d['end_flag'] == 'true' and d["id"] in kwargs["records"]:
        del kwargs["records"][d['id']]
        logging.info("%s done! by console" % d["id"])
        return d

    if d['init'] == "true":
        if kwargs["bandit"] is not None:
            baysian_model, random_model = model_constructor(d, kwargs["base_estimator"], kwargs["acq_func"])
            if kwargs["bandit"] == "BanditUCB2":
                bandit = UCB2(0.001, [], [])
                bandit.initialize(2)
                arm = bandit.select_arm()
            else:
                bandit = Exp3(0.01, [])
                bandit.initialize(2)
                arm = bandit.select_arm()
            kwargs["records"][d["id"]] = Record(0, [baysian_model, random_model], arm, bandit)
        else:
            baysian_model, _ = model_constructor(d, kwargs["base_estimator"], kwargs["acq_func"])
            kwargs["records"][d["id"]] = Record(0, [baysian_model])
        d['init'] = "false"
        print("%s job start" % d["id"])
        return d
    x = list()
    if d["id"] in kwargs["records"]:
        param, y = data_generator(d)
        x, record = evaluate(param, y, kwargs["records"][d["id"]])
        kwargs["records"][d["id"]] = record
        print("Tuning times: %d / %d" % (kwargs["records"][d["id"]].tune_times, kwargs['tune_times']))
    else:
        logging.error("Some error happen with id %s" % d["id"])

    if len(x) == 0:
        logging.error("%s some error happens!" % d['id'])
    params_name = list(map(lambda z: z["parameterName"], sorted(d['param_space'], key=lambda z: z["parameterName"])))
    params = {k: v for k, v in zip(params_name, x)}
    d["params"] = params

    if kwargs["records"][d["id"]].tune_times >= kwargs['tune_times']:
        d["end_flag"] = "true"
        logging.info("%s has already reach the optimition upper limit, done!" % d["id"])
        del kwargs["records"][d["id"]]

    print("return ======> ", json.dumps(d, ensure_ascii=False, cls=MyEncoder))

    return d


if __name__ == '__main__':
    print(sys.path)
