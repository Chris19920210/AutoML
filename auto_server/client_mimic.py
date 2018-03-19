import _thread
import json
import sys

import lightgbm as lgb
from sklearn import datasets
from sklearn.cross_validation import train_test_split

import mq
from bayesian_research.utils import MyEncoder
"""
Mimic for client with multiple threads
"""


class Mimic:
    def __init__(self, files):
        self.mq = mq.JobMq()
        i = 0
        jobs = []
        for file in files:
            X, Y = datasets.load_svmlight_file(file)
            job_id = "job_" + str(i)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            init_param = {"id": job_id,
                          "operation": "exec",
                          "end_flag": "false",
                          "worker_id": "14507986789798",
                          "job_id": "100",
                          "algo_name": "adaboost",
                          "init": "true",
                          "params": {"num_leaves": 30, "n_estimators": 200,
                                     "learning_rate": 0.5, "reg_alpha": 0.5, "reg_lambda": 0.5},
                          "Eval": 0.5,
                          "param_space": [{"parameterName": "num_leaves",
                                           "type": "INTEGER",
                                           "minValue": 20, "maxValue": 50,
                                           "feasiblePoints": ""},
                                          {"parameterName": "n_estimators",
                                           "type": "INTEGER",
                                           "minValue": 100, "maxValue": 300,
                                           "feasiblePoints": ""},
                                          {"parameterName": "learning_rate",
                                           "type": "DOUBLE",
                                           "minValue": 0.1, "maxValue": 1,
                                           "feasiblePoints": ""},
                                          {"parameterName": "reg_alpha",
                                           "type": "DOUBLE",
                                           "minValue": 0, "maxValue": 1,
                                           "feasiblePoints": ""},
                                          {"parameterName": "reg_lambda",
                                           "type": "DOUBLE",
                                           "minValue": 0, "maxValue": 1,
                                           "feasiblePoints": ""}]
                          }
            self.mq.init_job(init_param)
            jobs.append(dict(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                id=job_id,
                params=dict(
                    num_leaves=30,
                    n_estimators=200,
                    learning_rate=0.2,
                    reg_alpha=0.5,
                    reg_lambda=0.5
                )
            ))
        self.jobs = jobs

    def on_request(self, ch, method, props, body):
        job_ = json.loads(body)
        job_id = job_['id']
        if job_["end_flag"] == "true":
            self.jobs[int(job_id.split('_')[1])] = True

        flag = True
        for each in self.jobs:
            if each is not True:
                flag = False

        if flag:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.mq.channel.stop_consuming()
        else:
            job = self.jobs[int(job_id.split('_')[1])]
            job["params"]["num_leaves"] = job_["params"]["num_leaves"]
            job["params"]["n_estimators"] = job_["params"]["n_estimators"]
            job["params"]["learning_rate"] = job_["params"]["learning_rate"]
            job["params"]["reg_alpha"] = job_["params"]["reg_alpha"]
            job["params"]["reg_lambda"] = job_["params"]["reg_lambda"]
            self.train_thread(job)
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def train_thread(self, job):
        _thread.start_new_thread(self.train, (job,))

    def train(self, job):
        model = lgb.LGBMClassifier(job["params"])
        model.fit(job["X_train"], job["y_train"],
                  eval_set=[(job["X_test"], job["y_test"])],
                  eval_metric='auc',
                  early_stopping_rounds=5,
                  verbose=False)
        loss = max(model.evals_result_.values())

        job = {
            "id": job["id"],
            "operation": "exec",
            "end_flag": "false",
            "worker_id": "14507986789798",
            "job_id": "100",
            "algo_name": "lightgbm",
            "init": "false",
            "params": job["params"],
            "Eval": -loss,
            "param_space": [{"parameterName": "num_leaves",
                             "type": "INTEGER",
                             "minValue": 20, "maxValue": 50,
                             "feasiblePoints": ""},
                            {"parameterName": "n_estimators",
                             "type": "INTEGER",
                             "minValue": 100, "maxValue": 300,
                             "feasiblePoints": ""},
                            {"parameterName": "learning_rate",
                             "type": "DOUBLE",
                             "minValue": 0.1, "maxValue": 1,
                             "feasiblePoints": ""},
                            {"parameterName": "reg_alpha",
                             "type": "DOUBLE",
                             "minValue": 0, "maxValue": 1,
                             "feasiblePoints": ""},
                            {"parameterName": "reg_lambda",
                             "type": "DOUBLE",
                             "minValue": 0, "maxValue": 1,
                             "feasiblePoints": ""}]
        }
        j = json.dumps(job, cls=MyEncoder)
        self.mq.publish_job(j)

    def start(self):
        self.mq.consume(self.on_request)

    def stop(self):
        self.mq.close()


if __name__ == '__main__':
    msgs = Mimic(sys.argv[1:])
    msgs.start()
    msgs.stop()
