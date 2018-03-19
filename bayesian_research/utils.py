from sklearn.datasets import load_svmlight_file
import numpy as np
import json


def get_data(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def logit_loss(y_test, y_pred):
    return np.array(list(map(lambda y_y_hat: -np.log(y_y_hat[1]) if y_y_hat[0] > 0 else -np.log(1.0 - y_y_hat[1]),
                             zip(y_test, y_pred)))).mean()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
