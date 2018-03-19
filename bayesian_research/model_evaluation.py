import matplotlib.pyplot as plt
import argparse
import os
import json
import numpy as np


parser = argparse.ArgumentParser(description="Method Evaluation")
parser.add_argument("--bayesian", type=str, required=True, help="path to bayesian")
parser.add_argument("--random", type=str, required=True, help="path to random")
parser.add_argument("--output", type=str, required=True, help="path to bayesian output")
parser.add_argument("--banditUCB", type=str, default=None, help="path to ucb")
parser.add_argument("--banditEXP", type=str, default=None, help="path to exp")
args = parser.parse_args()


def plot(file):
    bayesian_path = os.path.join(args.bayesian, file)
    random_path = os.path.join(args.random, file)
    plt.figure(figsize=(8, 4))
    considerable = [("ET", "EI"), ('GBRT', 'EI'), ('GP', 'gp_hedge'), ('RF', 'gp_hedge')]
    colors = ["red", "green", "blue", "yellow"]
    x = np.array(range(1, 201))
    with open(bayesian_path, "r") as f:
        for each in f.readlines():
            record = json.loads(each)
            if (record["base_estimator"], record["acq_func"]) in considerable:
                y = []
                max_now = record["auc_trace"][0]
                for i in range(200):
                    max_now = max(record["auc_trace"][i], max_now)
                    y.append(max_now)
                plt.plot(x, y, label="%s, %s" % (record["base_estimator"], record["acq_func"]),
                         color=colors[considerable.index((record["base_estimator"], record["acq_func"]))], linewidth=1)

    with open(random_path, "r") as f:
        for each in f.readlines():
            record = json.loads(each)
            y = []
            max_now = record["auc_trace"][0]
            for i in range(200):
                max_now = max(record["auc_trace"][i], max_now)
                y.append(max_now)
            plt.plot(x, y, label="Random with uniform prior", color="black", linewidth=1)

    plt.xlabel("Tuning Times")
    plt.ylabel("AUC")
    plt.ylim(0.4, 1)
    plt.legend()
    out_ = os.path.join(args.output, file)
    plt.savefig(out_)


def plot_bandit(file):
    bayesian_path = os.path.join(args.bayesian, file)
    random_path = os.path.join(args.random, file)

    plt.figure(figsize=(8, 4))
    considerable = [('GP', 'gp_hedge')]
    colors = ["blue"]
    x = np.array(range(1, 201))
    with open(bayesian_path, "r") as f:
        for each in f.readlines():
            record = json.loads(each)
            if (record["base_estimator"], record["acq_func"]) in considerable:
                y = []
                max_now = record["auc_trace"][0]
                for i in range(200):
                    max_now = max(record["auc_trace"][i], max_now)
                    y.append(max_now)
                plt.plot(x, y, label="%s, %s" % (record["base_estimator"], record["acq_func"]),
                         color=colors[considerable.index((record["base_estimator"], record["acq_func"]))], linewidth=1)

    with open(random_path, "r") as f:
        for each in f.readlines():
            record = json.loads(each)
            y = []
            max_now = record["auc_trace"][0]
            for i in range(200):
                max_now = max(record["auc_trace"][i], max_now)
                y.append(max_now)
            plt.plot(x, y, label="Random with uniform prior", color="black", linewidth=1)

    if args.banditUCB is not None:
        ucb_path = os.path.join(args.banditUCB, file)
        with open(ucb_path, "r") as f:
            for each in f.readlines():
                record = json.loads(each)
                y = []
                max_now = record["auc_trace"][0]
                for i in range(200):
                    max_now = max(record["auc_trace"][i], max_now)
                    y.append(max_now)
                plt.plot(x, y, label="UCB2", color="orange", linewidth=1)

    if args.banditEXP is not None:
        exp_path = os.path.join(args.banditEXP, file)
        with open(exp_path, "r") as f:
            for each in f.readlines():
                record = json.loads(each)
                y = []
                max_now = record["auc_trace"][0]
                for i in range(200):
                    max_now = max(record["auc_trace"][i], max_now)
                    y.append(max_now)
                plt.plot(x, y, label="EXP3", color="red", linewidth=1)

    plt.xlabel("Tuning Times")
    plt.ylabel("AUC")
    plt.ylim(0.4, 1)
    plt.legend()
    out_ = os.path.join(args.output, file)
    plt.savefig(out_)


def main():
    if args.banditEXP is not None or args.banditUCB is not None:
        for each in os.listdir(args.bayesian):
            plot_bandit(each)
    else:
        for each in os.listdir(args.bayesian):
            plot(each)


if __name__ == '__main__':
    main()
