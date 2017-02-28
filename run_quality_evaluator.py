#!/usr/bin/env python
import argparse
import json
import os.path

import errno
import numpy as np

from help_scripts.simulation.simulator import SeriesSampler

from help_scripts.evaluator.quality_evaluator import QualityEvaluator
from help_scripts.helpers import ensure_dir_exists


def plot_money(learn, test, file_name):
    import plotly.graph_objs as go
    from plotly.offline import plot

    learn_time = [x for x in range(0, len(learn))]
    test_time = [x for x in range(0, len(test))]

    traces = [go.Scatter(x=learn_time, y=learn, mode='lines', name="Learn money"),
              go.Scatter(x=test_time, y=test, mode='lines', name="Test money")]
    layout = {"title": "Series"}
    fig = {"data": traces, "layout": layout}
    plot(fig, filename=file_name)


def plot(series, trend, noise, bought_indices, sell_indices, file_name):
    import plotly.graph_objs as go
    from plotly.offline import plot

    time = [x for x in range(0, len(series))]
    traces = [go.Scatter(x=time, y=series, mode='lines', name="series"),
              go.Scatter(x=time, y=trend, mode='lines', name="trend"),
              go.Scatter(x=time, y=noise, mode='lines', name="noise")]
    if len(bought_indices) > 0:
        traces.append(
            go.Scatter(x=bought_indices, y=np.take(series, bought_indices), mode="markers", name="bought"))
    if len(sell_indices) > 0:
        traces.append(go.Scatter(x=sell_indices, y=np.take(series, sell_indices), mode="markers", name="sell"))

    layout = {"title": "Series"}
    fig = {"data": traces, "layout": layout}
    plot(fig, filename=file_name)


def get_seed(params):
    if params.seed is not None:
        np.random.seed(params.seed)
    params.seed = np.random.randint(low=0,
                                    high=100000,
                                    size=params.folds)
    params.seed = list(params.seed)


def parse_cmd(*args, **kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--plot",
                        action='store_const',
                        const=True,
                        help="Plot series")

    parser.add_argument("-N", "--folds",
                        type=int,
                        default=10,
                        help="Total number of series to simulate")

    parser.add_argument("-l", "--learn_size",
                        type=int,
                        default=1000)

    parser.add_argument("-t", "--test_size",
                        type=int,
                        default=1000)

    parser.add_argument("-b", "--trend_start",
                        type=float,
                        default=1000,
                        help="Starting point for trend")
    parser.add_argument("-m", "--mean_tang",
                        type=float,
                        default=0,
                        help="Mean of tangence for trend")
    parser.add_argument("-v", "--var_tang",
                        type=float,
                        default=0.5,
                        help="Varience of tangence for trend")
    parser.add_argument("-s", "--var_noise",
                        type=float,
                        default=10,
                        help="Varience of noise")
    parser.add_argument("-e", "--exp_lamb",
                        type=float,
                        default=30,
                        help="Intensity of moments of trend change")

    parser.add_argument("--seed",
                        type=int,
                        default=np.random.randint(low=0,
                                                  high=100000,
                                                  size=1)[0],
                        help="Random seed")

    parser.add_argument("--learner",
                        type=str,
                        default=None,
                        help="Learner script")

    # parser.add_argument("--config",
    #                     type=str,
    #                     default="config.json",
    #                     help="Config filename")

    return parser.parse_args()


def fold_dir_name(fold_id):
    return "fold{0:03d}".format(fold_id)


def run_script(cmd):
    import subprocess

    stderr = open("stderr.log", "w")
    stdout = open("stdout.log", "w")
    subprocess.call(cmd, stdout=stdout, stderr=stderr, shell=True)


def main(*args, **kwargs):
    params = parse_cmd(*args, **kwargs)
    get_seed(params)

    # config = None
    # with open(params.config_name) as json_data:
    #     config = json.load(json_data)

    # if config is None:
    #     raise RuntimeError("Error: no config file was founded")

    sampler = SeriesSampler(tan_mean=params.mean_tang,
                            tan_var=params.var_tang,
                            intensity=params.exp_lamb,
                            wiener_var=params.var_noise)
    current_dir = os.getcwd()

    learn_scores = []
    test_scores = []

    for i in range(params.folds):
        fold_dir = os.path.join(current_dir, fold_dir_name(i))
        ensure_dir_exists(fold_dir)

        sample = sampler.simulate(params.learn_size + params.test_size, params.trend_start, seed=params.seed[i])
        series = sample["series"]

        learn = series[0:params.learn_size]
        test = series[params.learn_size:]

        learn_file = os.path.join(fold_dir, "learn.txt")
        np.savetxt(learn_file, learn)
        np.savetxt(os.path.join(fold_dir, "test.txt"), test)

        os.chdir(fold_dir)
        run_script(params.learner)
        os.chdir(current_dir)

        learn_evaluator = QualityEvaluator(learn)
        test_evaluator = QualityEvaluator(test)

        learn_decisions = np.loadtxt(os.path.join(fold_dir, "learn.decisions"))
        test_decisions = np.loadtxt(os.path.join(fold_dir, "test.decisions"))

        learn_evaluation = learn_evaluator.evaluate(learn_decisions)
        learn_scores.append(learn_evaluation["gain"])
        test_evaluation = test_evaluator.evaluate(test_decisions)

        buy_indices = np.concatenate(
            (learn_evaluation["buy_indices"], [x + len(learn) for x in test_evaluation["buy_indices"]]))
        sell_indices = np.concatenate(
            (learn_evaluation["sell_indices"], [x + len(learn) for x in test_evaluation["sell_indices"]]))
        test_scores.append(test_evaluation["gain"])

        plot(series, sample["trend"], sample["noise"],
             buy_indices, sell_indices,
             "fold{__id:03d}.html".format(__id=i))

        plot_money(learn_evaluation["money"],
                   test_evaluation["money"],
                   "money_fold{__id:03d}.html".format(__id=i))

        print("Learn score on fold #{__id:03d}: {__score}".format(__id=i, __score=learn_scores[-1]))
        print("Test score on fold #{__id:03d}: {__score}".format(__id=i, __score=test_scores[-1]))

    np.savetxt("learn.scores", learn_scores)
    np.savetxt("test.scores", test_scores)

    print("Mean learn score {__score}".format(__score=np.mean(learn_scores)))
    print("Mean test score {__score}".format(__score=np.mean(test_scores)))


if __name__ == "__main__":
    main()
