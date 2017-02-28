#!/usr/bin/env python
import os.path
import os

import errno
import numpy as np

from help_scripts.simulation.simulator import SeriesSampler
from help_scripts.helpers import ensure_dir_exists


def parse_cmd(*args, **kwargs):
    import argparse
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir",
                        type=str,
                        default=current_dir,
                        help="Output directory")
    parser.add_argument("-p", "--plot",
                        action='store_const',
                        const=True,
                        help="Plot series")
    parser.add_argument("-N", "--number_series",
                        type=int,
                        default=1,
                        help="Total number of series to simulate")

    parser.add_argument("-l", "--series_len",
                        type=int,
                        default=1000)

    parser.add_argument("-b", "--trend_start",
                        type=float,
                        default=100,
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
                        default=0.1,
                        help="Varience of noise")
    parser.add_argument("-e", "--exp_lamb",
                        type=float,
                        default=30,
                        help="Intensity of moments of trend change")
    parser.add_argument("--seed",
                        type=int,
                        default=int(np.random.randint(low=0,
                                                      high=100000,
                                                      size=1)[0]),
                        help="Random seed")

    return parser.parse_args()


def dump_params(params):
    import json
    with open(os.path.join(params.outdir, "params.txt"), 'w') as f:
        json.dump(vars(params), f, sort_keys=True, indent=4)


def get_seed(params):
    if params.seed is not None:
        np.random.seed(params.seed)
    params.seed = np.random.randint(low=0,
                                    high=100000,
                                    size=params.number_series)
    # np.int is not json-serializable in python3
    params.seed = [int(x) for x in params.seed]


def prepare_output_dir(params):
    ensure_dir_exists(params.outdir)


def plot_series(series):
    import plotly.graph_objs as go
    from plotly.offline import plot

    traces = []
    for sample in series:
        trace = go.Scatter(x=[x for x in range(1, len(sample) + 1)],
                           y=sample,
                           mode='lines')
        traces.append(trace)
    layout = {"title": "Series"}
    fig = {"data": traces, "layout": layout}
    plot(fig)


def main(*args, **kwargs):
    params = parse_cmd(*args, **kwargs)
    prepare_output_dir(params)
    get_seed(params)
    dump_params(params)

    sampler = SeriesSampler(tan_mean=params.mean_tang,
                            tan_var=params.var_tang,
                            intensity=params.exp_lamb,
                            wiener_var=params.var_noise)

    all_series = []
    for i in range(params.number_series):
        series = sampler.simulate(params.series_len, params.trend_start, seed=params.seed[i])
        all_series.append(series)
        np.savetxt(os.path.join(params.outdir, str(i) + ".txt"), series)

    if params.plot:
        plot_series(all_series)


if __name__ == "__main__":
    main()
