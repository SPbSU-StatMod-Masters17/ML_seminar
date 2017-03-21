#!/usr/bin/env python
import argparse
import json
import os.path

import errno
import numpy as np

from help_scripts.simulation.simulator import SeriesSampler

from help_scripts.evaluator.quality_evaluator import QualityEvaluator
from help_scripts.helpers import ensure_dir_exists


def main(*args, **kwargs):
    import plotly.graph_objs as go
    from plotly.offline import plot

    series = np.loadtxt("series.txt")
    predictions = np.loadtxt("predictions.txt")
    trend = np.loadtxt("trend.txt")

    time = [i for i in range(0, len(trend))]

    mse_with_trend = np.sqrt(np.sum((predictions - trend) ** 2) / len(trend))
    mse_with_series = np.sqrt(np.sum((predictions - series) ** 2) / len(trend))
    prev = np.concatenate(([0], series[0:(len(trend) - 1)]))
    mse_with_prev = np.sqrt(np.sum((prev - series) ** 2) / len(trend))

    traces = [go.Scatter(x=time, y=series, mode='lines', name="Series"),
              go.Scatter(x=time, y=trend, mode='lines', name="Trend"),
              go.Scatter(x=time, y=predictions, mode='lines', name="Predictions")]

    layout = {"title": "Series"}
    fig = {"data": traces, "layout": layout}
    print("MSE with trend {__mse}".format(__mse=mse_with_trend))
    print("MSE with series {__mse}".format(__mse=mse_with_series))
    print("MSE series with prev {__mse}".format(__mse=mse_with_prev))
    plot(fig, filename="mse_prediction_plot.html")


if __name__ == "__main__":
    main()
