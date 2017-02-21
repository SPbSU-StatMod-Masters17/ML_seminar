import argparse

import numpy as np
from scipy.stats import norm, expon


def get_wiener(N, sigma):
    n = norm.rvs(scale=sigma, size=N)
    return np.cumsum(n)


def get_trend(N, lamb, m_tan, v_tan, sigma, tr_start):
    left, right, cs = 0, 0, tr_start
    trend = []

    while right < N:
        left = right
        right += expon.rvs(scale=lamb)
        right = min(right, N)
        tan = norm.rvs(loc=m_tan, scale=v_tan)
        fleft, fright = np.floor(left), np.floor(right)
        supp = np.arange(fleft, fright) - fleft

        cs += tan * (right - left)
        if not len(supp):
            continue

        intercept = cs + tan * (left - fleft)
        trend.append(tan * supp + intercept)

    trend = np.hstack(trend)
    return trend


def get_series(trend, wiener):
    series = trend + wiener
    neg_ind = np.where(series < 0)[0]
    if len(neg_ind):
        series[neg_ind[0]:] = 0
    return series


def parse_cmd(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir",
                        type=str,
                        required=False, # TODO
                        help="Output directory")
    parser.add_argument("-p", "--plot",
                        action='store_const',
                        const=True,
                        help = "Plot series")
    parser.add_argument("-N", "--number_series",
                        type=int,
                        default=5,
                        help="Total number of series to simulate")
    parser.add_argument("-l", "--series_len",
                        type=int,
                        default=1000)
    parser.add_argument("-t", "--trend_start",
                        type=float,
                        default=100,
                        help="Starting point for trend")
    parser.add_argument("-m", "--mean_tang",
                        type=float,
                        default=0,
                        help="Mean of tangence for trend")
    parser.add_argument("-v", "--var_tang",
                        type=float,
                        default=5,
                        help="Varience of tangence for trend")
    parser.add_argument("-s", "--var_noise",
                        type=float,
                        default=1,
                        help="Varience of noise")
    parser.add_argument("-e", "--exp_lamb",
                        type=float,
                        default=0.2,
                        help="Intensity of moments of trend change")
    return parser.parse_args()


def main(*args, **kwargs):
    params = parse_cmd(*args, **kwargs)

    all_series = []
    for _ in xrange(params.number_series):
        wiener = get_wiener(N=params.series_len,
                            sigma=params.var_noise)
        trend = get_trend(N=params.series_len,
                          lamb=params.exp_lamb,
                          m_tan=params.mean_tang,
                          v_tan=params.var_tang,
                          sigma=params.var_noise,
                          tr_start=params.trend_start)
        series = get_series(trend, wiener)
        all_series.append(series)

    if params.plot:
        import plotly.graph_objs as go
        from plotly.offline import plot
        traces = []
        for series in all_series:
            trace = go.Scatter(x=range(1, params.series_len + 1),
                               y=series,
                               mode='lines')
            traces.append(trace)
        fig = {"data": traces}
        plot(fig)


if __name__ == "__main__":
    main()
