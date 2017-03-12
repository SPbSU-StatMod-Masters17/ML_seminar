import scipy.stats
import numpy as np


import numpy as np
from scipy.stats import norm, expon


from help_scripts.simulation.simulator import SeriesSampler

class LearnTrendEstimator(object):
    def __init__(self, window=10):
        self.window=window

    def estimate_trend(self, learn):
        diff = learn[1:] - learn[:-1]
        diff = list(diff)
        len_diff = len(diff)

        diff = [0] * (self.window - 1) + diff + [0] * (self.window - 1)

        left_ma, right_ma = [], []
        diff = learn
        r_diff = diff[::-1]
        for i in xrange(len_diff):
            left_ma.append(np.mean(diff[i:(i + self.window)]))
            right_ma.append(np.mean(r_diff[i:(i + self.window)]))

        right_ma = right_ma[::-1]

        trend = [0]
        for left, right in zip(left_ma[:-1], right_ma[1:]):
            print(left, right)
            if left == right:
                trend.append(trend[-1])
            else:
                trend.append(int(left < right))
        return trend


sampler = SeriesSampler(tan_mean=0,
                        tan_var=0.5,
                        intensity=30,
                        wiener_var=0.1)

sample = sampler.simulate(100, 100, 777)
diff_trend = (sample["trend"][1:] - sample["trend"][:-1]) > 0
print(diff_trend)

est = LearnTrendEstimator(4)
est_trend = est.estimate_trend(sample["series"])
print(est_trend)

print(1 - np.mean(diff_trend ^ est_trend))
