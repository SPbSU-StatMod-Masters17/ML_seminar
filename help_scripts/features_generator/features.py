from collections import deque
import numpy as np
import scipy.stats
import pandas as pd


class CountersCalcer(object):
    def __init__(self, window_size):
        self._window_size = window_size
        self._history = deque()
        self._feature = 0

    def _update_history(self, x):
        prev = 0
        if len(self._history) > self._window_size:
            prev = self._history[-self._window_size]
            self._history.popleft()
        self._history.append(x)
        return prev

    def add(self, x):
        pass

    def feature(self):
        return self._feature

    def add_batch(self, batch):
        batch = batch[-self._window_size:]
        for x in batch:
            self.add(x)

    def window_size(self):
        return self._window_size


class SignCountersCalcer(CountersCalcer):
    def __init__(self, window_size):
        super().__init__(window_size)

    def add(self, x):
        prev = self._update_history(x)
        self._feature -= 1 if prev > 0 else 0
        self._feature += 1 if x > 0 else 0

    def name(self):
        return "sign_{__name}".format(__name=self._window_size)


class WilcoxonCalcer(CountersCalcer):
    def __init__(self, window_size):
        super().__init__(window_size)

    def add(self, x):
        self._update_history(x)
        self._feature = scipy.stats.wilcoxon(self._history).statistic

    def name(self):
        return "wilcoxon_{__name}".format(__name=self._window_size)


class TTestCalcer(CountersCalcer):
    def __init__(self, window_size):
        super().__init__(window_size)

    def add(self, x):
        self._update_history(x)
        self._feature = scipy.stats.ttest_1samp(self._history, 0).statistic

    def name(self):
        return "ttest_{__name}".format(__name=self._window_size)


class ExponentialSmoothingCalcer(CountersCalcer):
    def __init__(self, alpha):
        super().__init__(int(np.math.log(1e-4) / np.math.log(alpha)))
        self.__alpha = alpha

    def add(self, x):
        self._update_history(x)
        self._feature = self.__alpha * self._feature + (1.0 - self.__alpha) * x

    def name(self):
        return "exp_smooth_{__name}".format(__name=self.__alpha)


class DifferenceCalcerWrapper(object):
    def __init__(self, calcer):
        self.__current = None
        self.__calcer = calcer

    def add(self, x):
        if self.__current is None:
            self.__current = x
        else:
            self.__calcer.add(x - self.__current)
            self.__current = x

    def add_batch(self, batch):
        batch = batch[-(self.__calcer.window_size() + 1):]
        for x in batch:
            self.add(x)

    def feature(self):
        return self.__calcer.feature()


    def name(self):
        return "difference_{__name}".format(__name=self.__calcer.name())


class SecondDifferenceCalcerWrapper(object):
    def __init__(self, calcer):
        self.__current = None
        self.__prev = None
        self.__calcer = calcer

    def add(self, x):
        if self.__prev is None:
            self.__prev = x
        elif self.__current is None:
            self.__current = x
        else:
            self.__calcer.add(x - 2 * self.__current + self.__prev)
            self.__prev = self.__current
            self.__current = x

    def add_batch(self, batch):
        batch = batch[-(self.__calcer.window_size() + 1):]
        for x in batch:
            self.add(x)

    def feature(self):
        return self.__calcer.feature()

    def name(self):
        return "second_difference_{__name}".format(__name=self.__calcer.name())


class DataSetBuilder(object):

    @staticmethod
    def create_all_generators():
        generators = []
        window_sizes = [10, 16, 24, 32, 64]
        alphas = [0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        for window in window_sizes:
            generators.append(DifferenceCalcerWrapper(WilcoxonCalcer(window)))
            generators.append(DifferenceCalcerWrapper(TTestCalcer(window)))
            generators.append(DifferenceCalcerWrapper(SignCountersCalcer(window)))
            generators.append(SecondDifferenceCalcerWrapper(TTestCalcer(window)))

        for alpha in alphas:
            generators.append(DifferenceCalcerWrapper(ExponentialSmoothingCalcer(alpha)))
            generators.append(SecondDifferenceCalcerWrapper(ExponentialSmoothingCalcer(alpha)))
            generators.append(ExponentialSmoothingCalcer(alpha))
        return generators

    @staticmethod
    def build(generators, series):
        df = pd.DataFrame(columns=["qid", "target", "qurl", "gid"] + [x.name() for x in generators])

        for i in range(0, len(series)):
            for g in generators:
                g.add(series[i])
            df.loc[i] = ["{}".format(i), series[i], "none", 0] + [x.feature() for x in generators]

        return df

    @staticmethod
    def build_with_history(generators, history, series):
        for g in generators:
            g.add_batch(history)
        return DataSetBuilder.build(generators, series)
