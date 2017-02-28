import scipy.stats
import numpy as np

from help_scripts.evaluator.quality_evaluator import QualityEvaluator


class WilcoxonDetector(object):
    def __init__(self, threshold=1e-3, sample_size=25, window=3, stripe=1):
        self.__threshold = threshold
        self.__sample_size = sample_size
        self.__window = window
        self.__stripe = stripe
        self.__samples = []
        self.__cumsums = [0]

    def __create_differences(self):
        # we don't have enough data
        if self.__stripe + self.__sample_size + self.__window >= len(self.__samples):
            return None

        differences = []
        for i in range(self.__sample_size, 0, -1):
            mean_cur = (self.__cumsums[-i] - self.__cumsums[-i - self.__window]) / self.__window
            mean_prev = (self.__cumsums[-i - self.__stripe] - self.__cumsums[
                -i - self.__window - self.__stripe]) / self.__window
            # we want different to be > 0 for positive trend
            differences.append(mean_cur - mean_prev)
        return np.array(differences)

    def add_sample(self, value):
        self.__samples.append(value)
        self.__cumsums.append(self.__cumsums[-1] + value)

    def current_trend(self):

        diffs = self.__create_differences()

        if diffs is None:
            return 0

        test_pvalue = scipy.stats.wilcoxon(diffs).pvalue
        # test_pvalue = scipy.stats.ttest_1samp(diffs, 0).pvalue

        if test_pvalue < self.__threshold:
            return np.sign(np.sum(diffs))

        return 0

    @staticmethod
    def make_decisions(series, sample_size, window_size, stripe, threshold):
        detector = WilcoxonDetector(threshold=threshold, sample_size=sample_size, window=window_size, stripe=stripe)
        decisions = []
        has_stocks = False

        for sample in series:
            detector.add_sample(sample)
            decision = detector.current_trend()
            # buy
            if decision > 0 and not has_stocks:
                decisions.append(1)
                has_stocks = True
            # sell
            elif decision < 0 and has_stocks:
                decisions.append(1)
                has_stocks = False
            else:
                decisions.append(0)
        return decisions


class GridSearch(object):

    def __init__(self, series):
        self.__series = series
        self.__evaluator = QualityEvaluator(series)

    def __calc_score(self, sample_size, window_size, stripe, threshold):
        decisions = WilcoxonDetector.make_decisions(self.__series, threshold=threshold, sample_size=sample_size,
                                                    window_size=window_size, stripe=stripe)
        return self.__evaluator.evaluate(decisions)["gain"]

    def search(self,
               min_sample_size=20, max_sample_size=45, sample_stripe=5,
               min_threshold=8e-6, max_threshold=2e-4, thresholds_count=50,
               min_window=1, max_window=2,
               min_stripe=1, max_stripe=2
               ):

        best_score = -100500
        best_params = {}

        for sample_size in range(min_sample_size, max_sample_size, sample_stripe):
            for window_size in range(min_window, max_window):
                for stripe in range(min_stripe, max_stripe):
                    threshold = min_threshold
                    step = (max_threshold - min_threshold) / thresholds_count
                    while threshold < max_threshold:
                        score = self.__calc_score(sample_size, window_size, stripe, threshold)
                        params = {"threshold": threshold, "sample_size": sample_size,
                                  "window_size": window_size, "stripe": stripe}

                        print("Params: {__params}; Score {__score}".format(__params=str(params), __score=score))

                        if score > best_score:
                            best_score = score
                            best_params = params
                        threshold += step

        print("Best gain: {__score}".format(__score=best_score))
        return best_params
