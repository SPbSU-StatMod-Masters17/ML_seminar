#!/usr/bin/env python

import os.path
import os

import errno
import numpy as np

from help_scripts.wilcoxon_baseline.wilcoxon_trend_estimator import GridSearch, WilcoxonDetector


def main(*args, **kwargs):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--threshold-grid",
                        type=int,
                        default=20,
                        help="Threshold grid size")

    args = parser.parse_args()
    learn = np.loadtxt("learn.txt")
    search = GridSearch(learn)
    best_params = search.search(thresholds_count=args.threshold_grid)
    print("Best params: " + str(best_params))

    test = np.loadtxt("test.txt")

    learn_decisions = WilcoxonDetector.make_decisions(learn, threshold=best_params["threshold"],
                                                      sample_size=best_params["sample_size"],
                                                      window_size=best_params["window_size"],
                                                      stripe=best_params["stripe"])
    np.savetxt("learn.decisions", learn_decisions)

    tmp = np.concatenate((learn, test))
    test_decisions = WilcoxonDetector.make_decisions(tmp, threshold=best_params["threshold"],
                                                     sample_size=best_params["sample_size"],
                                                     window_size=best_params["window_size"],
                                                     stripe=best_params["stripe"])[-len(test):]
    np.savetxt("test.decisions", test_decisions)





if __name__ == "__main__":
    main()
