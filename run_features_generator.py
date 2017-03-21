#!/usr/bin/env python

import os.path
import os

import numpy as np
import pandas as pd
from help_scripts.features_generator.features import DataSetBuilder


def main(*args, **kwargs):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backward_hist",
                        type=int,
                        default=32,
                        help="Backward step max")

    # args = parser.parse_args()
    learn = np.loadtxt("learn.txt")
    learn_builder = DataSetBuilder.build_with_history(DataSetBuilder.create_all_generators(), learn[:30], learn[30:])
    learn_builder.to_csv("features.txt", sep="\t", header=False, index=False)
    qids = pd.DataFrame(columns=["qid", "timestamp"])
    for i in range(0, len(learn)):
        qids.loc[i] = [i, i]
    qids.to_csv("query_timestamps.tsv", sep="\t", header=False, index=False)

    test = np.loadtxt("test.txt")
    test_builder = DataSetBuilder.build_with_history(DataSetBuilder.create_all_generators(), learn, test)
    test_builder.to_csv("featuresTest.txt", sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
