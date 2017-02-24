import argparse
import os
import os.path
import errno

import numpy as np
import help_scripts.simulation.simulator
from enum import Enum


class State(Enum):
    WITH_STOCKS = "With stocks",
    WITHOUT_STOCKS = "Without stocks"


class QualityEvaluator(object):
    def __init__(self, prices, pack_size=100):
        self.__prices = prices
        self.__pack_size = pack_size
        self.__state = State.WITHOUT_STOCKS

    def evaluate(self, decisions):
        gain = 0

        for i in range(0, len(decisions)):
            if decisions[i] == 1:
                if self.__state == State.WITHOUT_STOCKS:
                    gain -= self.__pack_size * self.__prices[i]
                    self.__state = State.WITH_STOCKS
                else:
                    gain += self.__pack_size * self.__prices[i]
                    self.__state = State.WITHOUT_STOCKS

        if self.__state == State.WITH_STOCKS:
            gain += self.__pack_size * self.__prices[i]

        return gain
