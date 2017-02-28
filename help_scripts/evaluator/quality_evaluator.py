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

        buy_indices = []
        sell_indices = []

        for i in range(0, len(decisions)):
            if decisions[i] == 1:
                if self.__state == State.WITHOUT_STOCKS:
                    gain -= self.__pack_size * self.__prices[i]
                    self.__state = State.WITH_STOCKS
                    buy_indices.append(i)
                else:
                    gain += self.__pack_size * self.__prices[i]
                    self.__state = State.WITHOUT_STOCKS
                    sell_indices.append(i)

        if self.__state == State.WITH_STOCKS:
            gain += self.__pack_size * self.__prices[-1]
            sell_indices.append(len(decisions) - 1)

        return {"gain": gain, "buy_indices": buy_indices, "sell_indices": sell_indices}
