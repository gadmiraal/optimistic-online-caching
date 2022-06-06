from collections import OrderedDict

import numpy as np
from numpy import ndarray

from policies.policy_abc import Policy


class LRU(Policy):

    def __init__(self, capacity: int, catalog: int, time_window: int) -> None:
        super().__init__(capacity, catalog, time_window)
        self.cache = OrderedDict()

    def get(self, y: ndarray) -> float:
        key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
        if key not in self.cache:
            return 0
        else:
            self.cache.move_to_end(key)
            return 1

    def put(self, y: ndarray) -> None:
        key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
        value = 1                     # Todo change when multiple requests are made
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.k:
            self.cache.popitem(last=False)

    def cost(self, r_t):
        content = np.fromiter(self.cache.keys(), dtype=int)
        x = np.zeros(self.N)
        x[content] = 1
        return np.sum(self.w * r_t * (1 - x))

    def cache_content(self):
        keys = np.arange(self.N)
        zipped = zip(keys, np.zeros(self.N))
        dic = dict(zipped)
        for key in self.cache.keys():
            dic[key] = 1.0

        return dic
