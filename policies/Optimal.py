import numpy as np

from policies.policy_abc import Policy
from system.cache import Cache


class Optimal(Policy):

    def __init__(self, capacity: int, catalog: int, time_window: int) -> None:
        super().__init__(capacity, catalog, time_window)
        self.x = np.zeros(catalog)
        self.name = "Optimal"


    def set_cache(self, x):
        self.x = x

    def get(self, y) -> int:
        key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
        return self.x[key]

    def put(self, y):
        pass

    def cache_content(self):
        pass
