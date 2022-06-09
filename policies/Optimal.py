import numpy as np

from policies.policy_abc import Policy
from system.cache import Cache


class Optimal(Policy):

    def __init__(self, capacity: int, catalog: int, time_window: int, cache_state) -> None:
        super().__init__(capacity, catalog, time_window)
        self.x = np.full(self.N, self.k / self.N) if cache_state is None else cache_state
        self.name = "Optimal"


    def set_cache(self, x):
        self.x = x

    def get(self, y) -> int:
        key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
        return self.x[key]

    def put(self, y):
        pass

    def cache_content(self):
        keys = np.arange(self.N)
        zipped = zip(keys, np.round(self.x, 6))
        return dict(zipped)

    def cost(self, r_t):
        return np.sum(self.w * r_t * (1 - self.x))

    def get_label(self) -> str:
        return self.name
