import numpy as np

from traces.trace_abc import Trace
# np.random.seed(42)


class FixedPop(Trace):

    def __init__(self, catalog, time):
        super().__init__(catalog, time)

    def zipf_distribution(self, s: float):
        c = np.sum((1 / np.arange(1, self.N + 1) ** s))
        return np.arange(1, self.N + 1) ** (-s) / c

    @staticmethod
    def get_name(): return "Fixed popularity"

    def generate(self):
        s = 0.6
        p = self.zipf_distribution(s)
        rs = np.random.choice(np.arange(self.N), p=p, size=self.T)
        self.T = rs.shape[0]

        return rs
