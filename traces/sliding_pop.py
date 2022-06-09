import numpy as np

from traces.trace_abc import Trace
from policies.LRU import LRU
from policies.LFU import LFU
from policies.OMD import OMD
# np.random.seed(42)

class SlidingPop(Trace):

    def __init__(self, catalog, time):
        super().__init__(catalog, time)

    def zipf_distribution(self, s: float):
        c = np.sum((1 / np.arange(1, self.N + 1) ** s))
        return np.arange(1, self.N + 1) ** (-s) / c

    @staticmethod
    def get_name(): return "Sliding popularity"

    def generate(self):
        divisions = np.random.randint(1, 5)
        roll_amount = np.random.randint(1, int(self.N / divisions))

        s = 0.6
        p = self.zipf_distribution(s)
        rs = np.random.choice(np.arange(self.N), p=p, size=self.T // divisions)

        for i in range(divisions - 1):
            p = np.roll(p, roll_amount)
            rs = np.hstack((rs, np.random.choice(np.arange(self.N), p=p, size=self.T // divisions)))

        self.T = rs.shape[0]

        return rs
