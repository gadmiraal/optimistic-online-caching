import numpy as np
import pandas as pd

from traces.trace_abc import Trace
np.random.seed(42)


class Adversarial(Trace):

    def __init__(self, catalog, time):
        super().__init__(catalog, time)

    @staticmethod
    def get_name():
        return "Adversarial"

    def generate(self):
        shots = []
        half = int(self.N / 2)
        cat1 = np.arange(half)
        cat2 = np.arange(half, self.N)
        gamma = 1
        arrival_new_request = np.random.exponential(1 / gamma, self.N // 10)
        Tau = np.cumsum(arrival_new_request)
        np.random.shuffle(Tau)

        for k in range(int(half / 10)):
            for i in range(self.T * 10):
                shots.append([cat1[i % 10 + 10 * k], Tau[k] + .05 * i])
                shots.append([cat2[i % 10 + 10 * k], Tau[k] + .05 * i])
        rs = pd.DataFrame(shots, columns=['R', 't', ]).sort_values(by='t').R.values[:self.T]

        return rs
