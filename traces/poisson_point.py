import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from traces.trace_abc import Trace


class PoissonPoint(Trace):

    def __init__(self, catalog, time):
        super().__init__(catalog, time)

    @staticmethod
    def get_name(): return "Poisson Point"

    def generate(self):
        portions = np.array([.044, .069, .041, .062, 1.196])
        vs = np.array([86.4, 41.9, 59.5, 36.9, 25.7]) * 3
        ls = np.array([1.14, 3.36, 6.40, 10.53, 24.61])
        portions = portions / sum(portions) * 100
        portions = np.hstack((0, np.round(portions).cumsum())).astype(int)
        L = np.ones(self.N)
        V = np.ones(self.N)
        for i in range(len(portions) - 1):
            V[portions[i]:portions[i + 1]] = vs[i]
            L[portions[i]:portions[i + 1]] = ls[i]

        gamma = 5
        arrival_new_request = np.random.exponential(1 / gamma, self.N)
        Tau = np.cumsum(arrival_new_request)
        np.random.shuffle(Tau)

        Rs = []
        for i in range(self.N):
            if V[i] != 0:
                vm = V[i]
                lm = L[i]
                tau_m = Tau[i]
                sigma = L[i]
                T = 0
                while True:
                    arr = np.random.exponential(sigma / vm, 1)
                    T = T + float(arr)
                    if (T > L[i]):
                        break
                    Rs.append([i, tau_m + T])

        rs = pd.DataFrame(Rs, columns=['R', 't', ]).sort_values(by='t').R.values

        rs = np.asarray(rs)
        df = pd.DataFrame(np.array([rs[:100_000], np.arange(rs[:100_000].size) * .01]).T, columns=['R', 't']).sample(
            frac=1).iloc[
             :10_000]

        df = df.round()
        zipped = list(zip(df.t.to_numpy(), df.R.to_numpy()))
        rs = np.full(len(zipped), -1)
        for (t, r) in zipped:
            if rs[int(t)] == -1:
                rs[int(t)] = int(r)

        print(rs.shape)
        plt.scatter(rs, s=0.5, color='C0')
        plt.show()
        return rs

