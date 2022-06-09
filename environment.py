import json

import numpy as np
from matplotlib import pyplot as plt

from policies.OMD_Network import OMD_Network
from policies.OOMD_Network import OOMD_Network
from policies.Optimal import Optimal
from system.cache import Cache
from traces.trace_abc import Trace
from policies.LRU import LRU
from policies.LFU import LFU
from policies.OMD import OMD
from policies.OOMD import OOMD
from traces.adversarial import Adversarial
from traces.poisson_point import PoissonPoint
from traces.fixed_pop import FixedPop
from traces.sliding_pop import SlidingPop


class Environment:

    def __init__(self, path: str):
        self.users = None
        self.trace = None
        self.caches = None
        self.k = None
        self.N = None
        self.T = None
        self.I = None
        self.J = None
        self.requests = None
        self.c_names = None
        self.read_config_file(path)

    def read_config_file(self, path: str):
        file = open(path)
        data = json.load(file)

        # Get system constants
        N = data["catalog"]
        k = data["cache size"]
        T = data["time window"]
        c_types = data["cache type"]
        trace_name = data["trace"]
        I = data["users"]
        J = data["caches"]

        # Create caches
        cs = []
        c_names = []
        for c_type in c_types:
            cache_name = list(c_type.keys())[0]
            c_names.append(cache_name)
            klass = globals()[cache_name]
            if cache_name == "OOMD":
                cache = Cache(klass(k, N, T, c_type[cache_name]))
                cs.append(cache)
            elif cache_name == "OMD_Network":
                cache = Cache(klass(k, N, T, I, J))
                cs.append(cache)
            elif cache_name == "OOMD_Network":
                cache = Cache(klass(k, N, T, c_type[cache_name], I, J))
                cs.append(cache)
            else:
                cs.append(Cache(klass(k, N, T)))

        self.k = k
        self.N = N
        self.T = T
        self.I = I
        self.J = J
        self.caches = cs
        self.c_names = c_names
        self.trace = globals()[trace_name](N, T)

    def plot_caches(self):
        for cache in self.caches:
            cost = cache.get_avg_cost()
            plt.plot(np.arange(len(cost)), cost, "--", label=cache.get_label())

        plt.title("Average cache cost, for trace: " + self.trace.get_name())
        plt.xlabel("Time")
        plt.ylabel("Average cache cost")
        plt.legend()
        plt.show()

    def print_caches(self):
        for cache in self.caches:
            cache.pretty_print()

    def set_trace(self, trace: Trace):
        self.trace = trace

    def execute(self):
        requests = np.zeros((self.T, self.I, self.N))
        for i in range(self.I):
            r = self.trace.transform_to_request_array(self.trace.generate())
            requests[:, i, :] = r
        # requests = self.trace.generate()
        # self.trace.plot(rs)
        # requests = self.trace.transform_to_request_array(requests)
        self.requests = requests
        for cache in self.caches:
            cache.process_trace(requests)

    def execute_with_optimal(self):
        self.execute()

        test = Cache(OOMD(self.k, self.N, self.T, 1.0))
        test.process_trace(self.requests)
        cache_state = test.policy.x
        optimal = Cache(Optimal(self.k, self.N, self.T, cache_state))
        optimal.process_trace(self.requests)

        self.caches.append(optimal)


