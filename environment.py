import json

import numpy as np
from matplotlib import pyplot as plt

from policies.Optimal import Optimal
from system.cache import Cache
from system.main_server import MainServer
from system.server_abc import Server
from system.user import User
from traces.trace_abc import Trace
from policies.LRU import LRU
from policies.LFU import LFU
from policies.OMD import OMD
from policies.OOMD import OOMD


class Environment:

    def __init__(self, path: str):
        self.users = None
        self.trace = None
        self.caches = None
        self.k = None
        self.N = None
        self.T = None
        self.c_type = None
        self.read_config_file(path)

    def read_config_file(self, path: str):
        file = open(path)
        data = json.load(file)

        # Get system constants
        N = data["catalog"]
        k = data["cache size"]
        T = data["time window"]
        c_type = data["cache type"]

        # Create caches
        cs = []
        klass = globals()[c_type]
        for i in range(data["caches"]):
            if c_type == "OOMD":
                cs.append(Cache(klass(k, N, T, data["chance"])))
            else:
                cs.append(Cache(klass(k, N, T)))

        # Create users
        main_server = MainServer()
        us: list[User] = []
        for user in data["config"]:
            # Create the server list with distances
            server_list: list[(float, Server)] = []
            for key in user["distance to caches"].keys():
                distance = user["distance to caches"][key]
                cache = cs[int(key)]
                server_list.append((distance, cache))
            server_list.append((user["distance to main"], main_server))

            us.append(User(server_list))

        self.k = k
        self.N = N
        self.T = T
        self.users = us
        self.caches = cs
        self.c_type = c_type

    def plot_caches(self, trace_name):
        for cache in self.caches:
            num_users = 1 #Todo change!!!
            x = np.array(cache.cost)
            cost_per_time = x.reshape((int(x.shape[0] / num_users), num_users)).sum(axis=1) / num_users
            T = len(cost_per_time)
            avg = np.zeros(T)
            avg[0] = cost_per_time[0]
            for t in range(1, T):
                avg[t] = sum(cost_per_time[:t]) / t

            plt.plot(np.arange(T), avg, "--")

        plt.title("Average cache cost with: " + self.c_type + ", for trace: " + trace_name)
        plt.xlabel("Time")
        plt.ylabel("Average cache cost")
        # plt.legend(loc="lower right")
        plt.show()

    def set_trace(self, trace: Trace):
        self.trace = trace

    def execute(self):
        # Todo change when moving to a bipartite system
        for user in self.users:
            rs = self.trace.generate()
            # self.trace.plot(rs)
            rs = self.trace.transform_to_request_array(rs)
            user.set_trace(rs)

        for t in range(self.trace.T):
            for user in self.users:
                request = user.trace[t]
                user.execute_request(request)

    def execute_with_optimal(self):
        self.execute()
        if self.c_type == "OMD":
            optimals = []
            for cache in self.caches:
                optimal = Optimal(self.k, self.N, self.T)
                optimal.set_cache(cache.return_x())
                optimals.append(optimal)






