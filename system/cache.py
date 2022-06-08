import numpy as np
from numpy import ndarray

from policies.policy_abc import Policy
from system.server_abc import Server


class Cache(Server):

    def __init__(self, policy: Policy) -> None:
        self.misses = []
        self.cost = []
        self.policy = policy

    def process_trace(self, trace: ndarray):
        for i, request in enumerate(trace):
            future_request = trace[i + 1] if i < len(trace) - 2 else None
            _ = self.process_request(request, future_request)


    def process_request(self, request: ndarray, future_request: ndarray) -> float:
        y = self.policy.get(request)
        self.misses.append(1 - y)
        self.cost.append(self.policy.cost(request))
        if self.policy.name == "OOMD":
            self.policy.set_future_request(future_request)
        self.policy.put(request)
        return y

    def get_cache_content(self):
        return self.policy.cache_content()

    def get_avg_hit_ratio(self) -> ndarray:
        T = len(self.misses)
        avg = np.zeros(T)
        avg[0] = self.misses[0]
        for t in range(1, T):
            avg[t] = sum(self.misses[:t]) / t

        return avg

    def get_avg_cost(self) -> ndarray:
        T = len(self.misses)
        avg = np.zeros(T)
        avg[0] = self.cost[0]
        for t in range(1, T):
            avg[t] = sum(self.cost[:t]) / t

        return avg

    def get_label(self):
        return self.policy.get_label()

    def pretty_print(self):
        print("==========================" + self.get_label() + "==========================")
        print("Total   cost: " + str(sum(self.misses)))
        print("Avg     cost: " + str(np.mean(self.cost)))
        print("Cache  state: " + str(self.get_cache_content()))

