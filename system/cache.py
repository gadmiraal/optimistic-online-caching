import numpy as np
from numpy import ndarray

from policies.policy_abc import Policy
from system.server_abc import Server


class Cache(Server):

    def __init__(self, policy: Policy) -> None:
        self.misses = []
        self.cost = []
        self.policy = policy

    def process_request(self, request: ndarray) -> float:
        y = self.policy.get(request)
        self.misses.append(1 - y)
        self.cost.append(self.policy.cost(request))
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

    def get_avg_regret(self) -> ndarray:
        T = len(self.misses)
        avg = np.zeros(T)
        avg[0] = self.cost[0]
        for t in range(1, T):
            avg[t] = sum(self.cost[:t]) / t

        return avg

    def pretty_print(self, index):
        name = type(self.policy).__name__ + str(index)
        print("==========================" + name + "==========================")
        print("Total misses: " + str(sum(self.misses)))
        print("Avg   misses: " + str(np.mean(self.misses)))
        print("Total   cost: " + str(sum(self.misses)))
        print("Avg     cost: " + str(np.mean(self.cost)))
        # print("Correct recs: " + str(self.policy.correct))

