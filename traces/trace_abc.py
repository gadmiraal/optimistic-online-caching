from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import numpy as np


class Trace(ABC):

    def __init__(self, catalog, time):
        self.N = catalog
        self.T = time

    @abstractmethod
    def generate(self):
        pass

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    def transform_to_request_array(self, rs):
        final = np.zeros((rs.size, self.N))
        for i in range(rs.size):
            request = rs[i]
            final[i][request] = 1

        return final

    def plot(self, rs):
        print(rs.shape)
        plt.scatter(np.arange(rs.shape[0]), rs, s=10)
        plt.show()
