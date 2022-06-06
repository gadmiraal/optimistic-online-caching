from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray


class Policy(ABC):

    def __init__(self, capacity: int, catalog: int, time_window: int) -> None:
        self.k = capacity
        self.N = catalog
        self.T = time_window
        self.w = np.ones(catalog)  # Todo when we move from a single cache system

    @abstractmethod
    def get(self, y: ndarray) -> float:
        pass

    @abstractmethod
    def put(self, y: ndarray) -> None:
        pass

    @abstractmethod
    def cache_content(self):
        pass

    @abstractmethod
    def cost(self, r_t) -> float:
        pass
