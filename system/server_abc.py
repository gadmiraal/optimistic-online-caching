from abc import ABC, abstractmethod

from numpy import ndarray


class Server(ABC):

    def __int__(self):
        pass

    @abstractmethod
    def process_request(self, request: ndarray) -> float:
        pass
