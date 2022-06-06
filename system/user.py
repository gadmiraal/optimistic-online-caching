from numpy import ndarray

from system.server_abc import Server


class User:

    def __init__(self, connected_servers: list[(float, Server)]):
        connected_servers.sort(key=lambda i: i[0])
        self.connected_servers = connected_servers
        self.trace = None

    def set_trace(self, trace: ndarray) -> None:
        self.trace = trace

    def execute_trace(self, trace: ndarray):
        total_cost = 0
        for request in trace:
            total_cost += self.execute_request(request)

    def execute_request(self, request: ndarray) -> int:
        collected = 0
        cost = 0
        for distance, server in self.connected_servers:
            found = server.process_request(request)
            if found > 0:
                collected += found
                cost += distance
            # If we collected a whole file return the cost it took
            if collected >= 1:
                return cost

        # If for some reason we could not get the whole file from all servers return the partial file
        return cost
