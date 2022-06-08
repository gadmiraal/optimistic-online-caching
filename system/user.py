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
        for i, request in enumerate(trace):
            collected = 0
            cost = 0
            for distance, server in self.connected_servers:
                future_request = trace[i+1] if i < len(trace) - 2 else None
                found = server.process_request(request, future_request)
                if found > 0:
                    collected += found
                    cost += distance
                # If we collected a whole file return the cost it took
                if collected >= 1:
                    break

            # If for some reason we could not get the whole file from all servers return the partial file
            total_cost += cost


