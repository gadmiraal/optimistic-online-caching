from system.server_abc import Server


class MainServer(Server):

    def __init__(self):
        super().__init__()

    def process_request(self, request, future_request) -> float:
        return 1
