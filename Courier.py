from abc import ABC, abstractmethod
class Courier(ABC):
    def __init__(self, clients_list) -> None:
        self.clients_list = clients_list
        self.message_pool = dict.fromkeys(clients_list, None)
        self.response_pool = dict.fromkeys(clients_list, None)
    
    @abstractmethod
    def respond(self):
        pass
    @abstractmethod
    def post(self):
        pass
    @abstractmethod
    def fetch(self):
        pass
    @abstractmethod
    def flush(self):
        pass
    
class StandAloneOneGPUCourier(Courier):
    def __init__(self, clients_list):
        super().__init__(clients_list)
    
    def respond(self, dst, server_message):
            self.response_pool[dst] = server_message

    def post(self, src, message):
        self.message_pool[src] = message
    
    def fetch(self, id):
        res = self.response_pool[id]
        return res
    
    def flush(self):
        map(lambda x: x.clear(), self.response_pool.values())
        map(lambda x: x.clear(), self.message_pool.values())

