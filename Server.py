from abc import ABC, abstractmethod
from typing import List
from Client import Client
from torch.nn import Module

class Server(ABC):
    clients: List[Client]
    global_model: Module

    @abstractmethod
    def _select_clients(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass

class FedAvgServer(Server):
    def __init__(self, clients, testloader, select_strategey):
        self.clients = clients
        self.testloader = testloader

