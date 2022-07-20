from abc import ABC, abstractmethod
from typing import List
from Client import Client
from torch.nn import Module
from torch.utils.data import DataLoader
import numpy as np


def random_selection(num_clients, num_selected):
    return np.random.choice(num_clients, num_selected, replace=False)

class Server(ABC):
    clients: List[Client]
    global_model: Module
    data_loader: DataLoader

    @abstractmethod
    def select_clients(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass

class FedAvgServer(Server):
    def __init__(self, clients, testloader, select_strategey, wandb=True):
        self.clients = clients
        self.testloader = testloader

