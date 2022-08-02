from abc import ABC, abstractmethod
from typing import List
from Client import Client
from torch.nn import Module
from torch.utils.data import DataLoader
import numpy as np
import wandb


def random_selection(num_clients, num_selected):
    return np.random.choice(num_clients, num_selected, replace=False)


class Server(ABC):
    clients_list: List[Client]
    global_model: Module
    data_loader: DataLoader

    @abstractmethod
    def select_clients(self):
        pass

    @abstractmethod
    def aggregate(self):
        pass

class BaseServer(Server):
    def __init__(self, clients_list, global_model, testloader):
        self.clients_list = clients_list
        self.global_model = global_model
        self.testloader = testloader
        wandb.watch(global_model)
    
    def select_clients(self):
        return self.client_list
    
    def evaluate(self):
        

