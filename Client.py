from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module


class Client(ABC):
    model: Module
    optimizer: Optimizer
    dataloader: DataLoader

    @abstractmethod
    def local_update(self):
        pass

    @abstractmethod
    def send(self):
        pass


class FedAvgClient(Client):
    def __init__(self, client_id, model, optimizer, dataloader):
        self.id = client_id
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader

    def local_update(self, local_epoch):
        pass

    def send(self):
        pass

    def __repr__(self):
        pass

    def _log(self):
        pass

