from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict
import tqdm
import wandb


class Client(ABC):
    model: Module
    optimizer: Optimizer
    dataloader: DataLoader
    message: Dict

    @abstractmethod
    def local_update(self):
        pass

    @abstractmethod
    def send(self):
        pass
    
    @abstractmethod
    def fetch(self, server_message):
        pass


class FedAvgClient(Client):
    def __init__(self, client_id, model, optimizer, dataloader, criterion, wandb=True, log_interval=100):
        self.id = client_id
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.criterion = criterion
        self.message = {
            'id': self.id,
            'model_parameters': None,
            'loss': None,
            'train_time': None,
            'message_bytes': None          
        }
        

    def local_update(self, local_epoch, round):
        for e in tqdm(range(local_epoch)):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        

    def send(self):
        pass
    
    def fetch(self, server_message):
        pass

    def __repr__(self):
        pass


