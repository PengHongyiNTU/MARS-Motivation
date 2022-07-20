from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict
import tqdm
import wandb
import time
from Federate import Courier



class Client(ABC):
    model: Module
    optimizer: Optimizer
    dataloader: DataLoader
    message: Dict
    courier: Courier
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
    def __init__(self, client_id, model, optimizer, dataloader, criterion, courier):
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
        self.courier = courier
        wandb.watch(self.model)
             
    def local_update(self, local_epoch, round):
        print(f'Round {round}: Client {self.id} was selected. Start Local Training.')
        pbar = tqdm(range(local_epoch))
        start = time.time()
        for e in pbar:
            running_loss = 0.0
            batch_loss = 0.0 
            for batch_idx, (data, target) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                batch_loss += loss.item()
                # print every 2000 mini-batches
                if batch_idx % 2000 == 1999:
                    running_loss /= 2000
                    pbar.set_postfix_str(f'Round: {round} Epoch: {local_epoch} Loss: {running_loss:.3f} ')
                    running_loss = 0.0
            batch_loss /= len(self.dataloader)
            self.message[loss] = batch_loss
            batch_loss = 0.0
        print(f'Round {round}: Client {self.id} Local Training Epoch {e} Finished.')
        self.message['train_time'] = time.time() - start
        self.message['message_bytes'] = wandb.util.get_model_size(self.model)
        self.message['model_parameters'] = self.model.state_dict()
        # after local training send the local paramters 
        
        

    def send(self):
        pass
    
    def fetch(self):
        pass

    def __repr__(self):
        pass


