from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List
import tqdm
import wandb
import time
import tqdm 



class Client(ABC):
    data_idx: List[int]
    message: Dict
    @abstractmethod
    def local_update(self, trainer):
        pass

    @abstractmethod
    def send(self, strategey):
        pass
    
    @abstractmethod
    def fetch(self, strategy):
        pass


class BaseSimulationClient(Client):
    def __init__(self, client_id, data_idx):
        self.id = client_id
        self.data_idx = data_idx
        self.message = {
            'id': self.id,
            'model_parameters': None,
            'loss': None,
            'train_time': None,
            'message_bytes': None          
        }
             
    def local_update(self, local_epoch, round):
        print(f'Round {round}: Client {self.id} was selected. Start Local Training.')
        pbar = tqdm(range(local_epoch))
        start = time.time()
        for e in pbar:
            running_loss = 0.0
            batch_loss = 0.0 
            pbar = tqdm(self.dataloader)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.model.device), target.to(self.model.device)
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
        return self.message
        # self.courier.post(self.id, self.message)
        
    
    def fetch(self, server_message):
        pass
        # received = self.courier.fetch(self.id)

    def __repr__(self):
        print(f'Client {self.id}, Model Type {self.model.__name__}')
        print(f'Message Contains: {self.message.keys()}')

        


