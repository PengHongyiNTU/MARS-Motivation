
import wandb
from Utils import get_best_gpu
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import tqdm
import FedAvg
import torch
import os

def random_selection(client_ids, num_selected):
    return np.random.choice(client_ids, num_selected, replace=False)

class LocalTrainer:
    def __init__(self, model, clients_dataidx_map, trainset, testset, cfg):
        self.global_model = model
        self.clients_dataidx_map = clients_dataidx_map
        self.trainset = trainset
        self.testset = testset
        self.cfg = cfg
        if self.cfg['use_gpu'] == True:
            self.gpu = get_best_gpu()
            self.device = torch.device(self.gpu)
        else:
            self.device = torch.device('cpu')
        saving_dir = self.cfg['saving_dir']
        # check contains three directories: local, global, and logs
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        self.local_dir = os.path.join(saving_dir, 'local')
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        self.global_dir = os.path.join(saving_dir, 'global')
        if not os.path.exists(self.global_dir):
            os.makedirs(self.global_dir)
        self.log_dir = os.path.join(saving_dir, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
                          
    
    def evaluate(self):
        print('Evaluating...')
        test_loss = 0
        with torch.no_grad():
            self.global_model.eval()
            test_loader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg['test_batch_size'])
            correct = 0
            total = 0
            for data, target in tqdm.tqdm(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
            test_loss /= total
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, total, 100. * correct / total))
            wandb.log({'test_loss': test_loss, 'test_accuracy': 100. * correct / total})
            if self.cfg['corruption_prob'] > 0.0:
                flipped_loss = 0
                restored_idx = self.trainset.get_restored_idxs(
                    self.cfg['test_batch_size'])
                correct_labels = torch.tensor(self.trainset.correct_labels[restored_idx])
                correct_labels = correct_labels.to(self.device)
                restored_dataset = torch.utils.data.Subset(self.trainset, restored_idx)
                restored_loader = torch.utils.data.DataLoader(restored_dataset,
                                                              batch_size=len(restored_dataset))
                flipped_correct = 0
                restored_correct = 0
                total = 0
                for data, target in restored_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.global_model(data)
                    flipped_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.max(1, keepdim=True)[1]
                    flipped_correct += pred.eq(target.view_as(pred)).sum().item()
                    restored_correct += pred.eq(correct_labels.view_as(pred)).sum().item()
                    total += len(data)
                flipped_loss /= total
                print(f'\nFlipped set: Flipped loss: {flipped_loss}, Flipped Accuracy: {flipped_correct}/{total} ({100. * flipped_correct / total:.0f}%) Restored Accuracy: {restored_correct}/{total} ({100. * restored_correct / total:.0f}%)\n')
                wandb.log({'flipped_loss': flipped_loss, 'flipped_accuracy': 100. * flipped_correct / total, 'restored_accuracy': 100. * restored_correct / total})
                
                    
                
    
    def train(self):
        epoch = self.cfg['epoch']
        batch_size = self.cfg['batch_size']
        lr = self.cfg['lr']
        if 'weight_decay' in self.cfg:
            weight_decay = self.cfg['weight_decay']
        local_epoch = self.cfg['local_round']
        num_clients_per_round = self.cfg['num_clients_per_round']
        print('Training Starts!')
        # store global model  
        communication_cost = 0
        for e in range(epoch):
            print('Saving the first global model...')
            save_path = f'{self.global_dir}/global_model_{e}.pth'
            torch.save(self.global_model.state_dict(), save_path)
            model_size = os.path.getsize(save_path)/(1024*1024)
            global_model_params = self.global_model.state_dict()
            client_ids = list(self.clients_dataidx_map.keys())
            # print(client_ids)
            selected_ids = random_selection(client_ids, num_clients_per_round)
            print(f'Round {e+1}: Selected Clients: {selected_ids}')
            params_dict = dict.fromkeys(selected_ids)
            for id in selected_ids:
                print()
                data_idx = self.clients_dataidx_map[id]
                train_loader = torch.utils.data.DataLoader(self.trainset,
                    sampler=SubsetRandomSampler(indices=data_idx),
                    batch_size=batch_size)
                # local_model = self.global_model.load_state_dict(global_model_params)
                local_model = self.global_model
                local_model.load_state_dict(global_model_params)
                optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
                criterion = torch.nn.CrossEntropyLoss()
                updated_params = FedAvg.local_train(id, local_model, 
                                                    train_loader, 
                                   optimizer, criterion, local_epoch, 
                                   self.device, self.cfg['log_freq'])
                params_dict[id] = updated_params
            torch.save(params_dict, f'{self.local_dir}/local_model_{e}.pth')
            communication_cost += len(selected_ids) * model_size
            wandb.log({'Communication Cost': communication_cost})
            global_params = FedAvg.aggregate(params_dict)
            self.global_model.load_state_dict(global_params)
            self.evaluate()
        torch.save(self.global_model.state_dict(), f'{self.global_dir}/global_model_final.pth')    
                
            
                
                
   
        
    

