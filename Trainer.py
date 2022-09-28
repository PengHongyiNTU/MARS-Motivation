
import wandb
from ResNet import ResNet34
from Split import ClientWiseNoisySplitter, LDASplitter
from Utils import get_best_gpu
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import tqdm
import FedAvg
import torch
import os
from Utils import get_model_size
import copy
import pandas as pd
from datetime import datetime

def random_selection(client_ids, num_selected):
    return np.random.choice(client_ids, num_selected, replace=False)

class LocalSingleGPUTrainer:
    def __init__(self, cfg, model, clients_dataidx_map, trainset, testset):
        # self.global_model = model
        self.model = model
        # saving GPU only one memory
        self.local_model = copy.deepcopy(model)
        self.clients_dataidx_map = clients_dataidx_map
        self.trainset = trainset
        self.testset = testset
        self.cfg = cfg
        if self.cfg['use_gpu'] == "best":
            self.gpu = get_best_gpu()
            self.device = torch.device(self.gpu)
        elif isinstance(self.cfg['use_gpu'], int):
            gpu_id = self.cfg['use_gpu']
            cuda = f"cuda:{gpu_id}"
            self.device = torch.device(cuda)
        elif self.cfg['use_gpu'] == 'cpu':
            self.device = torch.device("cpu")
        self.saving_dir = self.cfg['project_name']
        # check contains three directories: local, global, and logs
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)
        self.start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        torch.save(self.cfg, os.path.join(self.saving_dir, f'cfg_{self.start_time}.pth'))
        self.local_dir = os.path.join(self.saving_dir, 'local')
        self.results = []
        if self.cfg['save']:
            if not os.path.exists(self.local_dir):
                os.makedirs(self.local_dir)
            self.global_dir = os.path.join(self.saving_dir, 'global')
            if not os.path.exists(self.global_dir):
                os.makedirs(self.global_dir)
        # Easier for post process    
      
 
        
    
    
    def evaluate(self):
        print('Evaluating...')
        logging_msg = {}
        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)
            test_loader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg['test_batch_size'])
        
            correct = 0
            total = 0
            for data, target in tqdm.tqdm(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
            print('Test set:  Accuracy: {}/{} ({:.0f}%)'.format(
                correct, total, 100. * correct / total))
            logging_msg.update({'test_accuracy': 100. * correct / total})
            if self.cfg['corruption_prob'] > 0.0:
                restored_idx = self.trainset.get_restored_idxs(
                    self.cfg['test_batch_size'])
                print('Select a Subset of Train Data to investigate the effect of corruption')
                
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
                    output = self.model(data)
                    pred = output.max(1, keepdim=True)[1]
                    flipped_correct += pred.eq(target.view_as(pred)).sum().item()
                    restored_correct += pred.eq(correct_labels.view_as(pred)).sum().item()
                    total += len(data)
                print(f'Flipped set:  Flipped Accuracy: {flipped_correct}/{total} ({100. * flipped_correct / total:.0f}%) Restored Accuracy: {restored_correct}/{total} ({100. * restored_correct / total:.0f}%)\n')
                logging_msg.update({'flipped_accuracy': 100. * flipped_correct / total, 
                           'restored_accuracy': 100. * restored_correct / total
                           })
        return logging_msg
            
                    
                    
             
                    
                
    
    def run(self):
        epoch = self.cfg['epoch']
        batch_size = self.cfg['batch_size']
        lr = self.cfg['lr']
        if 'weight_decay' in self.cfg:
            weight_decay = self.cfg['weight_decay']
        if 'momentum' in self.cfg:
            momentum = self.cfg['momentum']
        local_epoch = self.cfg['local_round']
        num_clients_per_round = self.cfg['num_clients_per_round']
        print('Training Starts!')
        communication_cost = 0
        for e in range(1, epoch+1):
            # In MB
            model_size = get_model_size(self.model)
            global_params = copy.deepcopy(self.model.state_dict())
            # This should be on CPU
            client_ids = list(self.clients_dataidx_map.keys())
            selected_ids = random_selection(client_ids, num_clients_per_round)
            print(f'Round {e}: Selected Clients: {selected_ids}')
            params_dict = dict.fromkeys(selected_ids)
            for client_id in selected_ids:
                data_idx = self.clients_dataidx_map[client_id]
                train_loader = torch.utils.data.DataLoader(self.trainset,
                    sampler=SubsetRandomSampler(indices=data_idx),
                    batch_size=batch_size)
                if self.cfg['optimizer'] == "SGD":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                if self.cfg['optimizer'] == 'Adam':
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                criterion = torch.nn.CrossEntropyLoss()
                if self.cfg['algorithm'] == 'FedAvg':
                    updated_params = FedAvg.local_train(client_id, self.model, 
                                                        global_params,
                                                        train_loader, 
                                                        optimizer, 
                                                        criterion, local_epoch, 
                                                        self.device)
                else:
                    raise ValueError(f'Algorithm {self.cfg["algorithm"]} is not supported!')
                params_dict[client_id] = updated_params
            
            communication_cost += len(selected_ids) * model_size
            logging_msg = {
                'epoch': e+1,
                'communication_cost': communication_cost}
                    
            new_global_params = FedAvg.aggregate(params_dict)
            self.model.load_state_dict(new_global_params)
            results = self.evaluate()
            logging_msg.update(results)
            wandb.log(logging_msg)
            self.results.append(logging_msg)
            if self.cfg['save']:
                torch.save(self.model.state_dict(), f'{self.global_dir}/global_model_{e}.pth')
                torch.save(params_dict, f'{self.local_dir}/local_model_{e}.pth')
        df = pd.DataFrame(self.results)
        df.to_csv(self.saving_dir + f'/results_{self.start_time}.csv')
            
            
           
        
        
if __name__ == "__main__":
    import yaml
    from Split import IIDSplitter
    from Data import load_centralized_dataset
    from Model import ConvNet2, ConvNet5
    from ResNet import ResNet18, ResNet34, ResNet50
    import wandb
    from Data import ImbalancedNoisyDataWrapper
    wandb.login()
    cfg = yaml.safe_load(open('sample.yaml'))
    """
    model = ConvNet2(in_channels=1, h=32, w=32,
                 hidden=2048, class_num=10,  dropout=.0)
    """
    model = ResNet34(input_channels=3)
    # dataset_name = cfg['dataset']
    dataset_name = 'CIFAR10'
    trainset, testset = load_centralized_dataset(
        dataset_name, validation_split=0, download=False)
    cfg['corruption_prob'] = 0.2
    cfg['corruption_type'] = 'uniform'
    cfg['optimizer'] = 'SGD'
    cfg['local_round'] = 3
    trainset = ImbalancedNoisyDataWrapper(
        trainset, corruption_prob=cfg['corruption_prob'],
        imbalanced_ratio=0, num_classes=10, seed=1, noise_type=cfg['corruption_type'])
    client_num = 10
    iid_splitter = IIDSplitter(client_num)
    # clients_dataidx_map = iid_splliter(trainset)
    # non_iid_splitter = LDASplitter(client_num, alpha=5)
    # clients_dataidx_map = non_iid_splitter(trainset)
    # splitter = ClientWiseNoisySplitter(2, [1])
    clients_dataidx_map = iid_splitter(trainset)
    with wandb.init(project='debug', config=cfg):
        trainer = LocalSingleGPUTrainer(cfg, model, clients_dataidx_map,
                           trainset, testset)
        trainer.run()
                
                
   
        
    

