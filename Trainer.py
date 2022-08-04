
import wandb
from Utils import get_best_gpu
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import tqdm
import FedAvg
import torch
import os
from Utils import layerwise_diff

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
        columns = ['epoch', 'id']
        param_names = [param_name for param_name in 
                       self.global_model.state_dict().keys()]
        columns += param_names
        first_row = [0, 'global'] + [0]*len(param_names)
        self.table = wandb.Table(columns=columns, data=[first_row])
        
                          
    
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
            selected_avg_loss = [0]*len(selected_ids)
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
                
                updated_params, loss = FedAvg.local_train(id, local_model, 
                                                    train_loader, 
                                   optimizer, criterion, local_epoch, 
                                   self.device)
                params_dict[id] = updated_params
                selected_avg_loss.append(loss)
            torch.save(params_dict, f'{self.local_dir}/local_model_{e}.pth')
            communication_cost += len(selected_ids) * model_size
            wandb.log({'Communication Cost': communication_cost})
            logging_msg = { f'Selected Clients {i}': selected_avg_loss[i]
                for i in range(len(selected_avg_loss))
            }
            wandb.log(logging_msg)
            for id, params in params_dict.items():
                row = list(layerwise_diff(params, global_model_params).values())
                row = [e, id] + row
                self.table.add_data(*row)
            wandb.log({'layerwise_diff': self.table})  
            global_params = FedAvg.aggregate(params_dict)
            self.global_model.load_state_dict(global_params)
            self.evaluate()
        torch.save(self.global_model.state_dict(), f'{self.global_dir}/global_model_final.pth')    
                
if __name__ == "__main__":
    import yaml
    from Trainer import LocalTrainer
    from Split import IIDSplitter
    from Data import load_centralized_dataset
    from Model import ConvNet2
    import wandb
    from Data import ImbalancedNoisyDataWrapper
    wandb.login()
    cfg = yaml.safe_load(open('sample.yaml'))
    model = ConvNet2(in_channels=1, h=28, w=28,
                 hidden=2048, class_num=10,  dropout=.0)
    dataset_name = cfg['dataset']
    trainset, testset = load_centralized_dataset(
    dataset_name, validation_split=0, download=False)
    trainset = ImbalancedNoisyDataWrapper(
        trainset, corruption_prob=cfg['corruption_prob'],
        imblanced_ratio=0, num_classes=10, seed=1, noise_type='uniform')
    client_num = 10
    iid_splliter = IIDSplitter(client_num)
    clients_dataidx_map = iid_splliter(trainset)
    with wandb.init(project='debug', config=cfg):
        trainer = LocalTrainer(model, clients_dataidx_map,
                           trainset, testset, cfg)
        trainer.train()
        trainer.evaluate()
           
                
                
   
        
    

