
import wandb
from Split import ClientWiseNoisySplitter, LDASplitter
from Utils import get_best_gpu
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import tqdm
import FedAvg
import torch
import os
import pandas as pd
from Utils import compute_layerwise_diff, compute_cka_similarity
import copy
import sys

def random_selection(client_ids, num_selected):
    return np.random.choice(client_ids, num_selected, replace=False)

class LocalSingleGPUTrainer:
    def __init__(self, cfg, model, clients_dataidx_map, trainset, testset):
        # self.global_model = model
        self.model = model
        # saving GPU only one memory
        # self.local_model = copy.deepcopy(model)
        self.clients_dataidx_map = clients_dataidx_map
        self.trainset = trainset
        self.testset = testset
        self.cfg = cfg
        if self.cfg['use_gpu'] == True:
            self.gpu = get_best_gpu()
            self.device = torch.device(self.gpu)
        else:
            self.device = torch.device('cpu')
        self.saving_dir = self.cfg['project_name']
        # check contains three directories: local, global, and logs
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)
        self.local_dir = os.path.join(self.saving_dir, 'local')
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        self.global_dir = os.path.join(self.saving_dir, 'global')
        if not os.path.exists(self.global_dir):
            os.makedirs(self.global_dir)
        # Easier for post process    
        self.layerwise_diff_df = []
        self.cka_similarity_df = []
        
    
    
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
            print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, total, 100. * correct / total))
            logging_msg.update({'test_accuracy': 100. * correct / total})
            if self.cfg['corruption_prob'] > 0.0:
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
                    output = self.model(data)
                    pred = output.max(1, keepdim=True)[1]
                    flipped_correct += pred.eq(target.view_as(pred)).sum().item()
                    restored_correct += pred.eq(correct_labels.view_as(pred)).sum().item()
                    total += len(data)
                print(f'\nFlipped set:  Flipped Accuracy: {flipped_correct}/{total} ({100. * flipped_correct / total:.0f}%) Restored Accuracy: {restored_correct}/{total} ({100. * restored_correct / total:.0f}%)\n')
                logging_msg.update({'flipped_accuracy': 100. * flipped_correct / total, 
                           'restored_accuracy': 100. * restored_correct / total
                           })
        return logging_msg
            
                    
                    
             
                    
                
    
    def train(self):
        epoch = self.cfg['epoch']
        batch_size = self.cfg['batch_size']
        lr = self.cfg['lr']
        if 'weight_decay' in self.cfg:
            weight_decay = self.cfg['weight_decay']
        local_epoch = self.cfg['local_round']
        num_clients_per_round = self.cfg['num_clients_per_round']
        print('Training Starts!')
        communication_cost = 0
        for e in range(1, epoch+1):
            print('Saving the first global model...')
            save_path = f'{self.global_dir}/global_model_{e}.pth'
            torch.save(self.model.state_dict(), save_path)
            model_size = os.path.getsize(save_path)/(1024*1024)
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
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
                if self.cfg['optimizer'] == 'Adam':
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                criterion = torch.nn.CrossEntropyLoss()
                if self.cfg['algorithm'] == 'FedAvg':
                    updated_params = FedAvg.local_train(client_id, self.model, 
                                                        global_params,
                                                        train_loader, optimizer, 
                                                        criterion, local_epoch, 
                                                        self.device)
                else:
                    raise ValueError(f'Algorithm {self.cfg["algorithm"]} is not supported!')
                params_dict[client_id] = updated_params
                # print(f'Saving models dictionary cost {sys.getsizeof(params_dict)/(1024*1024)} MB')
                
            torch.save(params_dict, f'{self.local_dir}/local_model_{e}.pth')
            communication_cost += len(selected_ids) * model_size
            logging_msg = {
                'epoch': e+1,
                'communication_cost': communication_cost}
        
            if self.cfg['l2_diff']:
                print('Start calculating layerwise weight difference')
                param_names = [param_name for param_name in
                            self.model.state_dict().keys()]
                columns = ['epoch', 'client_id'] + param_names
                layerwise_diff = []
            
                for client_id, params in params_dict.items():
                    row = list(compute_layerwise_diff(params, global_params).values())
                    record= [e, str(client_id)] + row
                    layerwise_diff.append(row)
                    self.layerwise_diff_df.append(record)
                layerwise_diff = np.array(layerwise_diff)
                if self.cfg['watch_client'] == 'all':
                    mean_diff = np.mean(layerwise_diff, axis=0)
                    std_diff = np.std(layerwise_diff, axis=0)
                    assert len(mean_diff) == len(param_names) and len(std_diff) == len(param_names)
                    msg = { f'diff.mean.{param_names[i]}': 
                        v for i, v in enumerate(mean_diff)}
                    msg.update({f'diff.std.{param_names[i]}': 
                        v for i, v in enumerate(std_diff)})
                    
                elif isinstance(self.cfg['watch_client'], int):
                    client_to_watch = self.cfg['watch_client']
                    remain_clients = np.delete(layerwise_diff, client_to_watch, axis=0)
                    mean_diff = np.mean(remain_clients, axis=0)
                    std_diff = np.std(remain_clients, axis=0)
                    assert len(mean_diff) == len(param_names) and len(std_diff) == len(param_names)
                    msg = {f'diff.mean.{param_names[i]}': 
                        v for i, v in enumerate(mean_diff)}
                    msg.update({f'diff.std.{param_names[i]}': 
                        v for i, v in enumerate(std_diff)})
                    watched_client_diff = layerwise_diff[client_to_watch]
                    print(watched_client_diff)
                    msg.update({f'diff.watched.mean.{param_names[i]}':
                        v for i, v in enumerate(watched_client_diff)})
                logging_msg.update(msg)
    
 
 
            # CKA similarity
            if self.cfg['cka']:
                print('Start calculating layerwise cka similarity')
                cka_similarity = []
                with torch.no_grad():
                    self.model.load_state_dict(global_params)
                    local_model = copy.deepcopy(self.model)
                    comparing_dataset = torch.utils.data.Subset(
                        self.testset,
                        np.random.choice(range(len(self.testset)), 512, replace=False))
                    comparing_loader = torch.utils.data.DataLoader(
                        comparing_dataset, batch_size=128,
                        shuffle=False)
                    for client_id, params in params_dict.items():
                        local_model.load_state_dict(params)
                        result = compute_cka_similarity(
                            self.model, local_model, comparing_loader, self.device)
                        cka_matrix = result['CKA'].numpy()
                        row = cka_matrix.diagonal().tolist()
                        cka_similarity.append(row)
                        record = [e, str(client_id)] + row
                        self.cka_similarity_df.append(record)
                    layer_names = result['model1_layers']
                    similarity = np.array(cka_similarity) 
                    # Problem here
                    if self.cfg['watch_client'] == 'all':
                        mean_sim = np.mean(similarity, axis=0)
                        std_sim = np.std(similarity, axis=0)
                        assert len(mean_diff) == len(param_names) and len(std_diff) == len(param_names)
                        msg = { f'cka.mean.{param_names[i]}': 
                            v for i, v in enumerate(mean_sim)}
                        msg.update({f'cka.std.{param_names[i]}': 
                            v for i, v in enumerate(std_sim)})
                    
                    elif isinstance(self.cfg['watch_client'], int):
                        client_to_watch = self.cfg['watch_client']
                        remain_clients = np.delete(similarity, client_to_watch, axis=0)
                        mean_sim = np.mean(remain_clients, axis=0)
                        std_sim = np.std(remain_clients, axis=0)
                        assert len(mean_diff) == len(param_names) and len(std_diff) == len(param_names)
                        msg = {f'cka.mean.{param_names[i]}': 
                            v for i, v in enumerate(mean_diff)}
                        msg.update({f'cka.std.{param_names[i]}': 
                            v for i, v in enumerate(std_diff)})
                        watched_client_sim = similarity[client_to_watch]
                        print(watched_client_sim)
                        msg.update({f'diff.watched.mean.{param_names[i]}':
                            v for i, v in enumerate(watched_client_diff)})
                    logging_msg.update(msg)
                    
                
            new_global_params = FedAvg.aggregate(params_dict)
            self.model.load_state_dict(new_global_params)
            results = self.evaluate()
            logging_msg.update(results)
            wandb.log(logging_msg)

        diff_df_columns = ['epoch', 'client_id'] + param_names
        cka_df_columns = ['epoch', 'client_id'] + layer_names
        layerwise_diff_df = pd.DataFrame(self.layerwise_diff_df, columns=diff_df_columns)
        cka_similarity_df = pd.DataFrame(self.cka_similarity_df, columns=cka_df_columns)
        layerwise_diff_df.to_csv(f'{self.saving_dir}/layerwise_diff.csv')
        cka_similarity_df.to_csv(f'{self.saving_dir}/cka_similarity.csv')
        torch.save(self.model.state_dict(), f'{self.global_dir}/global_model_final.pth')    
                
if __name__ == "__main__":
    import yaml
    from Split import IIDSplitter
    from Data import load_centralized_dataset
    from Model import ConvNet2, ConvNet5
    from ResNet import ResNet18
    import wandb
    from Data import ImbalancedNoisyDataWrapper
    wandb.login()
    cfg = yaml.safe_load(open('sample.yaml'))
    """
    model = ConvNet2(in_channels=1, h=32, w=32,
                 hidden=2048, class_num=10,  dropout=.0)
    """
    model = ResNet18(input_channels=1)
    dataset_name = cfg['dataset']
    trainset, testset = load_centralized_dataset(
        dataset_name, validation_split=0, download=False)
    cfg['corruption_prob'] = 0.2
    cfg['corruption_type'] = 'uniform'
    trainset = ImbalancedNoisyDataWrapper(
        trainset, corruption_prob=cfg['corruption_prob'],
        imbalanced_ratio=0, num_classes=10, seed=1, noise_type=cfg['corruption_type'])
    client_num = 2
    # iid_splliter = IIDSplitter(client_num)
    # clients_dataidx_map = iid_splliter(trainset)
    # non_iid_splitter = LDASplitter(client_num, alpha=5)
    # clients_dataidx_map = non_iid_splitter(trainset)
    splitter = ClientWiseNoisySplitter(2, [1])
    clients_dataidx_map = splitter(trainset)
    with wandb.init(project='debug', config=cfg):
        trainer = LocalSingleGPUTrainer(cfg, model, clients_dataidx_map,
                           trainset, testset)
        trainer.train()
        trainer.evaluate()
           
                
                
   
        
    

