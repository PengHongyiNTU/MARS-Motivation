import yaml # for reading the config fil
import wandb
from Model import ConvNet2, ConvNet3, ConvNet4, ConvNet5, ConvNet5
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import Data
import Split
import Trainer
import torch
import numpy as np

class Simulator:
    def __init__(self, cfg=None, cfg_path=None):
        if cfg_path:
            self.cfg = yaml.safe_load(open(cfg_path))
        elif cfg:
            self.cfg = cfg
        self.project_name = self.cfg['project_name']
        wandb.login()
        # Reproducibility
        torch.manual_seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])
            
    def run(self):
        model = self.make_model()
        trainset, testset, client_data_idx_map = self.make_data()
        trainer = Trainer.LocalSingleGPUTrainer(self.cfg,
            model, client_data_idx_map, trainset, testset)
        with wandb.init(project=self.project_name, config=self.cfg):
            trainer.run()          
           
    
    def make_model(self):
        dataset_name = self.cfg['dataset']
        model = self.cfg['model']
        h = 32
        w = 32
        hidden = 2048
        if dataset_name == "MNIST":
            in_channels = 1
            class_num = 10
        elif dataset_name == "CIFAR10":
            in_channels = 3
            class_num = 10
        elif dataset_name == "CIFAR100":
            in_channels = 3
            class_num = 100
        elif dataset_name == "TinyImageNet":
            in_channels = 3
            class_num = 200
        
        if model == "ConvNet2":
            return ConvNet2(in_channels, h, w, hidden, class_num)
        elif model == "ConvNet3":
            return ConvNet3(in_channels, h, w, hidden, class_num)
        elif model == "ConvNet4":
            return ConvNet4(in_channels, h, w, hidden, class_num)
        elif model == "ConvNet5":
            return ConvNet5(in_channels, h, w, hidden, class_num)
        elif model == "ResNet18":
            return ResNet18(in_channels, class_num)
        elif model == "ResNet34":
            return ResNet34(in_channels, class_num)
        elif model == "ResNet50":
            return ResNet50(in_channels, class_num)
        elif model == "ResNet101":
            return ResNet101(in_channels, class_num)
        elif model == "ResNet152":
            return ResNet152(in_channels, class_num)
            
            
            
             
    
    def make_data(self):
        dataset_name = self.cfg['dataset']
        trainset, testset = Data.load_centralized_dataset(
            dataset_name, validation_split=0, download=False)
        if dataset_name == "TinyImageNet":
            num_classes = 200
        elif dataset_name == 'CIFAR100':
            num_classes = 100
        else:    
            num_classes = 10
        trainset = Data.ImbalancedNoisyDataWrapper(
            trainset, corruption_prob=self.cfg['corruption_prob'],
            imbalanced_ratio=self.cfg['imbalanced_ratio'],
            num_classes=num_classes, noise_type=self.cfg['corruption_type'])
        client_num = self.cfg['num_clients']
        if self.cfg['split'] == "iid":
            splitter = Split.IIDSplitter(client_num)
        elif self.cfg['split'] == "niid":
            alpha = self.cfg['alpha']
            splitter = Split.LDASplitter(client_num, alpha)
        elif self.cfg['split'] == 'noisy_clients':
            assert(self.cfg['num_noisy_clients'])
            num_noisy_clients = self.cfg['num_noisy_clients']
            noisy_clients_ids = list(range(client_num))[:-num_noisy_clients]
            splitter = Split.ClientWiseNoisySplitter(
                client_num, noisy_clients_ids)
        elif self.cfg['split'] == 'imbalanced_clients':
            assert(self.cfg['num_imbalanced_clients'])
            num_imbalanced_clients = self.cfg['num_imbalanced_clients']
            imblanced_clients_ids = list(range(client_num))[:-num_imbalanced_clients]
            splitter = Split.ClientWiseImbalancedSplitter(
                client_num, imblanced_clients_ids)  
        client_data_idx_map = splitter(trainset)
        return trainset, testset, client_data_idx_map


