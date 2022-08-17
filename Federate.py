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
    def __init__(self, cfg_path):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.project_name = self.cfg['project_name']
        wandb.login()
        # Reproducibility
        torch.manual_seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])
            
    def run(self):
        model = self.make_model()
        trainset, testset, client_data_idx_map = self.make_data()
        if self.cfg['multigpu'] == True:
            trainer = Trainer.MultiGPUTrainer(model, trainset, testset, client_data_idx_map, self.cfg)
        else:
            trainer = Trainer.LocalTrainer(
                model, client_data_idx_map, trainset, testset, 
                self.cfg)
        with wandb.init(project=self.project_name, config=self.cfg):
            trainer.train()
            trainer.evaluate()                   
           
    
    def make_model(self):
        dataset_name = self.cfg['dataset']
        model = self.cfg['model']
        h = 32
        w = 32
        hidden = 2048
        class_num = 10
        if dataset_name == "MNIST":
            in_channels = 1
        elif dataset_name == "CIFAR10":
            in_channels = 3
        if model == "ConvNet2":
            return ConvNet2(in_channels, h, w, hidden, class_num)
        elif model == "ConvNet3":
            return ConvNet3(in_channels, h, w, hidden, class_num)
        elif model == "ConvNet4":
            return ConvNet4(in_channels, h, w, hidden, class_num)
        elif model == "ConvNet5":
            return ConvNet5(in_channels, h, w, hidden, class_num)
        elif model == "ResNet18":
            return ResNet18()
        elif model == "ResNet34":
            return ResNet34()
        elif model == "ResNet50":
            return ResNet50()
        elif model == "ResNet101":
            return ResNet101()
        elif model == "ResNet152":
            return ResNet152()
            
            
            
             
    
    def make_data(self):
        dataset_name = self.cfg['dataset']
        trainset, testset = Data.load_centralized_dataset(
            dataset_name, validation_split=0, download=False)
        trainset = Data.ImbalancedNoisyDataWrapper(
            trainset, corruption_prob=self.cfg['corruption_prob'],
            imblanced_ratio=self.cfg['imbalanced_ratio'],
            num_classes=10)
        client_num = self.cfg['num_clients']
        if self.cfg['split'] == "iid":
            splitter = Split.IIDSplitter(client_num)
        elif self.cfg['split'] == "niid":
            alpha = self.cfg['alpha']
            splitter = Split.LDASplitter(client_num, alpha)
        elif self.cfg['split'] == 'local noise':
            noisy_client_id = client_num - 1
            splitter = Split.ClientWiseNoisySplitter(
                client_num, noisy_client_id)
        elif self.cfg['split'] == 'local imbalance':
            imbalanced_client_id = client_num -1
            splitter = Split.ClientWiseImbalancedSplitter(
                client_num, imbalanced_client_id)  
        client_data_idx_map = splitter(trainset)
        return trainset, testset, client_data_idx_map


