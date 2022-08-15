import yaml # for reading the config fil
import wandb
from Model import ConvNet2, ConvNet3, ConvNet4, ConvNet5, ConvNet5
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import Data
import Split
import Trainer


class Simulator:
    def __init__(self, cfg_path):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.project_name = self.cfg['project_name']
        if self.cfg['use_wandb'] == True:
            wandb.login()
            self.logger = wandb.init(project=self.project_name, config=self.cfg)
        else:
            self.logger = None
            
    def run(self):
        model = self.make_model()
        trainset, testset, client_data_idx_map = self.make_data()
        trainer = Trainer.LocalTrainer(
                model, client_data_idx_map, trainset, testset, 
                self.cfg, self.logger)
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
            return ConvNet5(in_channels, h, w, hidden, class_num)
            
            
            
             
    
    def make_data(self, mode=''):
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
        client_data_idx_map = splitter(trainset)
        return trainset, testset, client_data_idx_map


