import yaml # for reading the config fil
import wandb
from Model import ConvNet2, ConvNet5
import Data
import Split
import Trainer
class Simulator:
    def __init__(self, cfg_path, project_name):
        self.project_name = project_name 
        self.cfg = yaml.safe_load(open(cfg_path))
        self.init_wandb()
        
    
    def run(self):
        model = self.make_model()
        trainset, testset, client_data_idx_map = self.make_data()
        with wandb.init(project=self.project_name, config=self.cfg):
            trainer = Trainer.LocalTrainer(
                model, client_data_idx_map, trainset, testset, 
                self.cfg)
            trainer.train()
            trainer.evaluate()                   
    
    def init_wandb(self):
        wandb.login()
        
    
    def make_model(self):
        dataset_name = self.cfg['dataset']
        model = self.cfg['model']
        if dataset_name == "MNIST":
            in_channels = 1
            h = 28
            w = 28
            hidden=1024,
            class_num = 10
            if model == "ConvNet2":
                model = ConvNet2(in_channels, h, w, class_num=class_num)
            elif model == "ConvNet5":
                model = ConvNet5(in_channels, h, w, classs_num=class_num)
            else:
                raise ValueError("model on MNIST must be ConvNet2 or ConvNet5")            
        return model
    
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


