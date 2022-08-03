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
