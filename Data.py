from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
from Noise import MyDataSetNoisyWrapper

class MNISTDatasets:
    def __init__(self, num_clients, alpha, corruption_prob, noise_type, val_size=10000):
        self.num_clients = num_clients
        self.alpha = alpha
        self.corruption_prob = corruption_prob
        self.noise_type = noise_type
        self.val_size = val_size

    def _load_dataset(self):
        train_dataset = MNIST(root='./data', train=True, download=True)
        test_dataset = MNIST(root='./data', train=False, download=True)
        if isinstance(self.val_size, int) and 0.0 < self.val_size < 1:
            val_size = int(self.val_size * len(train_dataset))
        else:
            val_size = self.val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_size, val_size])
        return train_dataset, val_dataset, test_dataset

    def _add_noise(self, train_dataset):
        if self.noise_type == 'uniform':
            C = uniform_mix_C(self.corruption_prob, self.num_clients)