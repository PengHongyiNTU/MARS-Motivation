from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
from Noise import uniform_mix_C, flip_labels_C, flip_labels_C_two
from torch.utils.data import Dataset
import numpy as np


def load_dataset(self, name='MNIST', validation_split=0):
    if name == 'MNIST':
        train_dataset = MNIST(root='./data', train=True, download=False)
        test_dataset = MNIST(root='./data', train=False, download=False)
    if name == "CIFAR10":
        pass
    if name == "FEMNIST":
        pass
    if validation_split > 0:
        if isinstance(self.val_size, float):
            val_size = int(self.val_size * len(train_dataset))
        else:
            val_size = self.val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_size, val_size])
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset
    


class ImbalancedNoisyDataWrapper(Dataset):
    def __init__(self, base_dataset, corruption_prob, imblanced_ratio, num_classes, seed=1, noise_type='uniform'):
        self.base_dataset = base_dataset
        self.corruption_prob = corruption_prob
        self.imblanced_ratio = imblanced_ratio
        self.num_classes = num_classes
        self.seed = seed
        self.noise_type = noise_type
        self.correct_labels = np.array([labels for x, labels in self.base_dataset])
        self.flipped_labels = self.correct_labels.copy()
        self.is_flipped = np.zeros_like(self.correct_labels)


        # random generator for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.corruption_prob > 0:
            self._generate_label_noise()
        if self.imblanced_ratio > 0:
            self._downsample_datasets()

                
    def _generate_label_noise(self):
        if self.corruption_prob > 0:
            if self.noise_type == 'uniform':
                self.C = uniform_mix_C(self.corruption_prob, self.num_classes)
            else:
                if self.noise_type == 'target-1':
                    corruption = flip_labels_C
                elif self.noise_type == 'target-2':
                    corruption = flip_labels_C_two
                self.C = corruption(self.corruption_prob, self.num_classes, self.seed)
            for i in range(len(self.base_dataset)):
                y_corrupted = np.random.choice(np.arange(0, self.num_classes), p=self.C[self.correct_labels[i]])
                if y_corrupted == self.correct_labels[i]:
                    self.is_flipped[i] = 0
                else:
                    self.is_flipped[i] = 1
                self.flipped_labels[i] = y_corrupted
    
    def _downsample_datasets(self):
        # downsample the dataset according to the imbalanced ratio
        idx_class_map = dict.fromkeys(np.unique(self.correct_labels))
        for key in idx_class_map:
            idx_class_map[key] = np.where(self.correct_labels == key)
        selected_idxs = []
        x = np.linspace(0, 1, num=self.num_classes)
        number_sample_per_class = int(len(self.base_dataset) * 
                                      self.imblanced_ratio)
            
        self.dataset = torch.utils.data.Subset(self.base_dataset, idxs)
                                                         


        

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        y_corrupted = np.random.choice(np.arange(0, self.num_classes), p=self.C[y])
        if y_corrupted == y:
            flag = 0
        else:
            flag = 1
        return x, y_corrupted, flag

    def __repr__(self):
        return f'Noise Type : {self.noise_type}, Corruption Probability : {self.corruption_prob}, Corruption Matrix : {self.C}'
    

if __name__ == '__main__':
    pass
   