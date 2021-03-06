from cgi import test
from email.mime import base
from matplotlib.pyplot import axes
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
import torch
from Noise import uniform_mix_C, flip_labels_C, flip_labels_C_two
from torch.utils.data import Dataset
import numpy as np


def load_centralized_dataset(name='MNIST', validation_split=0, download=False):
    if name == 'MNIST':
        train_dataset = MNIST(root='./data', train=True, download=download)
        test_dataset = MNIST(root='./data', train=False, download=download)
    if name == "CIFAR10":
        # use torch vision to load CIFAR10  
        train_dataset = CIFAR10(root='./data', train=True, download=download)
        test_dataset = CIFAR10(root='./data', train=False, download=download)
    if validation_split > 0:
        if isinstance(validation_split, float):
            val_size = int(validation_split* len(train_dataset))
        else:
            val_size = validation_split
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_size, val_size])
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset
    


class ImbalancedNoisyDataWrapper(Dataset):
    def __init__(self, base_dataset, corruption_prob, imblanced_ratio, num_classes, seed=1, noise_type='uniform'):
        self.base_dataset = base_dataset
        self.dataset = self.base_dataset
        self.corruption_prob = corruption_prob
        self.imblanced_ratio = imblanced_ratio
        self.num_classes = num_classes
        self.seed = seed
        self.noise_type = noise_type
        self.portion_per_class = np.exp(-self.imblanced_ratio * np.linspace(0, 1, num=self.num_classes))
        self.C = np.eye(num_classes)
        # random generator for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


        if self.imblanced_ratio > 0:
            self._downsample_datasets()
        self.correct_labels = np.array([labels for _, labels in self.dataset])
        self.flipped_labels = self.correct_labels.copy()
        self.is_flipped = np.zeros_like(self.correct_labels)
        if self.corruption_prob > 0:
            self._generate_label_noise()
       

                
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
            for i in range(len(self.dataset)):
                y_corrupted = np.random.choice(np.arange(0, self.num_classes), p=self.C[self.correct_labels[i]])
                if y_corrupted == self.correct_labels[i]:
                    self.is_flipped[i] = 0
                else:
                    self.is_flipped[i] = 1
                self.flipped_labels[i] = y_corrupted
                
                
    def get_restored_dataset(self):
        # To compute restored gradient 
        return torch.utils.data.Subset(self.dataset, np.where(self.is_flipped == 0)[0])
    
    def _downsample_datasets(self):
        # downsample the dataset according to the imbalanced ratio
        raw_labels = np.array([labels for _, labels in self.base_dataset])
        class_idx_map = dict.fromkeys(np.unique(raw_labels))
        for key in class_idx_map:
            class_idx_map[key] = np.where(raw_labels == key)[0]
        selected_idxs = []        
        # print(class_idx_map)
        for i, key in enumerate(class_idx_map.keys()):
            num_samples_per_class = len(class_idx_map[key])-1
            #print(num_samples_per_class)
            #print(self.portion_per_class[i])
            num_samples_to_select = int(num_samples_per_class * self.portion_per_class[i])
            #print(class_idx_map[key])
            print(num_samples_to_select)
            selected_idx = np.random.choice(class_idx_map[key], num_samples_to_select, replace=False)
            selected_idxs.append(selected_idx)
        selected_idxs = np.concatenate(selected_idxs)
        # downsanmple the dataset
        self.dataset = torch.utils.data.Subset(self.base_dataset, selected_idxs)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        y_corrupted = self.flipped_labels[index]
        return x, y_corrupted

    def __repr__(self):
        output = f'Dataset: {self.base_dataset.__class__.__name__}\n'
        output += f'Noise Type : {self.noise_type}, Corruption Probability : {self.corruption_prob}, Corruption Matrix : {self.C}'
        output += f'\nNumber of classes : {self.num_classes}, Portion per class : {self.portion_per_class}'
        return output
    

if __name__ == '__main__':
    raw_dataset, _ = load_centralized_dataset(
        name='MNIST', validation_split=0, download=True)
    balanced_dataset = ImbalancedNoisyDataWrapper(
        base_dataset=raw_dataset,
        corruption_prob=0,
        imblanced_ratio=0,
        num_classes=10)
    imbalanced_dataset = ImbalancedNoisyDataWrapper(
        base_dataset=raw_dataset,
        corruption_prob=0,
        imblanced_ratio=3,
        num_classes=10)
    flipped_imnbalanced_dataset = ImbalancedNoisyDataWrapper(
        base_dataset=raw_dataset,
        corruption_prob=0.3,
        imblanced_ratio=3,
        num_classes=10)
    print(balanced_dataset)
    print(imbalanced_dataset)
    print(flipped_imnbalanced_dataset)
    ## Plot histogram of the labels
    raw_labels = np.array([labels for _, labels in balanced_dataset])
    imbalanced_labels = np.array([labels for _, labels in imbalanced_dataset])
    flipped_imnbalanced_labels = np.array([labels for _, labels in flipped_imnbalanced_dataset])
    import matplotlib.pyplot as plt 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(raw_labels, bins=10)
    axes[0].set_title('Raw')
    axes[1].hist(imbalanced_labels, bins=10)
    axes[1].set_title('Imbalanced')
    axes[2].hist(flipped_imnbalanced_labels, bins=10)
    axes[2].set_title('Flipped Imbalanced')
    plt.savefig('DataDistribution.png')
    
    
        