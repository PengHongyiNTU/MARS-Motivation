import numpy as np
from torch.utils.data import Dataset


def uniform_mix_C(mixing_ratio, num_classes):
    """
    returns a linear interpolation of a uniform matrix and an identity matrix
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
           (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C


class MyDataSetNoisyWrapper(Dataset):
    def __init__(self, base_dataset, corruption_prob, num_classes, seed=1, noise_type='uniform'):
        self.base_dataset = base_dataset
        self.corruption_prob = corruption_prob
        self.num_classes = num_classes
        self.seed = seed
        self.noise_type = noise_type
        if noise_type == 'uniform':
            self.C = uniform_mix_C(self.corruption_prob, self.num_classes)
        else:
            if noise_type == 'diag-1':
                corruption = flip_labels_C
            elif noise_type == 'diag-2':
                corruption = flip_labels_C_two
            self.C = corruption(self.corruption_prob, self.num_classes, self.seed)

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
