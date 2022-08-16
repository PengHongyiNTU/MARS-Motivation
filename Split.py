from Data import ImbalancedNoisyDataWrapper
import numpy as np

def dirichlet_distribution_noniid_slice(label, client_num, alpha, min_size=10):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid_partition/noniid_partition.py

    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')
    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f'The number of sample should be greater than {client_num * min_size}.'
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            prop = np.array([
                p * (len(idx_j) < num / client_num)
                for p, idx_j in zip(prop, idx_slice)
            ])
            prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.array_split(idx_k, prop))
            ]
            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice


class LDASplitter:
    def __init__(self, client_num, alpha=0.5):
        self.client_num = client_num
        self.alpha = alpha
        self.clients_dataidx_map = dict.fromkeys(
            list(range(client_num))
        )

    def __call__(self, dataset):
        label = np.array([y for x, y in dataset])
        idxs_slice = dirichlet_distribution_noniid_slice(
            label, self.client_num, self.alpha)
        for i, idx in enumerate(idxs_slice):
            self.clients_dataidx_map[i] = idx
        print('Splitting dataset into {} clients.'.format(self.client_num))
        print({id: len(idxs) for id, idxs in self.clients_dataidx_map.items()})
        return self.clients_dataidx_map
        
        
    def __repr__(self):
        return f'{self.__class__.__name__}(client_num={self.client_num}, alpha={self.alpha})'


class IIDSplitter:
    def __init__(self, client_num):
        self.client_num = client_num
        self.clients_dataidx_map = dict.fromkeys(
            list(range(client_num))
        )

    def __call__(self, dataset):
        idxs = np.arange(len(dataset))
        idxs_slice = np.array_split(idxs, self.client_num)
        for i, idx in enumerate(idxs_slice):
            self.clients_dataidx_map[i] = idx
        print('Splitting dataset into {} clients.'.format(self.client_num))
        print({id: len(idxs) for id, idxs in self.clients_dataidx_map.items()})
        return self.clients_dataidx_map

    def __repr__(self):
        return f'{self.__class__.__name__}(client_num={self.client_num})'
    
    
class ClientWiseNoisySplitter:
    def __init__(self, client_num, noisy_client_id):
        self.client_num = client_num
        self.noisy_client_id = noisy_client_id
        self.clients_dataidx_map = dict.fromkeys(list(range(client_num)))
        
    def __repr__(self):
        return f'{self.__class__.__name__}(client_num={self.client_num}, Noisy Client ID={self.noisy_client_id})'
    
    def __call__(self, dataset):
        # check dataset is a instance of ImbalancedNoisyWrapper
        if isinstance(dataset, ImbalancedNoisyDataWrapper):
            if dataset.corruption_prob == 0:
                raise ValueError('raw dataset does not have any noise')
            else:
                idxs = np.arange(len(dataset))
                is_filpped = dataset.is_flipped
                clean_idxs = idxs[is_filpped==0]
                noisy_idxs = idxs[is_filpped==1]
                idxs_slice = np.array_split(clean_idxs, self.client_num)
                for i, idx in enumerate(idxs_slice):
                    self.clients_dataidx_map[i] = idx
                self.clients_dataidx_map[self.noisy_client_id] = np.append(self.clients_dataidx_map[self.noisy_client_id], 
                                                                           noisy_idxs, axis=0)
                print('Splitting dataset into {} clients.'.format(self.client_num))
                print({id: len(idxs) for id, idxs in self.clients_dataidx_map.items()})
                print(f'Client {self.noisy_client_id} has {len(noisy_idxs)} noisy samples')
                return self.clients_dataidx_map
                
                
    
class ClientWiseImbalanceSplitter:
    def __init__(self, client_num, imbalanced_client_id, imbalanced_ratio=3):
        self.client_num = client_num 
        self.imbalanced_client_id = imbalanced_client_id
        self.clients_dataidx_map = dict.fromkeys(list(range(client_num)))
    def __repr__(self):
        return f'{self.__class__.__name__}(client_num={self.client_num}, Imbalanced Client ID={self.imbalanced_client_id})'
    def __call__(self, dataset):
        if not isinstance(dataset, ImbalancedNoisyDataWrapper):
            raise ValueError('dataset is not a instance of ImbalancedNoisyDataWrapper')
        else: 
            # Initialize the data index map
            idxs = np.arange(len(dataset))
            idxs_slice = np.array_split(idxs, self.client_num)
            for i, idx in enumerate(idxs_slice):
                self.clients_dataidx_map[i] = idx
            idx = self.clients_dataidx_map[self.imbalanced_client_id]
            labels = dataset.correct_labels[idx]
            class_idx_map = dict.fromkeys(np.unique(labels))
            for key in class_idx_map:
                class_idx_map[key] = np.where(labels==key)[0]
            selected_idx = []
            if dataset.imbalanced_ratio > 0:
                portion_per_class = dataset.portion_per_class[::-1]
            else:
                portion_per_class = np.exp(self.imbalanced_ratio * 
                                           np.linspace(0, 1, num=dataset.num_classes))
            selected_idxs = []       
            for i, key in enumerate(class_idx_map.keys()):
                num_samples = len(class_idx_map[key])-1
                num_samples = int(num_samples * portion_per_class[i])
                selected_idx = np.random.choice(class_idx_map[key], num_samples, replace=False)
                selected_idxs.append(selected_idx)
            selected_idx = np.concatenate(selected_idxs)
            clients_dataidx_map[self.imbalanced_client_id] = selected_idx
            print('Splitting dataset into {} clients.'.format(self.client_num))
            print(f'Global portion per class: {dataset.portion_per_class}')
            print(f'Client {self.imbalanced_client_id} portion per class: {portion_per_class}')
            return self.clients_dataidx_map
            
                
            
                
                
                
            
    
    
if __name__ == "__main__":
    from Data import load_centralized_dataset, ImbalancedNoisyDataWrapper
    train_set, test_set = load_centralized_dataset(
        name='MNIST', validation_split=0, 
    )
    client_num = 10
    alpha = 0.5
    niid_splitter = LDASplitter(client_num, alpha)
    iid_splitter = IIDSplitter(client_num)
    clients_dataidx_map = niid_splitter(train_set)
    clients_dataidx_map = iid_splitter(train_set)
    flipped_imnbalanced_dataset = ImbalancedNoisyDataWrapper(
            base_dataset=train_set,
            corruption_prob=0.3,
            imbalanced_ratio=3,
            num_classes=10)
    client_noise_splitter = ClientWiseNoisySplitter(client_num, noisy_client_id=0)
    clients_dataidx_map = client_noise_splitter(flipped_imnbalanced_dataset)
    client_imbalanced_splitter = ClientWiseImbalanceSplitter(client_num, imbalanced_client_id=0)
    clients_dataidx_map = client_imbalanced_splitter(flipped_imnbalanced_dataset)
    
   
