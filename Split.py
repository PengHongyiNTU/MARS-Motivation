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
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]
            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice


class LDASplitter:
    def __init__(self, client_num, alpha=0.5):
        self.client_num = client_num
        self.alpha = alpha

    def __call__(self, dataset):
        dataset = [ds for ds in dataset]
        label = np.array([y for x, y in dataset])
        idx_slice = dirichlet_distribution_noniid_slice(
            label, self.client_num, self.alpha)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}(client_num={self.client_num}, alpha={self.alpha})'


class IIDSplitter:
    def __init__(self, client_num):
        self.client_num = client_num

    def __call__(self, dataset):
        dataset = [ds for ds in dataset]
        data_list = [dataset[i::self.client_num] for i in range(self.client_num)]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}(client_num={self.client_num})'