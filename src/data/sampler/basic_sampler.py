import torch
from torch.utils import data

class FixedSampler(data.Sampler):
    '''
    A fixed-length sampler for SR dataset.

    Args:
        data_source (Dataset):
        batch_size (int):
        test_every (int):

    Note:
        1 Epoch is defined as test_every number of updates.
    '''

    def __init__(self, data_source, batch_size, test_every):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = max(len(data_source), batch_size * test_every)
        # To avoid a incomplete batch
        self.num_samples = (self.num_samples // batch_size) * batch_size

    def __iter__(self):
        mod_idx = torch.randperm(self.num_samples) % len(self.data_source)
        return iter(mod_idx.tolist())

    def __len__(self):
        return self.num_samples
