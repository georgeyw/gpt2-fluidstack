import random
import os

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from .constants import BATCH_SIZE

DIR = os.path.dirname(os.path.abspath(__file__))
INDICES_PATH = os.path.join(DIR, '../token_indices/', 'indicies.npy')

# # code from https://pytorch.org/docs/stable/notes/randomness.html
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# g = torch.Generator()
# g.manual_seed(0)

# def build_dataloader(batch_size=BATCH_SIZE):
#     indices = np.load(INDICES_PATH)
#     indices = np.array(indices, dtype=np.float64)
#     tensors = torch.Tensor(indices, dtype=torch.int64)
#     dataset = TensorDataset(tensors)
#     dataloader = DataLoader(dataset, 
#                             batch_size=batch_size, 
#                             num_workers=8,
#                             worker_init_fn=seed_worker,
#                             generator=g,
#                             shuffle=False)
#     return dataloader

class NPYDataset(Dataset):
    def __init__(self, file_path, length=None):
        self.file_path = file_path
        self.data_shape = np.load(file_path, mmap_mode='r').shape
        self.length = length

    def __len__(self):
        if self.length is not None:
            return self.length
        return self.data_shape[0]

    def __getitem__(self, idx):
        sample = np.load(self.file_path, mmap_mode='r')[idx]
        sample = np.array(sample, dtype=np.int64)
        tensor = torch.from_numpy(sample)
        tensor = tensor.to(torch.int64)
        return tensor


# def build_dataset():
#     indices = np.load(INDICES_PATH)
#     indices = np.array(indices, dtype=np.int64)
#     indices = indices[:800_000 * 256]  # magic numbers
#     tensors = torch.Tensor(indices, device='cpu')
#     tensors = tensors.to(torch.int64)
#     dataset = TensorDataset(tensors)
#     return dataset