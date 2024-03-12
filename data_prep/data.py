import random
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import BATCH_SIZE

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


def build_dataset():
    indices = np.load(INDICES_PATH)
    indices = np.array(indices, dtype=np.float64)
    tensors = torch.Tensor(indices, dtype=torch.int64)
    dataset = TensorDataset(tensors)
    return dataset