from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import torch

class DiskDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, x_shape: Tuple, y_shape: Tuple, x_dtype=np.float32, y_dtype=np.float32):
        self.x_mem = np.memmap(x_path, mode='r', shape=x_shape, dtype=x_dtype)
        self.y_mem = np.memmap(y_path, mode='r', shape=y_shape, dtype=y_dtype)
        self.length = x_shape[0]
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return torch.tensor(self.x_mem[idx]), torch.tensor(self.y_mem[idx])
    
