from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import librosa
import torch

from models.utils import ceil_mult8

device = "cuda" if torch.cuda.is_available() else "cpu"

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

class MyDataset(Dataset):
    def __init__(
        self, data
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_i = torch.tensor(self.data[idx]["audio"]["array"]).unsqueeze(0).float()
        sr = self.data[idx]["audio"]["sampling_rate"]

        if x_i.shape[1] < 22600:
            pad = torch.zeros(x_i.shape[0], 22600)
            pad[:x_i.shape[0], :x_i.shape[1]] = sample_input
            x_i = pad
            
        return x_i.to(device), sr
