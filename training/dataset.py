from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import librosa
import torch

from models.utils import ceil_mult8

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            
        return x_i.to(device), sr
