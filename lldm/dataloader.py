import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import librosa

from lldm.util import ceil_mult8, floor_mult8

device = "cuda" if torch.cuda.is_available() else "cpu"

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

class LLDMDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_i = self.data[idx]
        x_i = torch.tensor(
            librosa.feature.melspectrogram(
                y=x_i["audio"]["array"], 
                sr=x_i["audio"]["sampling_rate"]
            )
        )
        x_i = dynamic_range_compression(x_i)
        x_i = x_i.unsqueeze(0)
            
        return x_i.to(device)

def collate_fn(batch):
    max_len_in_batch = min(864, max(x.size(2) for x in batch))
    padded_batch = []
    
    for x in batch:
        if x.size(2) > 864:
            start_idx = random.randint(0, x.size(2) - 864)
            x = x[:, :, start_idx:start_idx+864]
            max_len_in_batch = 864 # Belum kompatibel, masih kedetek ukuran 864 dan error
        else:
            max_len_in_batch = int(ceil_mult8(max_len_in_batch))
            
        padded_x = torch.zeros(x.size(0), x.size(1), max_len_in_batch)
        padded_x[:, :, :x.size(2)] = x[:, :, :max_len_in_batch]
        
        padded_batch.append(padded_x)
    
    return torch.stack(padded_batch).to(device)

