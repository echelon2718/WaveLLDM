import os
import librosa
from torch.utils.data import Dataset, DataLoader
from models.utils import LogMelSpectrogram, count_parameters, get_padding_sample
import torch
import random
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample

def random_cut(audio, max_cuts=5, cut_duration=0.1, sample_rate=16000):
    """
    Membuat beberapa bagian audio menjadi putus-putus dengan mengganti segmen tertentu menjadi nol.

    Args:
        audio (np.array): Array numpy dari audio
        max_cuts (int): Jumlah maksimal potongan yang akan dilakukan
        cut_duration (float): Lama waktu setiap potongan dalam detik
        sample_rate (int): Nilai sample rate audio

    Returns:
        np.ndarray: Audio yang sudah dipotong-potong
    """
    audio_length = audio.shape[-1]
    cut_samples = int(cut_duration * sample_rate)

    num_cuts = torch.randint(1, max_cuts + 1, (1,)).item()
    starts = torch.randint(0, audio_length - cut_samples + 1, (num_cuts,))
    mask = torch.ones_like(audio)
    for start in starts:
        mask[..., start:start + cut_samples] = 0
        
    return audio * mask

def vanilla_collate_fn(batch): # Hanya berlaku untuk fixed length.
    clean_audios = [item['clean_audio'] for item in batch]
    noisy_audios = [item['noisy_audio'] for item in batch]

    clean_audios = torch.stack(clean_audios)
    noisy_audios = torch.stack(noisy_audios)

    return {
        "clean_audios": clean_audios,
        "noisy_audios": noisy_audios
    }

class DenoiserDataset(Dataset):
    def __init__(
        self, 
        clean_dir: str, 
        noisy_dir: str,
        add_random_cutting: bool = False,
        stage: int = 1,
        max_cuts: int = 5,
        cut_duration: float = 0.35,
        fixed_length: int = None,  # Baru: panjang sampel tetap (dalam jumlah sample)
        device: str = "cpu"
    ):
        self.clean_files = sorted(os.listdir(clean_dir))
        self.noisy_files = sorted(os.listdir(noisy_dir))
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.add_random_cutting = add_random_cutting
        self.stage = stage
        self.max_cuts = max_cuts
        self.cut_duration = cut_duration
        self.fixed_length = fixed_length
        self.device = device
    
    def __getitem__(self, idx):
        # Load audio clean
        clean_audio, clean_sr = torchaudio.load(os.path.join(self.clean_dir, self.clean_files[idx]))
        if clean_audio.size(0) > 1:
            clean_audio = clean_audio.mean(dim=0, keepdim=True) 

        # Pilih sample rate acak untuk audio noisy agar bervariasi, lalu resample ke clean_sr
        # noisy_sr = int(clean_sr / (torch.randint(1, 3, (1,)).item() + 1))
        noisy_sr = 16_000 # Set ke 16.000, menurunkan variasi data mempercepat konvergensi.
        noisy_audio, orig_noisy_sr = torchaudio.load(os.path.join(self.noisy_dir, self.noisy_files[idx]))
        if noisy_audio.size(0) > 1:
            noisy_audio = noisy_audio.mean(dim=0, keepdim=True)  

        # Resample noisy audio ke clean_sr [BARU]
        if orig_noisy_sr != noisy_sr:
            resampler = Resample(orig_freq=orig_noisy_sr, new_freq=noisy_sr)
            noisy_audio = resampler(noisy_audio)
        if noisy_sr != clean_sr:
            resampler = Resample(orig_freq=noisy_sr, new_freq=clean_sr)
            noisy_audio = resampler(noisy_audio)
        noisy_sr_new = clean_sr

        # Tambah random cut jika diaktifkan
        if self.add_random_cutting:
            noisy_audio = random_cut(noisy_audio, max_cuts=self.max_cuts, cut_duration=self.cut_duration, sample_rate=clean_sr)

        # Jika fixed_length diset, crop atau pad audio supaya panjangnya seragam
        assert self.fixed_length is not None, "Please define fixed audio length in samples"
        min_len = min(clean_audio.size(-1), noisy_audio.size(-1))
        clean_audio = clean_audio[..., :min_len]
        noisy_audio = noisy_audio[..., :min_len]
        
        # 2. Crop atau pad ke fixed_length
        if min_len > self.fixed_length:
            max_offset = min_len - self.fixed_length
            offset = torch.randint(0, max_offset + 1, (1,)).item()
            clean_audio = clean_audio[..., offset:offset + self.fixed_length]
            noisy_audio = noisy_audio[..., offset:offset + self.fixed_length]
        else:
            pad_amount = self.fixed_length - min_len
            clean_audio = F.pad(clean_audio, (0, pad_amount))
            noisy_audio = F.pad(noisy_audio, (0, pad_amount))

        item = {
            "clean_audio": clean_audio,
            "noisy_audio": noisy_audio,
            "clean_sr": clean_sr,
            "noisy_sr": noisy_sr_new
        }

        return item    

    def __len__(self):
        return len(self.clean_files)

class LatentDataset(Dataset): # OFFLINE TRAINING (NOT RECOMMENDED)
    def __init__(self, latent_clean_dir, latent_noisy_dir):
        self.latent_clean_dir = latent_clean_dir
        self.latent_noisy_dir = latent_noisy_dir
        self.latent_clean_files = sorted(os.listdir(latent_clean_dir))
        self.latent_noisy_files = sorted(os.listdir(latent_noisy_dir))
    
        assert len(self.latent_clean_files) == len(self.latent_noisy_files), "Mismatch in number of files"
    
    def __len__(self):
        return len(self.latent_clean_files)
    
    def __getitem__(self, idx):
        clean_file = os.path.join(self.latent_clean_dir, self.latent_clean_files[idx])
        noisy_file = os.path.join(self.latent_noisy_dir, self.latent_noisy_files[idx])
        
        z_clean = np.load(clean_file)
        z_noisy = np.load(noisy_file)

        z_clean = torch.tensor(z_clean, dtype=torch.float32)
        z_noisy = torch.tensor(z_noisy, dtype=torch.float32)

        if z_clean.shape[-1] % 16 != 0:
            new_size = math.ceil(z_clean.shape[-1] / 16) * 16
            pad_amount = new_size - z_clean.shape[-1]
            z_clean = F.pad(z_clean, (0, pad_amount), mode='constant', value=0)
            z_noisy = F.pad(z_noisy, (0, pad_amount), mode='constant', value=0)
        
        return {
            "z_clean": z_clean,
            "z_noisy": z_noisy,
            "pad_amount": pad_amount if pad_amount else 0
        }
