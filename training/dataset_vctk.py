import os
import librosa
from torch.utils.data import Dataset, DataLoader
from models.utils import LogMelSpectrogram, count_parameters, get_padding_sample
import torch
import random
import numpy as np

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
    num_cuts = random.randint(1, max_cuts)  # Pilih jumlah cut secara acak antara 1 hingga max_cuts
    for _ in range(num_cuts):
        start = random.randint(0, audio_length - cut_samples)
        audio[:, start:start + cut_samples] = 0
    return audio

class DenoiserDataset(Dataset):
    def __init__(
        self, 
        clean_dir: str, 
        noisy_dir: str,
        add_random_cutting: bool = False,
        stage: int = 1,
        max_cuts: int = 10,
        cut_duration: float = 0.35,
        fixed_length: int = None  # Baru: panjang sampel tetap (dalam jumlah sample)
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
    
    def __getitem__(self, idx):
        # Load audio clean
        clean_audio, clean_sr = librosa.load(os.path.join(self.clean_dir, self.clean_files[idx]), sr=None)
        # Pilih sample rate acak untuk audio noisy agar bervariasi, lalu resample ke clean_sr
        noisy_sr = int(clean_sr / (np.random.randint(1, 3) + 1))
        noisy_audio, noisy_sr = librosa.load(os.path.join(self.noisy_dir, self.noisy_files[idx]), sr=noisy_sr)
        noisy_audio = librosa.resample(noisy_audio, orig_sr=noisy_sr, target_sr=clean_sr)
        noisy_sr_new = clean_sr

        # Ubah ke tensor dan tambahkan dimensi channel
        clean_audio = torch.tensor(clean_audio).unsqueeze(0)
        noisy_audio = torch.tensor(noisy_audio).unsqueeze(0)

        # Tambah random cut jika diaktifkan
        if self.add_random_cutting:
            noisy_audio = random_cut(noisy_audio, max_cuts=self.max_cuts, cut_duration=self.cut_duration, sample_rate=clean_sr)

        # Jika fixed_length diset, crop atau pad audio supaya panjangnya seragam
        if self.fixed_length is not None:
            if clean_audio.size(-1) > self.fixed_length:
                # Crop secara acak jika audio lebih panjang dari fixed_length
                max_offset = clean_audio.size(-1) - self.fixed_length
                offset = random.randint(0, max_offset)
                clean_audio = clean_audio[..., offset:offset + self.fixed_length]
                noisy_audio = noisy_audio[..., offset:offset + self.fixed_length]
            else:
                # Pad dengan nol jika audio lebih pendek dari fixed_length
                pad_amount = self.fixed_length - clean_audio.size(-1)
                clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_amount))
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_amount))

        item = {
            "clean_audio_tensor": clean_audio,
            "noisy_audio_tensor": noisy_audio,
            "clean_sr": clean_sr,
            "noisy_sr_default": noisy_sr_new
        }
        
        return item       

    def __len__(self):
        return len(self.clean_files)
