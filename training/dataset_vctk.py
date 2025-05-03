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
    # num_cuts = random.randint(1, max_cuts)  # Pilih jumlah cut secara acak antara 1 hingga max_cuts
    # for _ in range(num_cuts):
    #     start = random.randint(0, audio_length - cut_samples)
    #     audio[:, start:start + cut_samples] = 0

    num_cuts = torch.randint(1, max_cuts + 1, (1,)).item()
    starts = torch.randint(0, audio_length - cut_samples + 1, (num_cuts,))
    mask = torch.ones_like(audio)
    for start in starts:
        mask[..., start:start + cut_samples] = 0
    # return audio
    return audio * mask

def collate_fn_latents(batch):
    # Extract and preprocess clean and noisy tensors from the batch
    zq_down_clean = [item['zq_down_clean_audio'].squeeze(0).permute(1, 0) for item in batch]  # Shape: [seq_len, 512]
    zq_down_noisy = [item['zq_down_noisy_audio'].squeeze(0).permute(1, 0) for item in batch]  # Shape: [seq_len, 512]
    melspec_lengths = [item['clean_audio_tensor'].shape[-1] for item in batch]  # Shape: [batch_size, 512, seq_len]

    # Get the sequence lengths of all items in the batch
    lengths = torch.tensor([t.shape[0] for t in zq_down_clean])

    # Find the maximum sequence length in the batch
    max_length = max(lengths)

    # Round up to the nearest multiple of 16
    padded_length = math.ceil(max_length / 16) * 16

    # Manually pad each sequence to the padded_length
    clean_padded = torch.stack([
        torch.nn.functional.pad(t, (0, 0, 0, padded_length - t.shape[0]), "constant", 0)
        for t in zq_down_clean
    ])
    noisy_padded = torch.stack([
        torch.nn.functional.pad(t, (0, 0, 0, padded_length - t.shape[0]), "constant", 0)
        for t in zq_down_noisy
    ])

    # Adjust dimensions to [batch_size, 512, padded_length]
    clean_padded = clean_padded.permute(0, 2, 1)
    noisy_padded = noisy_padded.permute(0, 2, 1)

    # Return the padded tensors and additional metadata
    return {
        "clean_audio_downsampled_latents": clean_padded,
        "noisy_audio_downsampled_latents": noisy_padded,
        "lengths": lengths,
        "clean_sr": torch.tensor([item["clean_sr"] for item in batch]),
        "noisy_sr_default": torch.tensor([item["noisy_sr_default"] for item in batch]),
        "melspec_lengths": melspec_lengths,
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
        spec_trans = None,
        encoder: nn.Module = None,
        quantizer = None,
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
        self.spec_trans = spec_trans.to("cpu") if spec_trans is not None else None
        self.encoder = encoder.to("cpu") if encoder is not None else None
        self.quantizer = quantizer.to("cpu") if quantizer is not None else None
        self.device = device
    
    def __getitem__(self, idx):
        # Load audio clean
        # clean_audio, clean_sr = librosa.load(os.path.join(self.clean_dir, self.clean_files[idx]), sr=44100)
        clean_audio, clean_sr = torchaudio.load(os.path.join(self.clean_dir, self.clean_files[idx]))
        clean_audio = clean_audio.unsqueeze(0)

        # Pilih sample rate acak untuk audio noisy agar bervariasi, lalu resample ke clean_sr
        # noisy_sr = int(clean_sr / (np.random.randint(1, 3) + 1))
        # noisy_audio, noisy_sr = librosa.load(os.path.join(self.noisy_dir, self.noisy_files[idx]), sr=noisy_sr)
        # noisy_audio = librosa.resample(noisy_audio, orig_sr=noisy_sr, target_sr=clean_sr)
        # noisy_sr_new = clean_sr
        noisy_sr = int(clean_sr / (torch.randint(1, 3, (1,)).item() + 1))
        noisy_audio, orig_noisy_sr = torchaudio.load(os.path.join(self.noisy_dir, self.noisy_files[idx]))
        noisy_audio = noisy_audio.unsqueeze(0)

        # Ubah ke tensor dan tambahkan dimensi channel
        # clean_audio = torch.tensor(clean_audio).unsqueeze(0)
        # noisy_audio = torch.tensor(noisy_audio).unsqueeze(0)

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
        # if self.fixed_length is not None:
        #     if clean_audio.size(-1) > self.fixed_length:
        #         # Crop secara acak jika audio lebih panjang dari fixed_length
        #         max_offset = clean_audio.size(-1) - self.fixed_length
        #         offset = random.randint(0, max_offset)
        #         clean_audio = clean_audio[..., offset:offset + self.fixed_length]
        #         noisy_audio = noisy_audio[..., offset:offset + self.fixed_length]
        #     else:
        #         # Pad dengan nol jika audio lebih pendek dari fixed_length
        #         pad_amount = self.fixed_length - clean_audio.size(-1)
        #         clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_amount))
        #         noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_amount))

        ################### MODIFIKASI #######################
        if self.fixed_length is not None:
            audio_length = clean_audio.size(-1)
            if audio_length > self.fixed_length:
                # Crop secara acak
                max_offset = audio_length - self.fixed_length
                offset = torch.randint(0, max_offset + 1, (1,)).item()
                clean_audio = clean_audio[..., offset:offset + self.fixed_length]
                noisy_audio = noisy_audio[..., offset:offset + self.fixed_length]
            else:
                # Pad dengan nol
                pad_amount = self.fixed_length - audio_length
                clean_audio = F.pad(clean_audio, (0, pad_amount))
                noisy_audio = F.pad(noisy_audio, (0, pad_amount))

        #######################################################

        if self.encoder is not None and self.quantizer is not None:
            clean_audio_spec = self.spec_trans(clean_audio.to("cpu"))
            noisy_audio_spec = self.spec_trans(noisy_audio.to("cpu"))

            with torch.no_grad():
                self.encoder.eval()
                self.quantizer.eval()

                self.encoder = self.encoder.to("cpu")
                self.quantizer = self.quantizer.to("cpu")

                # Encode audio ke latent space
                clean_audio_latent = self.encoder(clean_audio_spec.to("cpu"))          
                noisy_audio_latent = self.encoder(noisy_audio_spec.to("cpu"))

                # Quantize latent audio
                zq_down_clean_audio = self.quantizer(clean_audio_latent).latents
                zq_down_noisy_audio = self.quantizer(noisy_audio_latent).latents

            item = {
                "zq_down_clean_audio": zq_down_clean_audio,
                "zq_down_noisy_audio": zq_down_noisy_audio,
                "clean_audio_tensor": clean_audio_spec,
                "noisy_audio_tensor": noisy_audio_spec,
                "clean_sr": clean_sr,
                "noisy_sr_default": noisy_sr_new
            }

            return item

        item = {
            "clean_audio_tensor": clean_audio.to(self.device) if self.spec_trans is None else self.spec_trans(clean_audio.to(self.device)),
            "noisy_audio_tensor": noisy_audio.to(self.device) if self.spec_trans is None else self.spec_trans(noisy_audio.to(self.device)),
            "clean_sr": clean_sr,
            "noisy_sr_default": noisy_sr_new
        }
        
        return item       

    def __len__(self):
        return len(self.clean_files)

class LatentDataset(Dataset):
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
