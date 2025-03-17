import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from models.utils import LogMelSpectrogram

device = "cuda" if torch.cuda.is_available() else "cpu"

def feature_loss(fmap_r: List[torch.Tensor], fmap_g: List[torch.Tensor]):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss = loss + r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def melspec_loss(aud_x, aud_y, spec_transform):
    # Ensure same length in temporal dimension
    aud_x, aud_y = aud_x.to(device), aud_y.to(device)
    if aud_x.shape[-1] > aud_y.shape[-1]:
        aud_y = F.pad(aud_y, (0, aud_x.shape[-1] - aud_y.shape[-1]))
    elif aud_x.shape[-1] < aud_y.shape[-1]:
        aud_x = F.pad(aud_x, (0, aud_y.shape[-1] - aud_x.shape[-1]))
    
    # Compute mel-spectrograms
    spec_x = spec_transform(aud_x)
    spec_y = spec_transform(aud_y)
    
    # Calculate L1 loss
    loss = nn.L1Loss()(spec_x, spec_y)
    return loss

class MultiScaleSpecLoss(nn.Module):
    def __init__(
        self,
        n_mels_list : list = [5, 10, 20, 40, 80, 160, 320],
        n_ffts : list = [32, 64, 128, 256, 512, 1024, 2048],
        win_lengths : list = [32, 64, 128, 256, 512, 1024, 2048]
    ):
        super(MultiScaleSpecLoss, self).__init__()
        self.n_mels_list = n_mels_list
        self.n_ffts = n_ffts
        self.win_lengths = win_lengths
        self.hop_lengths = [w // 4 for w in win_lengths]

        self.spec_transforms = []

        for n_mels, n_fft, win_length, hop_length in zip(self.n_mels_list, self.n_ffts, self.win_lengths, self.hop_lengths):
            spec_transform = LogMelSpectrogram(
                sample_rate=44100,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                convert_to_fp_16=False
            )
            self.spec_transforms.append(spec_transform.to(device))      

        self.criterion = nn.L1Loss()

    def forward(self, aud_x, aud_y, device):
        aud_x, aud_y = aud_x.to(device), aud_y.to(device)
        
        # Sesuaikan panjang sinyal secara temporal
        if aud_x.shape[-1] > aud_y.shape[-1]:
            aud_y = F.pad(aud_y, (0, aud_x.shape[-1] - aud_y.shape[-1]))
        elif aud_x.shape[-1] < aud_y.shape[-1]:
            aud_x = F.pad(aud_x, (0, aud_y.shape[-1] - aud_x.shape[-1]))
        
        total_loss = 0.0
    
        # Iterasi untuk tiap transformasi (skala)
        for spec_transform in self.spec_transforms:
            # Hitung mel spectrogram untuk kedua sinyal
            spec_x = spec_transform(aud_x)
            spec_y = spec_transform(aud_y)
            
            # Tambahkan loss L1 dari masing-masing skala
            total_loss += self.criterion(spec_x, spec_y)
        
        return total_loss

class MultiScaleSTFTLoss(nn.Module):
    def __init__(
        self,
        n_ffts : list = [32, 64, 128, 256, 512, 1024, 2048],
        win_lengths : list = [32, 64, 128, 256, 512, 1024, 2048]
    ):
        super(MultiScaleSTFTLoss, self).__init__()

        self.n_ffts = n_ffts
        self.win_lengths = win_lengths
        self.hop_lengths = [w // 4 for w in win_lengths]

        self.criterion = nn.L1Loss()

    def stft(self, audio, n_fft, win_length, hop_length, device):
        """Menghitung STFT dari sinyal audio"""
        window = torch.hann_window(win_length).to(device)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return spec

    def forward(self, aud_x, aud_y, device):
        """
        Menghitung Multi-Scale STFT Loss antara sinyal target (aud_y) dan sinyal hasil rekonstruksi (aud_x).
        """

        # Pastikan kedua sinyal memiliki panjang yang sama
        if aud_x.shape[-1] > aud_y.shape[-1]:
            aud_y = F.pad(aud_y, (0, aud_x.shape[-1] - aud_y.shape[-1]))
        elif aud_x.shape[-1] < aud_y.shape[-1]:
            aud_x = F.pad(aud_x, (0, aud_y.shape[-1] - aud_x.shape[-1]))

        aud_x = aud_x.squeeze(1)  # [batch, T]
        aud_y = aud_y.squeeze(1)  # [batch, T]
        
        loss = 0.0

        # Hitung STFT loss untuk berbagai skala
        for n_fft, win_length, hop_length in zip(self.n_ffts, self.win_lengths, self.hop_lengths):
            spec_x = self.stft(aud_x, n_fft, win_length, hop_length, device)
            spec_y = self.stft(aud_y, n_fft, win_length, hop_length, device)

            # Pisahkan magnitudo dan fase
            mag_x = torch.abs(spec_x)
            mag_y = torch.abs(spec_y)

            # L1 Loss pada spektrum magnitude
            loss += self.criterion(mag_x, mag_y)

        return loss
