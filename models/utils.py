import torch
from typing import Tuple
import torchaudio.functional as F
from torch import Tensor, nn
from torchaudio.transforms import MelScale
import numpy as np
import math


################# RoFormer #######################
def precompute_freqs_cis(dim: int, seq_len: int, device: str = "cuda", theta: float = 10000.0):
    """Precomputes the rotary embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequencies for broadcasting to match input tensor."""
    ndim = x.ndim
    shape = [1] * ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary positional embedding to the input tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value tensors for multiple attention heads."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

############### Spectograms and audio processing utils ###########################
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def floor_mult8(x):
    return np.floor(x/8)*8

def ceil_mult8(x):
    return np.ceil(x/8)*8

def f(s):
    return np.floor((s - 2048)/512) + 5

def g(T):
    return (T - 1)*512 + 2048

def solve_n0(s_in, T_out, epsilon=0.000001):
    rhs = math.ceil((T_out + epsilon) / 8) * 8
    rhs_adjusted = rhs - 5
    lower_bound = rhs_adjusted * 512
    upper_bound = (rhs_adjusted + 1) * 512
    n0_lower = lower_bound + 2048 - s_in
    n0_upper = upper_bound + 2048 - s_in
    
    return n0_lower, n0_upper

def inverse_f(f_s, n_fft=2048, l_hop=512):
    s_lower = (f_s - 5) * l_hop + n_fft
    s_upper = (f_s - 4) * l_hop + n_fft
    return s_lower, s_upper

def get_padding_sample(sample, from_torch=False):
    if from_torch:
        sample = sample.numpy()
    sample_size = sample.shape[0]
    spec_length = f(sample_size)
    autoencoder_spec_length = floor_mult8(spec_length)
    infimum_d = solve_n0(sample_size, autoencoder_spec_length)[0]
    zeros_padding = np.zeros(infimum_d)
    if not from_torch:
        return np.concatenate((zeros_padding, sample))
    else:
        return torch.tensor(np.concatenate((zeros_padding, sample)))

class LinearSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, y: Tensor) -> Tensor:
        if y.ndim == 3:
            y = y.squeeze(1)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)

        fb = F.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer(
            "fb",
            fb,
            persistent=False,
        )

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def apply_mel_scale(self, x: Tensor) -> Tensor:
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)

    def forward(
        self, x: Tensor, return_linear: bool = False, sample_rate: int = None
    ) -> Tensor:
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = F.resample(x, orig_freq=sample_rate, new_freq=self.sample_rate)

        linear = self.spectrogram(x)
        x = self.apply_mel_scale(linear)
        x = self.compress(x)

        if return_linear:
            return x, self.compress(linear)

        return x
