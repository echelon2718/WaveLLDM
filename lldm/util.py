import numpy as np
import math
import torch

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