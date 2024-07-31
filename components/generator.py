# This file contains the generator model for the HifiGAN model.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.nn.utils import remove_weight_norm
import math

from model.components.blocks import LRELU_SLOPE, get_padding, init_weights, Conv1D, ConvTranspose1D, FusedLeakyReLU, ResidualBlock1, ResidualBlock2

class Generator(nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__():
        self.h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = Conv1D(in_channels=80, out_channels=h.upsample_initial_channel, kernel_size=7, stride=1, padding=3, dilation=1, activation=None)
        resblock = ResidualBlock1 if h.resblock_type == '1' else ResidualBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1D(in_channels=h.upsample_initial_channel // (2**i), out_channels=h.upsample_initial_channel // (2**(i+1)), kernel_size=k, stride=u, padding=(k - u) // 2, dilation=1, activation=None)
            )
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, kernel_size=k, dilation=d))
        
        self.conv_post = Conv1D(in_channels=ch, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, activation=None)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = FusedLeakyReLU(LRELU_SLOPE)(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels)
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        
        x = FusedLeakyReLU(LRELU_SLOPE)(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
    
    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre.conv)
        for up in self.ups:
            remove_weight_norm(up.conv)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        remove_weight_norm(self.conv_post.conv)

class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    
    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
