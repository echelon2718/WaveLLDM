import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm
import math
from einops import rearrange

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class FusedLeakyReLU(nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=True)

class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, activation=None, flrelu_negative_slope=None):
        super(Conv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        if activation is not None:
            if activation == 'lrelu':
                self.activation = nn.LeakyReLU()
            elif activation == 'silu':
                self.activation = SiLU()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'flrelu':
                if flrelu_negative_slope is None:
                    print('Warning: Negative slope value not provided for Fused Leaky ReLU. Using default value of 0.2')
                    flrelu_negative_slope = 0.2
                self.activation = FusedLeakyReLU(flrelu_negative_slope)
            else:
                try:
                    activation(torch.tensor(0))                  
                except:
                    raise ValueError('Activation function not supported')

                self.activation = activation
        else:
            self.activation = None

        self.conv = weight_norm(nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                          kernel_size=self.kernel_size, stride=self.stride, 
                                          dilation=self.dilation, padding=self.padding))

    def forward(self, x, return_out_size=False):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        if return_out_size:
            output_size = self.findOutputSize(x)
            return out, output_size
        
        return out
    
    def findOutputSize(self, x):
        N, _, L_in = x.size()
        L_out = math.floor((L_in + 2*self.padding - self.dilation*(self.kernel_size-1) - 1)/self.stride + 1)
        return N, self.out_channels, L_out

class ConvTranspose1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, dilation=1, activation=None):
        super(ConvTranspose1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.output_padding = output_padding
        if activation is not None:
            if activation == 'lrelu':
                self.activation = nn.LeakyReLU()
            elif activation == 'silu':
                self.activation = SiLU()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            else:
                try:
                    activation(torch.tensor(0))                  
                except:
                    raise ValueError('Activation function not supported')

                self.activation = activation
        else:
            self.activation = None

        self.conv = weight_norm(nn.ConvTranspose1d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                                   kernel_size=self.kernel_size, stride=self.stride, 
                                                   dilation=self.dilation, padding=self.padding))

    def forward(self, x, return_out_size=False):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        if return_out_size:
            output_size = self.findOutputSize(x)
            return out, output_size
        
        return out
    
    def findOutputSize(self, x):
        N, _, L_in = x.size()
        L_out = (L_in - 1)*self.stride - 2*self.padding + self.dilation*(self.kernel_size-1) + self.output_padding + 1
        return N, self.out_channels, L_out

class ResidualBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResidualBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            Conv1D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]), activation=None),
            Conv1D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]), activation=None),
            Conv1D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]), activation=None)
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            Conv1D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1), activation=None),
            Conv1D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1), activation=None),
            Conv1D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1), activation=None)
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = FusedLeakyReLU(LRELU_SLOPE)(x)
            xt = c1(xt)
            xt = FusedLeakyReLU(LRELU_SLOPE)(xt)
            xt = c2(xt)
            x = x + xt
        return x
    
    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_weight_norm(c1.conv)
            remove_weight_norm(c2.conv)

class ResidualBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResidualBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            Conv1D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]), activation=None),
            Conv1D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]), activation=None)
        ])
        self.convs.apply(init_weights)
    
    def forward(self, x):
        for c in self.convs:
            xt = FusedLeakyReLU(LRELU_SLOPE)(x)
            xt = c(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c in self.convs:
            remove_weight_norm(c.conv)

class DownBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class UpBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class MiddleBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

