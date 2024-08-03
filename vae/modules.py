import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(
        num_channels=in_channels, num_groups=num_groups, eps=1e-6, affine=True
    )

def nonlinearity(x):
    return x * torch.sigmoid(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, time_embedding_channel = 512, dropout = 0.1):
        super(ResNetBlock, self).__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.time_embedding_projection = nn.Linear(time_embedding_channel, out_channels) if time_embedding_channel > 0 else None
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else None
    
    def forward(self, x, time_embedding=None):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        if time_embedding is not None:
            h += self.time_embedding_projection(nonlinearity(time_embedding))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        
        return x + h

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(LinearAttentionBlock, self).__init__()
        self.norm = Normalize(in_channels)
        self.Q = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1)
        self.K = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1)
        self.V = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1)
    
    def forward(self, x):
        x = self.norm(x)
        
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        
        b, c, h, w = Q.shape
        
        Q = Q.view(b, c, -1)
        K = K.view(b, c, -1)
        V = V.view(b, c, -1)
        
        Q = F.softmax(Q, dim=-1)
        K = F.softmax(K, dim=-1)
        
        attention = torch.bmm(Q, K.transpose(1, 2))
        attention = attention / (c ** 0.5)
        
        out = torch.bmm(attention, V)
        out = out.view(b, c, h, w)
        out = self.out(out)
        
        return out

class MultiHeadLinearAttentionBlock(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(MultiHeadLinearAttentionBlock, self).__init__()
        self.heads = heads
        self.norm = Normalize(in_channels)
        self.Q = nn.Conv2d(in_channels, in_channels * heads, kernel_size=1, padding=0, stride=1)
        self.K = nn.Conv2d(in_channels, in_channels * heads, kernel_size=1, padding=0, stride=1)
        self.V = nn.Conv2d(in_channels, in_channels * heads, kernel_size=1, padding=0, stride=1)
        self.out = nn.Conv2d(in_channels * heads, in_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        
        Q = self.Q(x).view(b, self.heads, c, h * w)
        K = self.K(x).view(b, self.heads, c, h * w)
        V = self.V(x).view(b, self.heads, c, h * w)
        
        Q = F.softmax(Q, dim=-1)
        K = F.softmax(K, dim=-1)
        
        attention = torch.einsum('bhcq,bhck->bhqk', Q, K) / (c ** 0.5)
        out = torch.einsum('bhqk,bhck->bhcq', attention, V).contiguous()
        
        out = out.view(b, self.heads * c, h, w)
        out = self.out(out)
        
        return out

class MultiHeadCrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(MultiHeadCrossAttentionBlock, self).__init__()
        self.heads = heads
        self.norm_q = Normalize(in_channels)
        self.norm_kv = Normalize(in_channels)
        self.Q = nn.Conv2d(in_channels, in_channels * heads, kernel_size=1, padding=0, stride=1)
        self.K = nn.Conv2d(in_channels, in_channels * heads, kernel_size=1, padding=0, stride=1)
        self.V = nn.Conv2d(in_channels, in_channels * heads, kernel_size=1, padding=0, stride=1)
        self.out = nn.Conv2d(in_channels * heads, in_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x_q, x_kv):
        b, c, h, w = x_q.shape
        x_q = self.norm_q(x_q)
        x_kv = self.norm_kv(x_kv)
        
        Q = self.Q(x_q).view(b, self.heads, c, h * w)
        K = self.K(x_kv).view(b, self.heads, c, h * w)
        V = self.V(x_kv).view(b, self.heads, c, h * w)
        
        Q = F.softmax(Q, dim=-1)
        K = F.softmax(K, dim=-1)
        
        attention = torch.einsum('bhcq,bhck->bhqk', Q, K) / (c ** 0.5)
        out = torch.einsum('bhqk,bhck->bhcq', attention, V).contiguous()
        
        out = out.view(b, self.heads * c, h, w)
        out = self.out(out)
        
        return out

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels : int, 
                 intermediate_channels : int, 
                 channel_multipliers : list, 
                 resblock_counts : int, 
                 attn_resolutions : list, 
                 dropout : int = 0.0, 
                 resolution : int = 256, # Perlu disesuaikan dengan ukuran mel spektrogram
                 z_channels : int = 256, 
                 double_z : bool =True):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.time_embedding_channels = 0
        self.num_resolutions = len(channel_multipliers)
        self.resblock_counts = resblock_counts
        self.resolution = resolution

        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=self.intermediate_channels, kernel_size=3, padding=1, stride=1)

        current_resolution = resolution
        in_channels_multiplier = (1, ) + tuple(channel_multipliers)
        self.in_channels_multipliers = in_channels_multiplier
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = intermediate_channels * in_channels_multiplier[i_level]
            block_out = intermediate_channels * channel_multipliers[i_level]
            for i_block in range(self.resblock_counts):
                block.append(
                    ResNetBlock(in_channels=block_in,
                                out_channels=block_out,
                                time_embedding_channel=self.time_embedding_channels,
                                dropout=dropout)
                )
                block_in = block_out
                if current_resolution in attn_resolutions:
                    attn.append(
                        LinearAttentionBlock(in_channels=block_out)
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_out, with_conv=True)
                current_resolution //= 2
            
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResNetBlock(in_channels=block_in, out_channels=block_in, time_embedding_channel=self.time_embedding_channels, dropout=dropout)
        self.mid.attn_1 = LinearAttentionBlock(in_channels=block_in)
        self.mid.block_2 = ResNetBlock(in_channels=block_in, out_channels=block_in, time_embedding_channel=self.time_embedding_channels, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels * 2 if double_z else z_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x, time_embedding=None):
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.resblock_counts):
                h = self.down[i_level].block[i_block](hs[-1], time_embedding)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        h = hs[-1]
        h = self.mid.block_1(h, time_embedding)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, time_embedding)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h
            
class Decoder(nn.Module):
    def __init__(self, 
                 out_channels : int, 
                 intermediate_channels : int, 
                 channel_multipliers: list,
                 resblock_counts : int,
                 attn_resolutions : list,
                 dropout : int = 0.0,
                 resolution : int = 256,
                 z_channels : int = 256,
                 double_z : bool = True,
                 give_pre_end : bool = False,
                 tanh_out : bool = False):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.intermediate_channels = intermediate_channels
        self.time_embedding_channels = 0
        self.num_resolutions = len(channel_multipliers)
        self.resblock_counts = resblock_counts
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = intermediate_channels * channel_multipliers[self.num_resolutions - 1]
        current_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, z_channels, current_resolution, current_resolution*2) # Ini keknya perlu diperbaiki, ukurannya seharusnya (1, z_channels, current_resolution(dimensi mel), downsampled_timestep (gonna do something about this))
        print("Melakukan operasi pada z dengan dimensi {} = {} dimensi.".format(
              self.z_shape, np.prod(self.z_shape)))

        # z masuk ke dalam decoder, mulai dari conv_in
        self.conv_in = nn.Conv2d(in_channels=z_channels, out_channels=block_in, kernel_size=3, padding=1, stride=1)

        self.mid = nn.Module()
        self.mid_block_1 = ResNetBlock(in_channels=block_in, out_channels=block_in, time_embedding_channel=self.time_embedding_channels, dropout=dropout)
        self.mid_attn_1 = LinearAttentionBlock(in_channels=block_in)
        self.mid_block_2 = ResNetBlock(in_channels=block_in, out_channels=block_in, time_embedding_channel=self.time_embedding_channels, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = intermediate_channels * channel_multipliers[i_level]
            for i_block in range(self.resblock_counts + 1):
                block.append(
                    ResNetBlock(in_channels=block_in, out_channels=block_out, time_embedding_channel=self.time_embedding_channels, dropout=dropout)
                )
                block_in = block_out

                if current_resolution in attn_resolutions:
                    attn.append(
                        LinearAttentionBlock(in_channels=block_out)
                    )
                
            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                current_resolution = current_resolution * 2
            
            self.up.insert(0, up)
        
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, padding=1, stride=1)
    
    def forward(self, z, time_embedding=None):
        self.last_z_shape = z.shape

        h = self.conv_in(z)

        h = self.mid_block_1(h, time_embedding)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h, time_embedding)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.resblock_counts + 1):
                h = self.up[i_level].block[i_block](h, time_embedding)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        if self.give_pre_end:
            return h
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        
        return h
