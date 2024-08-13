import torch
import torch.nn as nn
import torch.nn.functional as F

from lldm.vae.modules import get_timestep_embedding, Downsample, ResNetBlock, LinearAttentionBlock

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, num_heads, num_downsamples, max_timestep, hidden_channels, condition_dim):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_heads = num_heads
        self.num_downsamples = num_downsamples
        self.max_timestep = max_timestep
        self.hidden_channels = hidden_channels
        self.condition_dim = condition_dim
        
        self.timestep_embedding = get_timestep_embedding(max_timestep, hidden_channels)
        self.condition_embedding = nn.Linear(condition_dim, hidden_channels)
        
        self.downsamples = nn.ModuleList([
            Downsample(hidden_channels, hidden_channels) for _ in range(num_downsamples)
        ])
        
        self.res_blocks = nn.ModuleList([
            ResNetBlock(hidden_channels, num_heads) for _ in range(num_res_blocks)
        ])
        
        self.attn_block = LinearAttentionBlock(hidden_channels, num_heads)
        
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)