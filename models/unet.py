import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from .modules import ModelArgs, ConvolutionalNet, TransConvNet, RotaryAttention, RMSNorm, ResBlock1

class SinusoidalTimeEmbedding(nn.Module):
    """
    Time embedding using sinusoidal positions, similar to transformer positional encoding.
    This helps the model understand the diffusion timestep.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=x.device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DownBlock(nn.Module):
    """
    Downsampling block with dual residual blocks and optional attention
    """
    def __init__(self, in_channels, out_channels, time_dim, use_attention=False):
        super().__init__()
        self.res = ResBlock1(in_channels, dilation=(1,1), time_embedding_dim=time_dim)
        self.downsample = nn.Conv1d(in_channels, out_channels, 4, stride=2, padding=1)
        
        if use_attention:
            # Convert channels to sequence for attention
            self.norm = RMSNorm(in_channels)
            self.attention = RotaryAttention(
                ModelArgs(dim=out_channels, n_heads=8, n_kv_heads=4),
                dim=in_channels,
                n_heads=8,
                n_kv_heads=4,
                max_seq_len=2048
            )
        else:
            self.attention = None

    def forward(self, x, temb=None):
        x = self.res(x, temb)
        
        if self.attention is not None:
            # Reshape for attention: [B, C, L] -> [B, L, C]
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = self.attention(x)
            x = x.transpose(1, 2)

        x = self.downsample(x)
        
        return x

class UpBlock(nn.Module):
    """
    Upsampling block with dual residual blocks and optional attention
    """
    def __init__(self, in_channels, out_channels, time_dim, use_attention=False):
        super().__init__()
        self.head = ConvolutionalNet(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)
        self.res = ResBlock1(in_channels, dilation=(1,1), time_embedding_dim=time_dim)
        self.upsample = TransConvNet(in_channels, out_channels, 4, stride=2)
        
        if use_attention:
            self.norm = RMSNorm(in_channels)
            self.attention = RotaryAttention(
                ModelArgs(dim=out_channels, n_heads=8, n_kv_heads=4),
                dim=in_channels,
                n_heads=8,
                n_kv_heads=4,
                max_seq_len=2048
            )
        else:
            self.attention = None

    def forward(self, x, skip_x=None, temb=None):
        if skip_x is not None:
            x = torch.cat([x, skip_x], dim=1)
            x = self.head(x)
        
        x = self.res(x, temb)
            
        if self.attention is not None:
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = self.attention(x)
            x = x.transpose(1, 2)

        x = self.upsample(x)
        
        return x

class DiffusionUNet(nn.Module):
    """
    Advanced U-Net architecture with rotary attention and residual blocks.
    Features 4 levels with dual residual blocks and attention integration.
    """
    def __init__(
        self,
        in_channels=4,
        model_channels=64,
        out_channels=4,
        time_dim=None,
        num_levels=4
    ):
        super().__init__()
        
        self.time_dim = time_dim if time_dim is not None else model_channels
        time_dim = self.time_dim
        
        # Time embedding
        self.time_mlp = SinusoidalTimeEmbedding(model_channels)
        
        # Initial projection
        self.init_conv = ConvolutionalNet(in_channels, model_channels, kernel_size=3)
        
        # Down blocks - increasing channel dimensions
        ch = model_channels
        self.down_blocks = nn.ModuleList()
        channels = []
        
        self.down_blocks.append(DownBlock(ch, ch*2, time_dim, use_attention=False))
        self.down_blocks.append(DownBlock(ch*2, ch*4, time_dim, use_attention=True))
        self.down_blocks.append(DownBlock(ch*4, ch*6, time_dim, use_attention=True))
        self.down_blocks.append(DownBlock(ch*6, ch*8, time_dim, use_attention=True))
        
        # Middle block with Residual-Attention-Residual structure
        self.mid_block1 = ResBlock1(ch*8, dilation=(1,))
        self.mid_attn = RotaryAttention(
            ModelArgs(dim=ch*8, n_heads=8, n_kv_heads=4),
            dim=ch*8,
            n_heads=8,
            n_kv_heads=4,
            max_seq_len=1024
        )
        self.mid_block2 = ResBlock1(ch*8, dilation=(1,))
        self.mid_upsample = TransConvNet(ch*8, ch*8, 4, stride=2)
        
        # Up blocks - decreasing channel dimensions
        self.up_blocks = nn.ModuleList()

        self.up_blocks.append(UpBlock(ch*16, ch*6, time_dim, use_attention=True))
        self.up_blocks.append(UpBlock(ch*12, ch*4, time_dim, use_attention=True))
        self.up_blocks.append(UpBlock(ch*8, ch*2, time_dim, use_attention=True))
        self.up_blocks.append(UpBlock(ch*4, ch, time_dim, use_attention=False))
        
            
        # Final blocks
        self.final_res = ResBlock1(ch*2, dilation=(1,1))
        self.final_conv = ConvolutionalNet(ch*2, out_channels, kernel_size=3)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias)
                
    def forward(self, x, time):
        # Time embedding
        t = self.time_mlp(time)
        
        # Initial projection
        x = self.init_conv(x)
        
        # Store skip connections
        skips = [x]
        
        # Down path
        for block in self.down_blocks:
            x = block(x, t)
            skips.append(x)
            
        
        # Middle block (R-A-R)
        x = self.mid_block1(x, t)
        # Prepare for attention
        x = x.transpose(1, 2)
        x = self.mid_attn(x)
        x = x.transpose(1, 2)
        x = self.mid_block2(x, t)

        # Up path with skip connections
        for block in self.up_blocks:
            skip_x = skips.pop()
            x = block(x, skip_x, t)

        # Final processing
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.final_conv(x)
        
        return x


'''
Prototype: To use diffusion model, you can run this function
'''

def create_diffusion_model(
    in_channels=384,
    base_channels=256,
    out_channels=384,
):
    """
    Creates a diffusion model with the specified configuration
    """
    return DiffusionUNet(
        in_channels=in_channels,
        model_channels=base_channels,
        out_channels=out_channels,
        num_levels=4
    )
