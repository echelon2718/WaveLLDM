import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.models.layers import DropPath
from models.unet import RMSNorm
from typing import Tuple

# Fungsi-fungsi rotary embedding seperti yang diberikan
def precompute_freqs_cis(dim: int, seq_len: int, device: str = "cuda", theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
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
    # Mengubah ke bentuk kompleks
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)
    # Terapkan rotary embedding
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

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

class LayerNorm2D(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GlobalResponseNorm(nn.Module):
    """ Global Response Normalization (GRN) layer.
    This layer normalizes the input tensor using the global response normalization technique.
    Expects input tensor of shape (batch_size, height, width, dim).
    
    Args:
        dim (int): The number of channels in the input tensor.
    
    Returns:
        Normalized tensor of the same shape as input.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x, eps=1e-6):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + eps)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., time_embedding_dim=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm2D(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.time_embedding_dim = time_embedding_dim
        
        if time_embedding_dim is not None:
            self.proj_network = nn.Sequential(
                nn.Linear(time_embedding_dim, 512),
                nn.SiLU(),
                nn.Linear(512, dim),
                nn.SiLU(),
            )

    def forward(self, x, temb=None):
        if temb is not None and self.time_embedding_dim is not None:
            assert len(temb.shape) == 2, f"Ukuran time embedding haruslah dua, batch_size x channels yaa, punya anda: {temb.shape}"
            x = x + self.proj_network(temb)[:, :, None, None] # Expected temb dimension is (bsz, ch), it will be projected to (bsz, ch, 1)
        else:
            print("Warning: Either time embedding or time embedding dim is None, so no time embedding will be added to the input.")

        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# Versi modifikasi Rotary Attention dengan Linear Attention untuk data gambar
class RotaryLinearAttention(nn.Module):
    def __init__(
        self, 
        dim: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        use_rmsnorm: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Parameter:
          - dim         : Dimensi fitur (misal: dim embedding)
          - n_heads     : Jumlah head untuk query
          - n_kv_heads  : Jumlah head untuk key & value (setelah nanti direplikasi)
          - device      : Device yang digunakan
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.norm = RMSNorm(dim) if use_rmsnorm else None
        self.device = device

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=device)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, device=device)

    def forward(self, x):
        """
        x bisa berupa:
        - tensor gambar dengan shape (B, C, H, W) → kita flatten H & W
        - atau tensor sekuens (B, N, dim)
        """
        is_image = False
        if x.ndim == 4:  # Asumsi input gambar: (B, C, H, W)
            is_image = True
            B, C, H, W = x.shape
            # Asumsikan x sudah berupa embedding dengan C == self.dim
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        B, seq_len, _ = x.shape

        if self.norm is not None:
            x = self.norm(x)

        # Hitung query, key, value
        xq = self.wq(x)  # (B, seq_len, n_heads * head_dim)
        xk = self.wk(x)  # (B, seq_len, n_kv_heads * head_dim)
        xv = self.wv(x)  # (B, seq_len, n_kv_heads * head_dim)

        # Reshape ke multi-head
        xq = xq.view(B, seq_len, self.n_heads, self.head_dim)         # (B, seq_len, n_heads, head_dim)
        xk = xk.view(B, seq_len, self.n_kv_heads, self.head_dim)         # (B, seq_len, n_kv_heads, head_dim)
        xv = xv.view(B, seq_len, self.n_kv_heads, self.head_dim)         # (B, seq_len, n_kv_heads, head_dim)

        # Precompute rotary frequencies untuk sequence yang sekarang
        freqs_cis = precompute_freqs_cis(self.head_dim, seq_len, device=self.device)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Replikasi keys dan values agar jumlah head sesuai dengan query
        keys = repeat_kv(xk, self.n_rep)    # (B, seq_len, n_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)    # (B, seq_len, n_heads, head_dim)

        # Pindahkan dim head ke posisi kedua
        xq = xq.transpose(1, 2)      # (B, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (B, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (B, n_heads, seq_len, head_dim)

        # Gunakan kernel φ(x) = elu(x) + 1 untuk linear attention
        phi = lambda x: F.elu(x) + 1

        Q_lin = phi(xq)      # (B, n_heads, seq_len, head_dim)
        K_lin = phi(keys)    # (B, n_heads, seq_len, head_dim)

        # Hitung KV = sum_{s} ( φ(K(s)) ⊗ V(s) ) → (B, n_heads, head_dim)
        KV = torch.einsum("bn s d, bn s d -> bn d", K_lin, values)
        # Hitung K_sum = sum_{s} φ(K(s)) → (B, n_heads, head_dim)
        K_sum = K_lin.sum(dim=2)
        # Hitung normalisasi per posisi: Z = ⟨φ(Q(s)), sum_s φ(K(s))⟩ → (B, n_heads, seq_len, 1)
        Z = torch.einsum("bn s d, bn d -> bn s", Q_lin, K_sum).unsqueeze(-1)

        # Hitung output: Y = ⟨φ(Q(s)), KV⟩ / (Z + eps)
        Y = torch.einsum("bn s d, bn d -> bn s d", Q_lin, KV)
        eps = 1e-6
        Y = Y / (Z + eps)

        # Kembalikan shape ke (B, seq_len, n_heads * head_dim)
        Y = Y.transpose(1, 2).contiguous().view(B, seq_len, -1)
        output = self.wo(Y)  # (B, seq_len, self.dim)

        if is_image:
            # Kembalikan ke bentuk (B, C, H, W) (dengan C == self.dim)
            output = output.transpose(1, 2).view(B, self.dim, H, W)
        
        return output

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
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

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_attn=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.use_attn = use_attn

        self.block1 = ConvNeXtV2Block(in_channels, time_embedding_dim=time_dim)
        self.block2 = ConvNeXtV2Block(in_channels, time_embedding_dim=time_dim)
        self.downsample = Downsample(in_channels, out_channels, with_conv=True)

        if use_attn:
            self.attn = RotaryLinearAttention(in_channels, n_heads=32, n_kv_heads=32)
        else:
            self.attn = None
    
    def forward(self, x, temb=None):
        x = self.block1(x, temb)
        x = self.block2(x, temb)
        
        if self.attn is not None:
            x = self.attn(x)

        x = self.downsample(x)
        
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_attn=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.use_attn = use_attn

        self.head = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.block1 = ConvNeXtV2Block(in_channels, time_embedding_dim=time_dim)
        self.block2 = ConvNeXtV2Block(in_channels, time_embedding_dim=time_dim)
        
        self.upsample = Upsample(in_channels, out_channels, with_conv=True)
        
        if use_attn:
            self.attn = RotaryLinearAttention(in_channels, n_heads=32, n_kv_heads=32)
        else:
            self.attn = None

    def forward(self, x, skip_x=None, temb=None):
        if skip_x is not None:
            x = torch.cat([x, skip_x], dim=1)
            x = self.head(x)
        
        x = self.block1(x, temb)
        x = self.block2(x, temb)
            
        if self.attn is not None:
            x = self.attn(x)

        x = self.upsample(x)
        
        return x

class RotaryUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        model_channels=64,
        out_channels=4,
        time_dim=None,
        num_levels=4    
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.num_levels = num_levels

        # Time embedding
        self.time_mlp = SinusoidalTimeEmbedding(model_channels)

        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=1, stride=1, padding=0)

        # Down blocks - increasing channel dimensions
        ch = model_channels
        self.down_blocks = nn.ModuleList()

        self.down_blocks.append(DownBlock(ch, ch*2, time_dim, use_attn=False))
        self.down_blocks.append(DownBlock(ch*2, ch*4, time_dim, use_attn=True))
        self.down_blocks.append(DownBlock(ch*4, ch*6, time_dim, use_attn=True))
        self.down_blocks.append(DownBlock(ch*6, ch*8, time_dim, use_attn=True))

        # Middle block with Residual-Attention-Residual structure
        # self.mid_block1 = ResBlock1(ch*8, dilation=(1,))
        self.mid_block1 = ConvNeXtV2Block(ch*8, time_embedding_dim=time_dim)
        self.mid_block2 = ConvNeXtV2Block(ch*8, time_embedding_dim=time_dim)
        self.mid_attn = RotaryLinearAttention(ch*8, n_heads=32, n_kv_heads=32)

        # Up blocks - decreasing channel dimensions
        self.up_blocks = nn.ModuleList()

        self.up_blocks.append(UpBlock(ch*16, ch*6, time_dim, use_attn=True))
        self.up_blocks.append(UpBlock(ch*12, ch*4, time_dim, use_attn=True))
        self.up_blocks.append(UpBlock(ch*8, ch*2, time_dim, use_attn=True))
        self.up_blocks.append(UpBlock(ch*4, ch, time_dim, use_attn=False))
        
            
        # Final block
        self.final_conv = nn.Conv2d(ch*2, out_channels, kernel_size=1, stride=1, padding=0)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(self, x, time, cond=None):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Time embedding
        t = self.time_mlp(time)
        # print(t.shape)
        
        # Initial projection
        if cond is not None:
            if len(cond.shape) == 3:
                cond = cond.unsqueeze(1)
                
            assert cond.shape[1] == x.shape[1], "Condition tensor must have the same number of channels (C) as input tensor."
            x = torch.cat([x, cond], dim=1)

        x = self.init_conv(x)
        # print(f"Initial Conv: {x.shape}")
        
        # Store skip connections
        skips = [x]
        
        # Down path
        # print(f"Down path:")
        for i, block in enumerate(self.down_blocks):
            x = block(x, t)
            # print(f"Layer {i+1}: {x.shape}")
            skips.append(x)
            
        
        # Middle block (R-A-R)
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        x = self.mid_attn(x)
        # print(f"Middle block: {x.shape}")

        # Up path with skip connections
        # print(f"Up path:")
        for i, block in enumerate(self.up_blocks):
            skip_x = skips.pop()
            x = block(x, skip_x, t)
            # print(f"Layer {i+1}: {x.shape}") 

        # Final processing
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.final_conv(x)
        # print(f"Final Conv: {x.shape}")
        
        return x.squeeze(1) if len(x.shape) == 4 else x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias)

'''
Prototype: To use diffusion model, you can run this function
'''

def create_diffusion_model(
    in_channels=1,
    base_channels=64,
    out_channels=1,
    time_dim=64
):
    """
    Creates a diffusion model with the specified configuration
    """
    return RotaryUNet(
        in_channels=in_channels,
        model_channels=base_channels,
        out_channels=out_channels,
        num_levels=4,
        time_dim=time_dim
    )
