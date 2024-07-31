import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.elu = nn.ELU()

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        
        # Apply ELU activation to q and k
        q = self.elu(q) + 1  # Ensure positivity
        k = self.elu(k) + 1  # Ensure positivity

        # Attention mechanism
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        
        # Reshape and project out
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)
    
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.Q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        Q = self.Q(h_)
        K = self.K(h_)
        V = self.V(h_)

        B, C, H, W = Q.shape
        Q = rearrange(Q, 'b c h w -> b (h w) c')
        K = rearrange(K, 'b c h w -> b (h w) c')
        context_ = torch.einsum('bij,bjk->bik', Q, K)

        context_ = context_ * (int(C) ** -0.5)
        context_ = F.softmax(context_, dim=2)

        V = rearrange(V, 'b c h w -> b c (h w)')
        context_ = rearrange(context_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', V, context_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=H, w=W)
        h_ = self.proj_out(h_)

        return x + h_
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, is_inplace=True):
        use_flash_attention = False
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.heads = heads
        self.dim_head = dim_head

        self.scale = dim_head ** -0.5

        if context_dim is None:
            context_dim = query_dim
        
        # QKV Mappings
        d_attn = dim_head * heads
        self.Q = nn.Linear(query_dim, d_attn, bias=False)
        self.K = nn.Linear(context_dim, d_attn, bias=False)
        self.V = nn.Linear(context_dim, d_attn, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(d_attn, query_dim),
            nn.Dropout(dropout)
        )

        # Trying to use flash attention if exist:
        try:
            from flash_attn.flash_attention import FlashAttention

            self.flash = FlashAttention()
            self.flash.softmax_scale = self.scale
        except ImportError:
            self.flash = None
    
    def forward(self, x, context=None, mask=None):
        has_cond = context is not None
        if not has_cond:
            context = x
        
        Q = self.Q(x)
        K = self.K(context)
        V = self.V(context)

        if (
            CrossAttention.use_flash_attention
            and self.flash is not None
            and not has_cond
            and self.dim_head <= 128
        ):
            return self.flash_attention(Q, K, V)
        else:
            return self.standard_attention(Q, K, V)
    
    def flash_attention(self, Q, K, V):
        B, N, C  = Q.shape
        QKV = torch.stack((Q, K, V), dim=2)
        QKV = QKV.view(B, N, 3, self.heads, self.dim_head)

        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f"dim_head {self.d_head} not supported: Too large for Flash Attention")
        
        if pad:
            QKV = torch.cat(
                (QKV, QKV.new_zeros(B, N, 3, self.heads, pad)), dim=-1
            )
        
        out, _ = self.flash(QKV.type(torch.float16))
        out = out[:, :, :, :self.dim_head].float()
        out = out.reshape(B, N, self.heads * self.dim_head)

        return self.to_out(out)

    def standard_attention(self, Q, K, V):
        B, N, C = Q.shape
        Q = rearrange(Q, 'b n (h d) -> b n h d', h=self.heads)
        K = rearrange(K, 'b m (h d) -> b h m d', h=self.heads)
        V = rearrange(V, 'b m (h d) -> b h m d', h=self.heads)

        context = torch.einsum('bqhd,bhmd->bqhm', Q, K) * self.scale

        # if self.is_inplace:
        #     half = context.shape[0]//2
        #     context[half:] = context[half:].softmax(dim=-1)
        #     context[:half] = context[:half].softmax(dim=-1)

        context = F.softmax(context, dim=-1)
        out = torch.einsum('bqhm,bhmd->bqhd', context, V)
        out = rearrange(out, 'b n h d -> b n (h d)')
        out = self.to_out(out)
        return out

