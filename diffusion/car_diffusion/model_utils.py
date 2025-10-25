import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,)
        device = t.device
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: (b, c, h, w)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = x.view(b, c, h * w)  # (b, c, n)

        qkv = self.qkv(x)  # (b, 3c, n)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # split heads
        q = q.view(b, self.num_heads, c // self.num_heads, -1)
        k = k.view(b, self.num_heads, c // self.num_heads, -1)
        v = v.view(b, self.num_heads, c // self.num_heads, -1)

        scale = 1.0 / math.sqrt(c // self.num_heads)
        attn = torch.einsum('bhcn,bhcm->bhnm', q * scale, k * scale)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.contiguous().view(b, c, -1)
        out = self.proj(out)
        out = out.view(b, c, h, w)

        return out + x_in

