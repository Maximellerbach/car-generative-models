import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import SinusoidalPosEmb, AttentionBlock


class FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # add time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    """A compact U-Net intended for diffusion models.

    - Input: (B, in_channels, H, W)
    - t: (B,) or (B,1) timestep tensor
    - Output: same shape as input: predicted noise or denoised image
    
    Args:
        image_height: Height of input images (not used directly but kept for compatibility)
        image_width: Width of input images (not used directly but kept for compatibility)
        channels: Number of input/output channels (e.g., 3 for RGB)
        model_channels: Base number of channels in the model
        channel_multipliers: Tuple of multipliers for each resolution level
        num_res_blocks: Number of residual blocks per resolution level
        time_emb_dim: Dimension of time embedding
        use_attention: Whether to use attention blocks
        attention_resolutions: Tuple indicating which resolution levels get attention
    """

    def __init__(
        self,
        image_height: int = 96,
        image_width: int = 144,
        channels: int = 3,
        model_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        use_attention_at: tuple = (False, False, True, True),
    ):
        super().__init__()
        
        # Store config for reference
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        
        use_attention_at = use_attention_at if use_attention_at else (False,) * len(channel_multipliers)
        
        # Ensure use_attention_at has the right length
        if len(use_attention_at) < len(channel_multipliers):
            use_attention_at = use_attention_at + (False,) * (len(channel_multipliers) - len(use_attention_at))

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # input conv
        self.init_conv = nn.Conv2d(channels, model_channels, kernel_size=3, padding=1)

        # encoder
        self.downs = nn.ModuleList()
        self.resblocks_down = nn.ModuleList()
        in_ch = model_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.resblocks_down.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
            attn = AttentionBlock(out_ch) if (use_attention_at[i]) else nn.Identity()
            self.downs.append(nn.ModuleDict({
                'attn': attn,
                'down': Downsample(out_ch) if i != len(channel_multipliers) - 1 else nn.Identity()
            }))

        # middle
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_emb_dim)

        # decoder
        self.ups = nn.ModuleList()
        self.resblocks_up = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = model_channels * mult
            for j in range(num_res_blocks + 1):  # +1 because of skip connection concat
                # First block in each stage concatenates with skip, others don't
                in_channels_for_block = in_ch + out_ch if j == 0 else in_ch
                self.resblocks_up.append(ResidualBlock(in_channels_for_block, out_ch, time_emb_dim))
                in_ch = out_ch

            attn = AttentionBlock(out_ch) if (use_attention_at[i]) else nn.Identity()
            self.ups.append(nn.ModuleDict({
                'attn': attn,
                'up': Upsample(out_ch) if i != 0 else nn.Identity()
            }))

        # final
        self.norm_out = nn.GroupNorm(8, in_ch)
        self.conv_out = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1)
        )
        
        # Store num_res_blocks for forward pass
        self.num_res_blocks = num_res_blocks

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (b, c, h, w), t: (b,) or (b,1)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.float()
        t_emb = self.time_embedding(t)

        h = self.init_conv(x)
        skips = []

        # encoder
        res_iter = iter(self.resblocks_down)
        for stage in self.downs:
            # apply residual blocks for this stage
            for _ in range(self.num_res_blocks):
                blk = next(res_iter)
                h = blk(h, t_emb)
            
            # Save skip connection before attention and downsampling
            skips.append(h)
            h = stage['attn'](h)
            h = stage['down'](h)

        # middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # decoder
        res_iter_up = iter(self.resblocks_up)
        for stage in self.ups:
            # run residual blocks for this stage (num_res_blocks + 1)
            for j in range(self.num_res_blocks + 1):
                # Concatenate skip connection before first block in each stage
                if j == 0 and skips:
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                
                blk = next(res_iter_up)
                h = blk(h, t_emb)

            h = stage['attn'](h)
            h = stage['up'](h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h
