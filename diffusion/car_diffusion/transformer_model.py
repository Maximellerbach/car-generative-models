"""
Implementation of the U-ViT, from https://arxiv.org/abs/2209.12152.
https://github.com/baofff/U-ViT/blob/main/libs/uvit.py
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import SinusoidalPosEmb


class ViTBlock(nn.Module):
    """Vision Transformer Block with Multi-Head Attention and Feed-Forward Network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        # Use SDPA-compatible attention for automatic flash attention / memory efficient attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # Multi-head attention with residual
        x_norm = self.norm_1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        # Feed-forward with residual
        x_norm = self.norm_2(x)
        x_ffn = self.fc1(x_norm)
        x_ffn = self.act(x_ffn)
        x_ffn = self.dropout1(x_ffn)
        x_ffn = self.fc2(x_ffn)
        x_ffn = self.dropout2(x_ffn)
        x = x + x_ffn

        return x


class UViT(nn.Module):
    """U-shaped Vision Transformer for diffusion models.

    This model processes images as sequences of patches with transformer blocks,
    using U-Net style skip connections between encoder and decoder layers.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.0,
        patch_stride: Optional[int] = None,
        skip_connection_spacing: int = 1,
    ):
        """
        Args:
            image_height: Height of input images
            image_width: Width of input images
            channels: Number of image channels
            patch_size: Size of image patches
            embed_dim: Embedding dimension (d_model)
            depth: Number of transformer layers (must be even for U-shape)
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            time_emb_dim: Time embedding dimension (if None, uses embed_dim)
            dropout: Dropout rate
            patch_stride: Stride for patchify convolution (if None, uses patch_size for non-overlapping patches)
            skip_connection_spacing: Spacing between skip connections (1 = every block, 2 = every other block, etc.)
                                    As per U-ViT paper, long skip connections should bypass multiple blocks
        """
        super().__init__()

        assert depth % 2 == 0, "depth must be even for U-shaped architecture"

        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.skip_connection_spacing = skip_connection_spacing

        # Calculate number of patches based on stride
        # Formula: (input_size - kernel_size) / stride + 1
        self.num_patches_h = (image_height - patch_size) // self.patch_stride + 1
        self.num_patches_w = (image_width - patch_size) // self.patch_stride + 1
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patchify: Convert image to patches with specified stride
        # Using Conv2d is more efficient than manual patching
        self.patchify = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=self.patch_stride,
        )

        # positional embedding for patches
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # time embedding
        if time_emb_dim is None:
            time_emb_dim = embed_dim
        self.time_emb_dim = time_emb_dim

        # sinusoidal time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.num_encoder_layers = depth // 2
        self.num_decoder_layers = depth // 2

        # encoder blocks
        self.shallow_blocks = nn.ModuleList(
            [
                ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(self.num_encoder_layers)
            ]
        )

        # decoder blocks
        self.deep_blocks = nn.ModuleList(
            [
                ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(self.num_decoder_layers)
            ]
        )

        # skip connection projections
        self.skip_indices = list(
            range(0, self.num_decoder_layers, skip_connection_spacing)
        )
        self.skip_projections = nn.ModuleList(
            [nn.Linear(embed_dim * 2, embed_dim) for _ in self.skip_indices]
        )

        # long skip connection
        self.long_skip_proj = nn.Linear(embed_dim * 2, embed_dim)

        # unpatchify: convert patches embeddings back to image
        self.unpatchify = nn.ConvTranspose2d(
            embed_dim,
            channels,
            kernel_size=patch_size,
            stride=self.patch_stride,
        )

        # smoothing convolution to remove patch artifacts
        self.smoothen = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (batch, channels, height, width)
            t: Timestep tensor of shape (batch,) or (batch, 1)

        Returns:
            Denoised image of shape (batch, channels, height, width)
        """
        batch_size = x.shape[0]

        # Handle timestep shape
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.float()

        x = self.patchify(x)  # (batch, channels, H, W) -> (batch, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (batch, embed_dim, H', W') -> (batch, num_patches, embed_dim)

        input_patches = x.clone()  # (batch, num_patches, embed_dim)
        x = x + self.pos_emb

        # embed time and add as a token
        time_emb = self.time_mlp(t)  # (batch, embed_dim)
        time_token = time_emb.unsqueeze(1)  # (batch, 1, embed_dim)
        x = torch.cat([time_token, x], dim=1)  # (batch, num_patches + 1, embed_dim)

        # Encoder: Apply shallow blocks and store skip connections
        skip_connections = []
        for i, block in enumerate(self.shallow_blocks):
            x = block(x)
            # Only store skip connections at specified spacing intervals
            # Store from the end of encoder to match with decoder blocks
            if (self.num_encoder_layers - 1 - i) in self.skip_indices:
                skip_connections.append(x)

        # Decoder: Apply deep blocks with spaced skip connections
        # Skip connections were stored in reverse order during encoding
        skip_idx = 0
        for i, block in enumerate(self.deep_blocks):
            x = block(x)
            # Apply skip connection only at specified spacing intervals
            if i in self.skip_indices and skip_idx < len(skip_connections):
                skip = skip_connections[skip_idx]
                x = torch.cat([x, skip], dim=-1)
                x = self.skip_projections[skip_idx](x)
                skip_idx += 1

        # Remove time token: (batch, num_patches + 1, embed_dim) -> (batch, num_patches, embed_dim)
        x = x[:, 1:, :]

        # Apply long skip connection from input patches (helps reconstruction)
        # Concatenate with input patches and project
        x = torch.cat([x, input_patches], dim=-1)  # (batch, num_patches, embed_dim * 2)
        x = self.long_skip_proj(x)  # (batch, num_patches, embed_dim)

        # Rearrange back to 2D: (batch, num_patches, embed_dim) -> (batch, embed_dim, H', W')
        # Using reshape is faster than einops for simple reshaping
        x = x.transpose(1, 2).reshape(
            batch_size, self.embed_dim, self.num_patches_h, self.num_patches_w
        )

        # Unpatchify: (batch, embed_dim, H', W') -> (batch, channels, H, W)
        x = self.unpatchify(x)

        # Handle size mismatch when using overlapping patches
        if x.shape[2] != self.image_height or x.shape[3] != self.image_width:
            # Crop or pad to match original dimensions
            if x.shape[2] > self.image_height or x.shape[3] > self.image_width:
                x = x[:, :, : self.image_height, : self.image_width]
            else:
                # Pad if needed (shouldn't happen with proper stride calculation)
                pad_h = self.image_height - x.shape[2]
                pad_w = self.image_width - x.shape[3]
                x = F.pad(x, (0, pad_w, 0, pad_h))

        # Smooth to remove patch artifacts
        x = self.smoothen(x)

        return x
