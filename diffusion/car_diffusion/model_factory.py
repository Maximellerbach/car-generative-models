"""Factory for creating different model architectures."""

from typing import Union
import torch.nn as nn

from .unet_model import UNet
from .transformer_model import UViT
from .config import ModelConfig


def create_model(
    config: ModelConfig,
    image_height: int,
    image_width: int
) -> nn.Module:
    """Create a model based on the configuration.
    
    Args:
        config: Model configuration
        image_height: Height of input images
        image_width: Width of input images
    
    Returns:
        Model instance (UNet or UViT)
    
    Raises:
        ValueError: If architecture type is not supported
    """
    architecture = config.architecture.lower()
    
    if architecture == "unet":
        print("Creating UNet model...")
        model = UNet(
            image_height=image_height,
            image_width=image_width,
            channels=config.channels,
            model_channels=config.model_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            time_emb_dim=config.time_emb_dim,
            use_attention_at=tuple(config.use_attention_at),
        )
        
    elif architecture == "uvit":
        print("Creating UViT model...")
        model = UViT(
            image_height=image_height,
            image_width=image_width,
            channels=config.channels,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            patch_stride=config.patch_stride,
            skip_connection_spacing=config.skip_connection_spacing,
        )
        
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported architectures: unet, uvit"
        )
    
    # Print model information
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {architecture}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    return model
