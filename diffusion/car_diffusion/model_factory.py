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
    architecture = config.get("architecture", "unet").lower()
    
    if architecture == "unet":
        print("Creating UNet model...")
        model = UNet(
            image_height=image_height,
            image_width=image_width,
            channels=config.channels,
            model_channels=config.model_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            time_emb_dim=config.get("time_emb_dim", 256),
            use_attention=config.get("use_attention", True),
            attention_resolutions=tuple(config.get("attention_resolutions", [1])),
            attention_type=config.get("attention_type", "channel"),
        )
        
    elif architecture == "uvit":
        print("Creating UViT model...")
        model = UViT(
            image_height=image_height,
            image_width=image_width,
            channels=config.channels,
            patch_size=config.get("patch_size", 8),
            embed_dim=config.get("embed_dim", 384),
            depth=config.get("depth", 12),
            num_heads=config.get("num_heads", 6),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            patch_stride=config.get("patch_stride", None),
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
