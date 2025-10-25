import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    dataset_path: Path
    batch_size: int = 32
    image_height: int = 100
    image_width: int = 150
    num_workers: int = 4

@dataclass
class ModelConfig:
    channels: int = 3
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    
    # Model architecture selection
    architecture: str = "unet"  # Options: "unet", "uvit", "uvit_small", "uvit_base", "uvit_large"
    
    # UNet architecture parameters
    model_channels: int = 32  # Base number of channels in the UNet
    channel_multipliers: tuple = (1, 2)  # Channel multipliers for each resolution level
    num_res_blocks: int = 1  # Number of residual blocks per resolution level
    time_emb_dim: int = 256  # Dimension of time embedding
    
    # Attention parameters (for UNet)
    use_attention_at: tuple = (False, True)  # Which resolution levels get attention
    
    # Vision Transformer (ViT/UViT) parameters
    patch_size: int = 5  # Size of patches for ViT
    patch_stride: int | None = None
    embed_dim: int = 384  # Embedding dimension for ViT
    depth: int = 12  # Number of transformer layers
    num_heads: int = 6  # Number of attention heads
    mlp_ratio: float = 4.0  # MLP hidden dimension ratio
    vit_dropout: float = 0.0  # Dropout rate for ViT
    skip_connection_spacing: int = 1  # Spacing between skip connections in UViT
    
    def get(self, key: str, default=None):
        """Get attribute with default fallback."""
        return getattr(self, key, default)


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-4
    checkpoint_dir: Path = Path("checkpoints")
    save_interval: int = 10
    sample_interval: int = 5
    device: str = "cuda"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    use_amp: bool = True  # Enable automatic mixed precision by default

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default=None):
        """Get attribute with default fallback."""
        return getattr(self, key, default)


@dataclass
class GenerationConfig:
    num_images: int = 16
    num_inference_steps: int = 50
    save_dir: Path = Path("outputs")

    def __post_init__(self):
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig

    @staticmethod
    def from_yaml(config_path: Path) -> "Config":
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        if "data" not in config_dict or "dataset_path" not in config_dict["data"]:
            raise ValueError(
                "dataset_path must be specified in config under 'data' section. "
                "This should point to the directory containing your images."
            )
        
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        generation_config = GenerationConfig(**config_dict.get("generation", {}))
        
        return Config(
            data=data_config,
            model=model_config,
            training=training_config,
            generation=generation_config,
        )
    
    def save_to_yaml(self, config_path: Path):
        config_dict = {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "generation": self.generation.__dict__,
        }
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
