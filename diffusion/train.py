"""Train script for diffusion model."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from car_diffusion import Config, CustomImageDataset, Trainer, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a car diffusion model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional: Path to checkpoint to resume training from.",
    )
    return parser.parse_args()


def build_dataloaders(config):
    print(f"Loading dataset from {config.data.dataset_path}")

    transform = transforms.Compose([
        transforms.Resize((int(config.data.image_height * 1.2), int(config.data.image_width * 1.2))),
        # data augm
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, expand=True),
        transforms.CenterCrop((config.data.image_height, config.data.image_width)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # normalize to [-1, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = CustomImageDataset(
        dataset_path=config.data.dataset_path,
        split="train",
        transform=transform,
    )

    val_dataset = CustomImageDataset(
        dataset_path=config.data.dataset_path,
        split="val",
        transform=transform,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = Config.from_yaml(config_path)
    print("Configuration loaded successfully.")
    print(f"Dataset path: {config.data.dataset_path}")
    print(f"Training on device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Config summary:\n{config}")

    # Build datasets & dataloaders
    train_loader, val_loader = build_dataloaders(config)

    # Build model & trainer
    diffusion_model = build_model(config)
    trainer = Trainer(
        model=diffusion_model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint=Path(args.resume) if args.resume else None,
    )

    # Train the model
    print("Starting training loop...")
    trainer.train()

    # Save final model
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = checkpoint_dir / "final_model.pt"

    torch.save(diffusion_model.state_dict(), final_model_path)
    print(f"Training complete! Final model saved at: {final_model_path}")


if __name__ == "__main__":
    main()
