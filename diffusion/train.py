"""Train script for diffusion model."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from car_diffusion import Config, DiffusionModel, CarImageDataset, Trainer, create_model


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

    train_dataset = CarImageDataset(
        dataset_path=config.data.dataset_path,
        image_height=config.data.image_height,
        image_width=config.data.image_width,
        split="train",
        train_split=config.data.train_split,
    )

    val_dataset = CarImageDataset(
        dataset_path=config.data.dataset_path,
        image_height=config.data.image_height,
        image_width=config.data.image_width,
        split="val",
        train_split=config.data.train_split,
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


def build_model(config):
    print("Initializing model and DiffusionModel...")

    # Use model factory to create the appropriate architecture
    model = create_model(
        config=config.model,
        image_height=config.data.image_height,
        image_width=config.data.image_width,
    )

    diffusion = DiffusionModel(
        model=model,
        image_size=(config.data.image_height, config.data.image_width),
        timesteps=config.model.num_timesteps,
        beta_start=config.model.beta_start,
        beta_end=config.model.beta_end,
    )


    return diffusion


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
    )

    # Resume training if checkpoint provided
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"Resuming training from checkpoint: {ckpt_path}")
            trainer.load_checkpoint(ckpt_path)
        else:
            print(f" Warning: Checkpoint not found at {ckpt_path}. Starting fresh.")

    # Train the model
    print("Starting training loop...")
    trainer.train(epochs=config.training.epochs)

    # Save final model
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = checkpoint_dir / "final_model.pt"

    torch.save(diffusion_model.state_dict(), final_model_path)
    print(f"Training complete! Final model saved at: {final_model_path}")


if __name__ == "__main__":
    main()
