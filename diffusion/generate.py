"""Generate images using trained diffusion model."""

import argparse
from pathlib import Path

import torch

from car_diffusion import Config, DiffusionModel, create_model
from car_diffusion.visualization import plot_images


def generate_images(checkpoint_path: str, config_path: str):
    config = Config.from_yaml(Path(config_path))

    device = torch.device(
        config.training.device
        if torch.cuda.is_available() and config.training.device == "cuda"
        else "cpu"
    )

    print("Creating model...")
    model = create_model(
        config=config.model,
        image_height=config.data.image_height,
        image_width=config.data.image_width,
    )

    print("Creating diffusion model...")
    diffusion_model = DiffusionModel(
        model=model,
        image_size=(config.data.image_height, config.data.image_width),
        timesteps=config.model.num_timesteps,
        beta_start=config.model.beta_start,
        beta_end=config.model.beta_end,
    )

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    diffusion_model.load_state_dict(checkpoint["model_state_dict"])

    diffusion_model: DiffusionModel = diffusion_model.to(device)
    diffusion_model.eval()

    save_dir = Path(config.generation.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "generated_images.png"

    print(f"Generating {config.generation.num_images} images...")
    with torch.no_grad():
        generated_images = diffusion_model.sample(
            batch_size=config.generation.num_images,
            channels=config.model.channels,
            num_steps=config.generation.num_inference_steps,
        )

        generated_images = torch.clamp(generated_images, -1.0, 1.0)
        generated_images = (generated_images + 1.0) / 2.0

        # Save the generated images
        plot_images(generated_images, save_path=save_path, num_images=len(generated_images))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    generate_images(checkpoint_path=args.checkpoint, config_path=args.config)
