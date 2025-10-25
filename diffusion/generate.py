"""Generate images using trained diffusion model."""

import argparse
from pathlib import Path

import torch
from torchvision import transforms

from car_diffusion import Config, DiffusionModel, plot_images
from train import build_model

def generate_images(config: Config, model: DiffusionModel, device: torch.device) -> torch.Tensor:
    """Generate images using the diffusion model."""
    model = model.to(device)
    model.eval()

    num_images = config.generation.num_images
    batch_size = config.data.batch_size
    all_images = []

    with torch.no_grad():
        for _ in range((num_images + batch_size - 1) // batch_size):
            current_batch_size = min(batch_size, num_images - len(all_images))
            samples = model.sample(
                batch_size=current_batch_size,
                num_steps=config.generation.num_inference_steps,
                channels=config.model.channels,
            )
            all_images.append(samples.cpu())

    all_images = torch.cat(all_images, dim=0)[:num_images]
    return all_images


def main():
    parser = argparse.ArgumentParser(description="Generate images using diffusion model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path("outputs/generated_images.png")
    config = Config.from_yaml(config_path)

    device = torch.device(
        config.training.device
        if torch.cuda.is_available() and config.training.device == "cuda"
        else "cpu"
    )

    model = build_model(config)
    model.compile()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    # # TODO: find a better way to handle this
    # # Handle state dict from compiled model (torch.compile adds _orig_mod. prefix)
    # if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    #     state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    generated_images = generate_images(config, model, device)

    # Denormalize images from [-1, 1] to [0, 1]
    generated_images = generated_images.clamp(-1, 1)
    generated_images = (generated_images + 1) / 2.0
    generated_images = generated_images.detach().cpu().numpy()
    
    # Save main output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_images(generated_images, output_path)
    print(f"Generated images saved to {output_path}")

if __name__ == "__main__":
    main()