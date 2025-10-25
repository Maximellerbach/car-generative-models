"""Training loop for diffusion models."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .diffusion import DiffusionModel


class Trainer:
    """Trainer for diffusion models."""

    def __init__(
        self,
        model: DiffusionModel,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint: Path | None = None,
    ):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = torch.device(
            config.training.device
            if torch.cuda.is_available() and config.training.device == "cuda"
            else "cpu"
        )
        self.model = self.model.to(self.device)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

        self.model = torch.compile(self.model)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=config.training.weight_decay,
            eps=1e-8,
        )

        # AMP setup
        self.use_amp = config.training.use_amp
        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.loss_fn = nn.MSELoss()
        self.gen_dir = Path("gen")
        self.gen_dir.mkdir(exist_ok=True)

        self.global_step = 0
        self.start_epoch = 0

        if checkpoint:
            self.load_checkpoint(checkpoint)
            print(f"Resuming training from checkpoint: {checkpoint}")

    def train(self):
        self.model.train()

        epochs = self.config.training.epochs
        for epoch in range(self.start_epoch, epochs):
            epoch_loss = 0.0

            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=80)
            for step, batch in pbar:
                batch = batch.to(self.device)
                batch_size = batch.shape[0]

                t = torch.randint(
                    0, self.model.timesteps, (batch_size,), device=self.device
                )

                self.optimizer.zero_grad()

                if self.scaler:
                    with autocast('cuda'):
                        loss = self.model.p_losses(batch, t)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.model.p_losses(batch, t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm,
                    )
                    self.optimizer.step()

                epoch_loss += loss.item()
                self.global_step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "epoch": f"{epoch + 1}/{epochs}"})

            # Validation
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(epoch + 1)

            if (epoch + 1) % self.config.training.sample_interval == 0:
                # Generate samples
                self.generate_and_save_sample(epoch=epoch + 1, num_images=4)

    def validate(self) -> float:
        self.model.eval()
        val_loss = 0.0
        num_steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                batch_size = batch.shape[0]
                t = torch.randint(0, self.model.timesteps, (batch_size,), device=self.device)
                loss = self.model.p_losses(batch, t)
                val_loss += loss.item()
                num_steps += 1

        self.model.train()
        return val_loss / num_steps if num_steps > 0 else 0.0

    def save_checkpoint(self, epoch: int) -> str:
        checkpoint_dir = self.config.training.checkpoint_dir
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_dict = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.use_amp and self.scaler is not None:
            checkpoint_dict["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint_dict, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)

    def generate_and_save_sample(self, epoch: int, num_images: int = 4):
        self.model.eval()
        with torch.no_grad():
            x = self.model.sample(batch_size=num_images, num_steps=self.config.generation.num_inference_steps, channels=self.config.model.channels)
            x = torch.clamp(x, -1.0, 1.0)
            x = (x + 1.0) / 2.0

            for i in range(num_images):
                img_tensor = x[i].permute(1, 2, 0).cpu().numpy()
                img_tensor = (img_tensor * 255).astype(np.uint8)

                if img_tensor.shape[2] == 1:
                    img_tensor = img_tensor.squeeze(2)
                else:
                    img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_RGB2BGR)

                save_path = self.gen_dir / f"ep_{epoch}_img_{i:02d}.png"
                cv2.imwrite(str(save_path), img_tensor)
        self.model.train()

    def load_checkpoint(self, checkpoint_path: str | Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.use_amp and self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("Restored AMP scaler state")

        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"]
            print(f"Resuming from epoch {self.start_epoch}")

        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
            print(f"Resuming from global step {self.global_step}")

        print(f"Checkpoint loaded from {checkpoint_path}")
