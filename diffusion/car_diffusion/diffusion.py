import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import Config
from .model_factory import create_model

class DiffusionModel(nn.Module):
    """
    Implements a basic DDPM training and sampling wrapper around a UNet.

    Args:
        model: UNet model (predicts noise ε given x_t and t)
        image_size: int, height/width of input images
        device: torch.device
        timesteps: number of diffusion steps (default: 1000)
        beta_start, beta_end: range for linear beta schedule
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: tuple,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.image_size = image_size
        self.timesteps = timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        # Precompute useful constants
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (add noise) at a given timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        """
        Sample x_{t-1} from x_t using the model’s predicted noise.
        """
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]

        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return model_mean
        else:
            posterior_var_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, batch_size=16, num_steps=500, channels=3):
        """
        Generate samples by reversing the diffusion process.
        """
        x = torch.randn(batch_size, channels, self.image_size[0], self.image_size[1], device=self.device)
        for i in tqdm(reversed(range(num_steps)), desc="Sampling"):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x

    def training_step(self, x):
        """
        Compute one training step loss.
        """
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        loss = self.p_losses(x, t)
        return loss

    def train_loop(self, dataloader, optimizer, epochs=10, log_interval=100):
        self.train()
        for epoch in range(epochs):
            for step, (x, _) in enumerate(dataloader):
                x = x.to(self.device)

                optimizer.zero_grad()
                loss = self.training_step(x)
                loss.backward()
                optimizer.step()

                if step % log_interval == 0:
                    print(f"Epoch {epoch} Step {step}: Loss = {loss.item():.4f}")

def build_model(config: Config) -> DiffusionModel:
    print("Initializing model and DiffusionModel...")
    im_w, im_h = config.data.image_width, config.data.image_height

    # Use model factory to create the appropriate architecture
    model = create_model(
        config=config.model,
        image_height=im_h,
        image_width=im_w,
    )

    diffusion = DiffusionModel(
        model=model,
        image_size=(im_h, im_w),
        timesteps=config.model.num_timesteps,
        beta_start=config.model.beta_start,
        beta_end=config.model.beta_end,
    )

    return diffusion
