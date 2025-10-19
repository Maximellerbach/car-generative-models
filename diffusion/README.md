# Car Diffusion Model

A modern PyTorch implementation of a diffusion model for generating synthetic car images.

## Training

To train the model from scratch:

```bash
python train.py --config configs/your_config.yaml
```

To resume training from a checkpoint:

```bash
python train.py --config configs/your_config.yaml --resume checkpoints/checkpoint_epoch_0010.pt
```

