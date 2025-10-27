# Car Generative Models

This repo contains two different approaches to image generation (applied to cars using the Stanford Cars Dataset).

## Diffusion Model

A diffusion approach with two models available:

- U-Net with attention
- U-ViT (Vision Transformer)

Both models work quite well, I was more successful with the U-net because it had fewer parameters and was easier to train.

## DCGAN

A good old GAN using Deep Convolutional layers.
-> Hard to train, but can produce decent images.

This was the first model implemented in this repo quite a long time ago. It is now deprecated due to poor documentation from my end. It would still work if you found all the dependencies, but I would not recommend using it.
