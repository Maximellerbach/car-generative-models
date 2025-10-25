from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_images(
    images: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Generated Images",
    num_images: int = 16,
) -> None:
    if images.shape[1] in [1, 3]:
        images = np.transpose(images, (0, 2, 3, 1))
    
    num_images = min(num_images, len(images))
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))
    
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx in range(num_images):
        ax = axes[idx]
        img = images[idx]
        
        if img.shape[2] == 1:
            img = img.squeeze(2)
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        
        ax.axis("off")
    
    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    plt.show()
