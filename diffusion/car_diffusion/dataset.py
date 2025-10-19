import os
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CarImageDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        image_height: int = 100,
        image_width: int = 150,
        split: str = "train",
        train_split: float = 0.8,
    ):
        self.dataset_path = Path(dataset_path)
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        
        self.image_paths = self._find_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.dataset_path}")
        
        num_train = int(len(self.image_paths) * train_split)
        if split == "train":
            self.image_paths = self.image_paths[:num_train]
        elif split == "val":
            self.image_paths = self.image_paths[num_train:]
        else:
            raise ValueError(f"Invalid split: {split}")

    def _find_images(self) -> List[str]:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = []
        
        for file_path in sorted(self.dataset_path.rglob("*")):
            if file_path.suffix.lower() in valid_extensions:
                image_paths.append(str(file_path))
        
        return image_paths

    def _load_image(self, file_path: str) -> Optional[np.ndarray]:
        try:
            img = cv2.imread(file_path)
            if img is None:
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = img.astype(np.float32) / 127.5 - 1.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            return img
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._load_image(self.image_paths[idx])
        
        if img is None:
            return torch.randn(3, self.image_height, self.image_width)
        
        return img
