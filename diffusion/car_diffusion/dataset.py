from pathlib import Path
from typing import Optional, List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        split: str = "train",
        train_split: float = 0.8,
        transform: Optional[transforms.Compose] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform

        all_images: List[Path] = sorted(
            [
                p
                for p in self.dataset_path.glob("*")
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        num_train = int(len(all_images) * train_split)
        self.images = (
            all_images[:num_train]
            if self.split == "train"
            else all_images[num_train:]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
