"""Because stanford cars dataset images are not directly available anymore,
this script downloads the dataset using the datasets library and saves all images to a specified directory.
We mix train and test splits to have more data for training the diffusion model.
This is as bit ugly, but gets the job done.
"""

import os
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

def save_image(image, out_path):
    try:
        image.convert("RGB").save(out_path)
    except Exception as e:
        print(f"Failed saving image {out_path}: {e}")

def download_and_merge(base_outdir, max_workers: int = 8):
    os.makedirs(base_outdir, exist_ok=True)

    dataset = load_dataset("tanganke/stanford_cars")

    for split in ["train", "test"]:
        ds = dataset[split]
        print(f"Processing split '{split}' with {len(ds)} images...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, example in enumerate(ds):
                image = example["image"]
                out_name = f"{split}_{idx:05d}.png"
                out_path = os.path.join(base_outdir, out_name)
                futures.append(executor.submit(save_image, image, out_path))

            # todo: show progress
            for _ in as_completed(futures):
                pass

    print("images saved to", base_outdir)

if __name__ == "__main__":
    PATH = os.path.expanduser("~/dataset/car_images")
    download_and_merge(PATH)
