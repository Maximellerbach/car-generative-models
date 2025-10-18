import os
import imageio
from glob import glob
from tqdm import tqdm

# Initialize paths
base_dir = os.path.dirname(os.path.abspath(__file__))
car_dir = os.path.join(base_dir, "car")

filenames = glob(os.path.join(car_dir, '*'))  # load training images

output_path = os.path.join(base_dir, 'training.gif')
with imageio.get_writer(output_path, mode='I', duration=0.033) as writer:
    for filename in tqdm(filenames):
        image = imageio.imread(filename)
        writer.append_data(image)
