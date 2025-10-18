import imageio
from glob import glob
from tqdm import tqdm

filenames = glob('car\\*')#load training image

with imageio.get_writer('training.gif', mode='I',duration=0.033) as writer:
    for filename in tqdm(filenames):
        image = imageio.imread(filename)
        writer.append_data(image)
