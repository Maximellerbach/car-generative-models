import os

from keras.models import load_model
import cv2
from tqdm import tqdm
import numpy as np
from glob import glob
import time

# Initialize paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
car_img_dir = os.path.join(base_dir, "car")
gen_dir = os.path.join(base_dir, "gen")
os.makedirs(gen_dir, exist_ok=True)

gen = load_model(os.path.join(models_dir, 'vroumgen.h5'))
dis = load_model(os.path.join(models_dir, 'vroumdis.h5'))

en = load_model(os.path.join(base_dir, 'encoder.h5'))
de = load_model(os.path.join(base_dir, 'decoder.h5'))

dos = glob(os.path.join(car_img_dir, '*'))


while(1):

    
    rand_noise = np.random.normal(0, 1, (1, 100))
    pred = gen.predict(rand_noise)
    confidence = dis.predict(pred)

    gen_img = (0.5 * pred[0] + 0.5)*255

    output_path = os.path.join(gen_dir, f'{time.time()}_{confidence[0][0]}.png')
    cv2.imwrite(output_path, gen_img)
    
'''

for i in dos:
    img = cv2.imread(i)
    img = cv2.resize(img, (150, 100))

    pred = np.expand_dims(img/255, axis=0)
    
    y = en.predict(pred)
    enimg = de.predict(y)[0]
    enimg = (enimg*0.5)+0.5

    cv2.imshow('img', img)
    cv2.imshow('en', enimg)

    cv2.waitKey(0)
'''