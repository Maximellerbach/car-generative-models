from keras.models import load_model
import cv2
from tqdm import tqdm
import numpy as np
from glob import glob
import time


gen = load_model('vroum\\vroumgen.h5')
dis = load_model('vroum\\vroumdis.h5')

en = load_model('encoder.h5')
de = load_model('decoder.h5')

dos = glob('car_img\\*')


while(1):

    
    rand_noise = np.random.normal(0, 1, (1, 100))
    pred = gen.predict(rand_noise)
    confidence = dis.predict(pred)

    gen_img = (0.5 * pred[0] + 0.5)*255

    cv2.imwrite('gen\\'+str(time.time())+'_'+str(confidence[0][0])+'.png', gen_img)
    
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