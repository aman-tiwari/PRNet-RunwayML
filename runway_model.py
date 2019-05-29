import sys
import os.path
sys.path.append(os.path.abspath("./PRNET"))

import numpy as np
from PIL import Image
import runway
from runway.data_types import image

from api import PRN

ZEROS = np.zeros((256, 256, 4), dtype=np.uint8)

def preprocess(img):
    arr = np.array(img).astype(np.float32)
    arr /= 255.0
    return arr

def postprocess(uv):
    return uv.astype(np.uint8)

@runway.setup
def setup():
    return PRN(prefix='PRNet')

@runway.command('process', inputs={ 'photo': image(width=256, height=256, channels=3) }, 
                            outputs={ 'uv': image(width=256, height=256, channels=3) })
def process(prn, input_):
    img = input_['photo']
    arr = preprocess(img)
    uv = prn.net_forward(arr)
    # If no face is detected the image, process returns None
    if uv is None:
        return { 'uv': ZEROS }
    uv = postprocess(uv)
    return { 'uv': uv }

if __name__ == '__main__':
    runway.run()