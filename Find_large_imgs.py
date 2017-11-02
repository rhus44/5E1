import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import sklearn.feature_extraction
import scipy.ndimage
import time
from skimage import io
from skimage import util
from skimage import measure
import matplotlib.pyplot as plt
import shutil

opendir = r'\\dellserver\js15\buggyr\Documents\My Pictures\Saved Pictures'
savedir = r'\\dellserver\js15\buggyr\Documents\My Pictures\Camera Roll'

for filename in os.listdir(opendir):
    if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".tif"):
        sc=os.path.join(opendir,filename)
        im=io.imread(sc)
        s = im.shape
        if (s[0]*s[1]>600000):
            print(filename)
            shutil.copy(os.path.join(opendir,filename), savedir)

print('Done')
