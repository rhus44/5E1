import os
import numpy as np
import keras
import skimage

class Save_predictions(keras.callbacks.Callback):
    def __init__(self, save_file):
        self.savf = save_file
        self.opendir = r'C:\Users\buggyr\Mosaic_Experiments\data\processed\Callback_test'

    def on_epoch_end(self, epoch, logs={}):
        for filename in os.listdir(self.opendir):
            sc=os.path.join(self.opendir,filename)
            nup = np.load(sc)
            pred = self.model.predict(np.array([nup]),batch_size=1)[0]
            skimage.io.imsave(os.path.join(self.savf,str(epoch) + '_' + os.path.splitext(filename)[0])+'.png',pred)
