import os
import numpy as np
import keras
import skimage
import tensorflow as tf


def rgb2yuv(x):
    '''
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    U = 0.492 * (b - Y)
    V = 0.877 * (r - Y)
    '''
    #reshape to 3 chnls, flattened batch
    warp = tf.reshape(x,(-1,3))
    print(warp.shape)
    
    #define yuv in terms of reshape
    ry = tf.constant([[0.299, -0.147, 0.615], [ 0.587, -0.289, -0.515], [0.114, 0.436, -0.1]])

    yuv = tf.matmul(warp,ry)
    
    #reshape to yuv batch
    res = tf.reshape(yuv,tf.shape(x))

    return res



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
