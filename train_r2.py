import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
import os
import keras
import skimage
import tensorflow as tf
from PIL import Image
from keras.utils import Sequence
from keras import backend as K


class Save_predictions(keras.callbacks.Callback):
    def __init__(self, save_file):
        self.savf = save_file
        self.opendir = r'C:\Users\buggyr\Mosaic_Experiments\data\processed\Callback_test'

    def on_epoch_end(self, epoch, logs={}):
        for filename in os.listdir(self.opendir):
            sc=os.path.join(self.opendir,filename)
            nup = np.load(sc)
            pred = self.model.predict(np.array([nup]),batch_size=1)[0]
            
            pred = ((pred + 1)/2).astype(np.float32)
            
            sv = os.path.join(self.savf,str(epoch) + '_' + os.path.splitext(filename)[0])
            np.save(sv,pred)
            #skimage.io.imsave(sv + '.png', (cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)))
            skimage.io.imsave(sv + '.png', (cv2.cvtColor(pred,cv2.COLOR_YUV2RGB)))
            
            
class train_seq(Sequence):

    def __init__(self, train_dir, patch_size, batch_size, shuffle = True):
      
        self.train_dir = train_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        img_names = os.listdir(self.train_dir)
        self.img_names = img_names

    def __len__(self):
        return math.ceil(len(self.img_names) / self.batch_size)

    
    def __getitem__(self, idx):
        img_name = os.listdir(train_dir)
        # Repeat inner loops based on number of backgrounds to composite on    
        # Loop through the image names according to batchsize 
        for i in range(0, len(img_name), batch_size):
            batch_img_names = img_names[i:i+batch_size]
            im_pat_ls=[]
            im_mos_pat_ls=[]
            # Process each image in batch
            for img_name  in enumerate(batch_img_names):
                sc = os.path.join(train_dir,img_name)
                try:
                    im=Image.open(sc)

                    im_raw_mos,im_orig = mosaic(im)

                    patch_mos, patch_orig = get_ptchs(im_raw_mos,im_orig,patch_size)

                    im_pat_ls.extend(patch_orig)
                    im_mos_pat_ls.extend(patch_mos)

                except Exception as e:
                    print('file open error')
            input_im_pat=((np.stack(im_pat_ls,0))/255).astype(np.float32)
            input_im_mos_pat=((np.stack(im_mos_pat_ls,0))/255).astype(np.float32)

            return input_im_mos_pat, input_im_pat
     
    
def yuv_init(shape, dtype=None):
    #RGB2YUV
#     return K.constant([[ 0.29900, -0.147,  0.615],
#                  [0.58700, -0.289, -0.515],
#                  [ 0.11400, 0.436, -0.1]],dtype=dtype,shape=shape)
  #BGR2YUV
    return K.constant([[ 0.11400, 0.436, -0.1],
                 [0.58700, -0.289, -0.515],
                 [ 0.29900, -0.147,  0.615]],dtype=dtype,shape=shape)


def extractPatches(im,patch_size):
    return image.extract_patches_2d(im, (patch_size*2,patch_size*2), max_patches=96)


def mosaic(im_ptchs):
          
    g1 = im_ptchs[:,::2,::2,1]
    #print(g1.shape)
    r  = im_ptchs[:,::2,1::2,2]
    #print(r.shape)
    g2 = im_ptchs[:,1::2,::2,1]
    #print(g2.shape)
    b  = im_ptchs[:,1::2,1::2,0]     
    #print(b.shape)
                    
    return np.stack((g1,b,g2,r),-1), im_ptchs


def mosaic1(im_ptchs):
          
    g1 = im_ptchs[::2,::2,1]
    #print(g1.shape)
    r  = im_ptchs[::2,1::2,2]
    #print(r.shape)
    g2 = im_ptchs[1::2,::2,1]
    #print(g2.shape)
    b  = im_ptchs[1::2,1::2,0]     
    #print(b.shape)
                    
    return np.stack((g1,b,g2,r))


def normalise1(array):
    return (array/255).astype(np.float32)


def normalise(array):
    return ((array/127.5) - 1).astype(np.float32)


def rgb2yuv(array):
    #RGB2YUV
    #m = np.array([[ 0.29900, -0.147,  0.615],
    #             [0.58700, -0.289, -0.515],
    #             [ 0.11400, 0.436, -0.1]])
      
    #BGR2YUV    
    m = np.array([[ 0.11400, 0.436, -0.1],
                 [0.58700, -0.289, -0.515],
                 [ 0.29900, -0.147,  0.615]])
        
    yuv = np.dot(array,m)
 
    return yuv.astype(np.float32)


def train_generator_rgb(train_dir, patch_size, batch_size):
    '''
    Python generator that loads imgs and batches
    '''

    while True:
        img_name = os.listdir(train_dir)
        
        
        rem_mos = []
        rem_orig = []
        for img  in enumerate(img_name):

            sc = os.path.join(train_dir,img[1])
            #print(sc)
            try:
                im = cv2.imread(sc)

                img_ptch = extractPatches(im,patch_size) 

                mos,orig = mosaic(img_ptch)
                
                input_mos = np.zeros((batch_size,patch_size,patch_size,4))
                imput_orig = np.zeros((batch_size,patch_size,patch_size,3))
                num_ptchs = mos.shape[0]
                 
                
                btchs = int(num_ptchs/batch_size)
                #print(str(btchs))
                for i in range(btchs):
                    b = i*batch_size
                    input_mos = mos[b:b+batch_size]
                    input_orig = orig[b:b+batch_size]
                    
                    yield normalise(input_mos), normalise(input_orig)                    
                    #print('Batched Input')
                
                  
            except Exception as e:
                print('file open error: ' + str(e))                
    return

def train_generator_yuv(train_dir, patch_size, batch_size):
    '''
    Python generator that loads imgs and batches
    '''

    while True:
        img_name = os.listdir(train_dir)
        
        
        rem_mos = []
        rem_orig = []
        for img  in enumerate(img_name):

            sc = os.path.join(train_dir,img[1])
            #print(sc)
            try:
                im = cv2.imread(sc)       #BGR output

                img_ptch = extractPatches(im,patch_size)     

                mos,orig = mosaic(img_ptch)
                
                input_mos = np.zeros((batch_size,patch_size,patch_size,4))
                imput_orig = np.zeros((batch_size,patch_size,patch_size,3))
                num_ptchs = mos.shape[0]
                 
                
                btchs = int(num_ptchs/batch_size)    #number of batches/image
                #print(str(btchs))
                for i in range(btchs):
                    b = i*batch_size
                    input_mos = mos[b:b+batch_size]
                    input_orig = orig[b:b+batch_size]
                    
                    #yield normalised batches
                    yield normalise(input_mos), rgb2yuv(normalise(input_orig))                       
            except Exception as e:
                print('file open error: ' + str(e))                
    return


def predict_generator2(train_dir):
    '''
    Python generator that loads imgs and batches
    '''

    while True:
        img_name = os.listdir(train_dir)
               
        for img  in enumerate(img_name):

            sc = os.path.join(train_dir,img[1])
            #print(sc)
            try:
                im = cv2.imread(sc)

                mos = mosaic1(img_ptch)
                
                yield normalise(mos)                  
                    #print('Batched Input')
                                  
            except Exception as e:
                print('file open error: ' + str(e))
                 
    return


def YUV_loss(y_true,y_pred):
    b = y_true[0]
    g = y_true[1]
    r = y_true[2]
    
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    U = 0.492 * (b - Y)
    V = 0.877 * (r - Y)
    
    y_ty = np.stack(Y,U,V)
    
    b = y_pred[0]
    g = y_pred[1]
    r = y_pred[2]
    
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    U = 0.492 * (b - Y)
    V = 0.877 * (r - Y)
    
    y_py = np.stack(Y,U,V)
    
    
    return K.mean(K.square(y_py - y_tp), axis=-1)



