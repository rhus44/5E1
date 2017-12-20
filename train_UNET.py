import cv2
import numpy as np
import math
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
    def __init__(self, save_pred):
        self.save_pred = save_pred
        self.opendir = r'C:\Users\buggyr\Mosaic_Experiments\data\processed\Callback_test'

    def on_epoch_end(self, epoch, logs={}):
        for filename in os.listdir(self.opendir):
            dr=os.path.join(self.opendir,filename)
            nup = np.load(dr)
            
            nup =get_unet_input(nup)    
            pred = self.model.predict(np.array([nup]),batch_size=1)[0]
            
            pred = denormalise1(pred)
            
            sv = os.path.join(self.save_pred,str(epoch) + '_' + os.path.splitext(filename)[0])
            np.save(sv,pred)
            skimage.io.imsave(sv + '.png', (cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)))
            


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
        
        train_dir = self.train_dir
        batch_size = self.patch_size
        patch_size = self.patch_size
        
        img_name = os.listdir(train_dir)
        # Repeat inner loops based on number of backgrounds to composite on    
        # Loop through the image names according to batchsize 
        while True:
            img_name = os.listdir(train_dir)


            rem_mos = []
            rem_orig = []
            for img  in enumerate(img_name):

                sc = os.path.join(self.train_dir,img[1])
                #print(sc)
                try:
                    im = cv2.imread(sc)

                    img_ptch = extractPatches(im,patch_size) 
                    print(img_ptch.shape)
                    mos = mosaic(img_ptch)

                    input_mos = np.zeros((batch_size,patch_size,patch_size,4))
                    imput_orig = np.zeros((batch_size,patch_size,patch_size,3))
                    num_ptchs = mos.shape[0]


                    btchs = int(num_ptchs/self.batch_size)
                    #print(str(btchs))
                    for i in range(btchs):
                        b = i*batch_size
                        input_mos = mos[b:b+self.batch_size]
                        input_orig = img_ptch[b:b+self.batch_size]
                        print(input_mos.shape)
                        if(input_mos.shape[1]==32):
                            return normalise(input_mos), normalise(input_orig)                    


                except Exception as e:
                    print('file open error: ' + str(e))                
    
    def on_epoch_end(self):
            return

        
        
def get_unet_input(array):
    y = []
    #print(str(array.shape))
    h,w,c = array.shape
    for x in range(0,max(h,w)):
        x1=x
        if(x%32 ==0):
            y.append(x1)
            
    h1 = max(int(t) for t in y if h>=t)
    w1 = max(int(t) for t in y if w>=t)
    return array[:h1,:w1:,:]
    
  

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
                    
    return np.stack((g1,b,g2,r),-1)


def mosaic1(im_ptchs):
          
    g1 = im_ptchs[::2,::2,1]
    #print(g1.shape)
    r  = im_ptchs[::2,1::2,2]
    #print(r.shape)
    g2 = im_ptchs[1::2,::2,1]
    #print(g2.shape)
    b  = im_ptchs[1::2,1::2,0]     
    #print(b.shape)
                    
    return np.stack((g1,b,g2,r),-1)


def normalise(array):
    return ((array/127.5) - 1).astype(np.float32)


def denormalise(array):
    return ((array + 1)*127.5).astype(np.uint8)



def normalise1(array):
    return (array/255).astype(np.float32)



def denormalise1(array):
    return ((array)*255).astype(np.uint8)


def addBayer(orig, predicted):

    pred1 = predicted
    
    predicted[::2,::2,1] = orig[::2,::2,1]
    
    predicted[::2,1::2,2] = orig[::2,1::2,2]
    
    predicted[1::2,::2,1] = orig[1::2,::2,1]

    predicted[1::2,1::2,0] = orig[1::2,1::2,0]
    
    diff = pred1 - predicted
    
    return predicted



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
                #print(img_ptch.shape)
                mos = mosaic(img_ptch)
                
                input_mos = np.zeros((batch_size,patch_size,patch_size,4))
                imput_orig = np.zeros((batch_size,patch_size,patch_size,3))
                num_ptchs = mos.shape[0]
                 
                
                btchs = int(num_ptchs/batch_size)
                #print(str(btchs))
                for i in range(btchs):
                    b = i*batch_size
                    input_mos = mos[b:b+batch_size]
                    input_orig = img_ptch[b:b+batch_size]
                    #print(input_mos.shape)
                    
                    yield normalise(input_mos), normalise(input_orig)                    
                    
                
            except Exception as e:
                print('file open error: ' + str(e))                
    return



def val_generator_rgb(train_dir):
    
    while True:
        img_name = os.listdir(train_dir)
               
        for img  in enumerate(img_name):

            sc = os.path.join(train_dir,img[1])
            print(sc)
            
            try:
                im = cv2.imread(sc)

                im = get_unet_input(im)

                mos = np.array([mosaic1(im)])
                print(normalise1(mos).shape)

                yield normalise(mos), normalise(np.array([im]))                  
                #print('Batched Input')
                                  
            except Exception as e:
                print('file open error: ' + str(e))
                  
    return



def predict_generator_rgb(train_dir):
    '''
    Python generator that loads imgs and batches
    '''
    
    while True:
        img_name = os.listdir(train_dir)
               
        for img  in enumerate(img_name):

            sc = os.path.join(train_dir,img[1])
            print(sc)
            
            try:
                im = cv2.imread(sc)

                im = get_unet_input(im)

                mos = np.array([mosaic1(im)])
           

                yield normalise1(mos), im, img[1]                  
                #print('Batched Input')
                                  
            except Exception as e:
                print('file open error: ' + str(e))
                  
    return



def predict_generator(model,k_gen,steps,data,bp, save_file):
    
    psnr = []
    ssim = []
    
    for i in range(steps):
        p_img_gen, orig_gen, img_name = next(k_gen)

        pred_img = (model.predict(p_img_gen,batch_size=1))[0]
        pred_img[(pred_img > 1)] = 1
        pred_img = denormalise1(pred_img)
        print(pred_img.min())
        print(pred_img.max())
        
        cv2.imshow('image',pred_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
        
        pred_img = addBayer(orig_gen, pred_img)
        
        cv2.imshow('image',abs((pred_img-orig_gen)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        #cv2.imshow('image',pred_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        #cv2.imwrite(os.path.join(save_file,('predicted_'+img_name)),pred_img)
        
        psnr.append(skimage.measure.compare_psnr(pred_img[bp:-bp,bp:-bp,:],orig_gen[bp:-bp,bp:-bp,:]))
        ssim.append(skimage.measure.compare_ssim(pred_img[bp:-bp,bp:-bp,:],orig_gen[bp:-bp,bp:-bp,:],multichannel=True))
               
    psnr_avg = sum(psnr)/steps
    ssim_avg = sum(ssim)/steps
    print('Kodak PSNR Average: '+str(psnr_avg))
    print('Kodak SSIM Average: '+str(ssim_avg))
    
    return psnr, ssim, psnr_avg, ssim_avg
    
