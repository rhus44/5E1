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
            
            pred = denormalise(pred)
            
            sv = os.path.join(self.savf,str(epoch) + '_' + os.path.splitext(filename)[0])
            np.save(sv,pred)
            skimage.io.imsave(sv + '.png', (cv2.cvtColor(pred,cv2.COLOR_YUV2BGR)))
            
            
            
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
                    
    return np.stack((g1,b,g2,r),-1)



def normalise(array):
    return ((array/127.5) - 1).astype(np.float32)



def denormalise(array):
    return ((array + 1)*127.5).astype(np.uint8)



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


def predict_generator_yuv(train_dir):
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

                mos = np.array([mosaic1(im)])
                
                yield normalise(mos), im, img[1]                   
                    #print('Batched Input')
                                  
            except Exception as e:
                print('file open error: ' + str(e))
                 
    return


def predict_generator(model,k_gen,steps,data,bp,save_file):
    
    psnr = []
    ssim = []
    
    for i in range(steps):
        p_img_gen, orig_gen, img_name = next(k_gen)

        pred_img = (model.predict(p_img_gen,batch_size=1))[0]
        pred_img = denormalise(pred_img)
        pred_img_bgr = cv2.cvtColor(pred_img,cv2.COLOR_YUV2BGR)
        print(pred_img_bgr.max())
        print(pred_img_bgr.min())
        
        cv2.imshow('image',np.concatenate((orig_gen, pred_img_bgr), axis=1))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #skimage.io.imsave(os.path.join(save_file,('predicted_'+img_name)), (cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)))
        cv2.imwrite(os.path.join(save_file,('predicted_'+img_name)),pred_img_bgr)
        
        
        psnr.append(skimage.measure.compare_psnr(pred_img_bgr[bp:-bp,bp:-bp,:],orig_gen[bp:-bp,bp:-bp,:]))
        ssim.append(skimage.measure.compare_ssim(pred_img_bgr[bp:-bp,bp:-bp,:],orig_gen[bp:-bp,bp:-bp,:],multichannel=True))
               
    psnr_avg = sum(psnr)/steps
    ssim_avg = sum(ssim)/steps
    print('Kodak PSNR Average: '+str(psnr_avg))
    print('Kodak SSIM Average: '+str(ssim_avg))
    
    return psnr, ssim, psnr_avg, ssim_avg
    
