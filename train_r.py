import os
import numpy as np
import keras
import skimage
import tensorflow as tf
from PIL import Image


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
    

from PIL import Image
from keras.utils import Sequence


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
            skimage.io.imsave(sv + '.png', (pred*255).astype(np.uint8))

            
            
            
            
            
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
                        
        
        
def mosaic(im):
    h=int(im.size[0]/2) - 1
    w=int(im.size[1]/2) - 1

    #print('w: '+str(w)+'  h: '+str(h))

    chnl = im.split()
    r=np.array(chnl[0])
    g=np.array(chnl[1])
    b=np.array(chnl[2])

    g1=np.zeros((w,h))
    g2=np.zeros((w,h))
    r1=np.zeros((w,h))
    b1=np.zeros((w,h))

    for x in range(0,w):
        for y in range(0,h):
            x1=x*2
            y1=y*2
            #g1[x,y]=(g[x1,y1]+g[x1+1,y1+1])/2
            g1[x,y]=g[x1,y1]
            g2[x,y]=g[x1+1,y1+1]
            r1[x,y]=r[x1+1,y1]
            b1[x,y]=b[x1,y1+1]

    return np.stack((g1,b1,g2,r1),0),np.stack((r,g,b))

def get_ptchs(im_mos,im_orig,sub_image_res):
    c,h,w = im_mos.shape
    pat_ls=[]
    mos_pat_ls=[]
    for i in range (0,99):
        w1=np.random.randint(0,w-sub_image_res-1)
        h1=np.random.randint(0,h-sub_image_res-1)

        w2=w1*2
        h2=h1*2
        image_res=sub_image_res*2

        patch_mos = im_mos[:,h1:h1+sub_image_res,w1:w1+sub_image_res]
        patch_orig = im_orig[:,h2:h2+image_res,w2:w2+image_res]
        im_mos_pat = np.moveaxis(patch_mos, 0, -1)
        im_pat=np.moveaxis(patch_orig, 0, -1)

        mos_pat_ls.append(im_mos_pat)
        pat_ls.append(im_pat)


        return   mos_pat_ls, pat_ls

def savepatch(opendir,savedir,im_mos_pat_ls,im_pat_ls):
    input_im_pat=np.stack(im_pat_ls,0)
    input_im_mos_pat=np.stack(im_mos_pat_ls,0)

    sd=os.path.join(opendir,savedir)
    os.mkdir(sd)
    print('Made '+sd)

    sv1=os.path.join(sd,'input_img_patch')
    np.save(sv1,input_im_pat)

    sv2=os.path.join(sd,'input_img_mos_patch')
    np.save(sv2,input_im_mos_pat)
    print('Saved '+sv2)
    return


def train_generator(train_dir, patch_size, batch_size):
    '''
    Python generator that loads imgs and batches
    '''

    while True:
        img_name = os.listdir(train_dir)
        
        im_pat_ls = []
        im_mos_pat_ls = []
        im_pat_bat_ls = []
        im_mos_pat_bat_ls = []
        for img  in enumerate(img_name):

            sc = os.path.join(train_dir,img[1])

            try:
                im=Image.open(sc)

                im_raw_mos,im_orig = mosaic(im)

                patch_mos, patch_orig = get_ptchs(im_raw_mos,im_orig,patch_size)

                im_pat_ls.extend(patch_orig)
                im_mos_pat_ls.extend(patch_mos)
                
                while (len(im_pat_ls) > 32):
                    im_pat_bat_ls = im_pat_ls[0:32]
                    del im_pat_ls[0:32]
                    im_mos_pat_bat_ls = im_mos_pat_ls[0:32]
                    del im_mos_pat_ls[0:32]
                    
                    input_im_pat=(((np.stack(im_pat_bat_ls,0))/127.5) - 1).astype(np.float32)
                    input_im_mos_pat=(((np.stack(im_mos_pat_bat_ls,0))/127.5) - 1).astype(np.float32)
                    del im_pat_bat_ls[0:32]
                    del im_mos_pat_bat_ls[0:32]
                    print('Batched Input')
                    yield input_im_mos_pat, input_im_pat
                
                  
            except Exception as e:
                print('file open error' + str(e))

                    
    return
                           
def train_generator1(train_dir, patch_size, batch_size):
    '''
    Python generator that loads imgs and batches
    '''

    while True:
        img_name = os.listdir(train_dir)
        
        # Loop through the image names according to batchsize 
        for i in range(0, len(img_name), batch_size):
            batch_img_names = img_name[i:i+batch_size]
            im_pat_ls=[]
            im_mos_pat_ls=[]
            # Process each image in batch
            for img  in enumerate(batch_img_names):
               
                sc = os.path.join(train_dir,img[1])
                
                try:
                    im=Image.open(sc)

                    im_raw_mos,im_orig = mosaic(im)

                    patch_mos, patch_orig = get_ptchs(im_raw_mos,im_orig,patch_size)

                    im_pat_ls.extend(patch_orig)
                    im_mos_pat_ls.extend(patch_mos)

                except Exception as e:
                    print('file open error' + str(e))
              
            print('Batched Input')
            input_im_pat=(((np.stack(im_pat_ls,0))/127.5) - 1).astype(np.float32)
            input_im_mos_pat=(((np.stack(im_mos_pat_ls,0))/127.5) - 1).astype(np.float32)
            
            #print(input_im_mos_pat[0])
            #print(input_im_pat[0])
            yield input_im_mos_pat, input_im_pat
          


