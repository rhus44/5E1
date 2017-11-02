import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import sklearn.feature_extraction
import scipy.ndimage
import time

def mosaic(im):
    h=int(im.size[0]/2)
    w=int(im.size[1]/2)

    print(filename+'...')
    print('w: '+str(w)+'  h: '+str(h))

    chnl = im.split()
    r=np.array(chnl[0])
    g=np.array(chnl[1])
    b=np.array(chnl[2])

    g1=np.zeros((w,h))
    g2=np.zeros((w,h))
    r1=np.zeros((w,h))
    b1=np.zeros((w,h))

    for x in range(0,w-1):
        for y in range(0,h-1):
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
    for i in range (0,int(h/32)*2):
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

opendir = r'C:\Training Images 3'

patch_size=64
lslen=int(len(os.listdir(opendir)))
print(lslen)
completed=1
im_pat_ls=[]
im_mos_pat_ls=[]
for filename in os.listdir(opendir):
    if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".tif"):
        sc=os.path.join(opendir,filename)
        try:
            im=Image.open(sc)
            #im.show()

            im_raw_mos,im_orig = mosaic(im)

            #print(filename+"_im-raw_"+str(im_raw.shape))
            # sv=os.path.join(savedir1,os.path.splitext(filename)[0])
            # np.save(sv,im_raw_mos)
            # sv=os.path.join(savedir2,os.path.splitext(filename)[0])
            # np.save(sv,im_orig)

            patch_mos, patch_orig = get_ptchs(im_raw_mos,im_orig,patch_size)

            im_pat_ls.extend(patch_orig)
            im_mos_pat_ls.extend(patch_mos)

            print(str(completed)+' of '+str(lslen))
            completed=completed+1
        except Exception as e:
            print('file open error')

    else:
        print("cannot open file:"+filename)

savefile = 'Training_Patches' + str(patch_size)
savepatch(opendir,savefile,im_mos_pat_ls,im_pat_ls)
im_pat_ls=[]
im_mos_pat_ls=[]




print("Completed: "+str(completed))
