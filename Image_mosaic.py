import numpy as np
import os
from PIL import Image
import pickle



opendir = r'C:\Users\buggyr\Mosaic_Experiments\data\interim\Kodak'
savedir1 = r'C:\Users\buggyr\Mosaic_Experiments\data\processed\Kodak_test\mos_npy'
savedir2 = r'C:\Users\buggyr\Mosaic_Experiments\data\processed\Kodak_test\img_npy'
savedir3 = r'C:\Users\buggyr\Mosaic_Experiments\data\processed\Validation Images1'

completed=0
lsi=[]
lsm=[]
for filename in os.listdir(opendir):
    if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".tif") or filename.endswith(".png"):
        sc=os.path.join(opendir,filename)


        im=Image.open(sc)
        #im.show()
        h=int(im.size[0]/2) - 1
        w=int(im.size[1]/2) - 1

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

        for x in range(0,w):
            for y in range(0,h):
                x1=x*2
                y1=y*2
                #g1[x,y]=(g[x1,y1]+g[x1+1,y1+1])/2
                g1[x,y]=g[x1,y1]
                g2[x,y]=g[x1+1,y1+1]
                r1[x,y]=r[x1+1,y1]
                b1[x,y]=b[x1,y1+1]

        im_raw=np.stack((g1,b1,g2,r1),0)
        im_raw=np.moveaxis(im_raw, 0, -1)
        #print(filename+"_im-raw_"+str(im_raw.shape))
        sv=os.path.join(savedir1,os.path.splitext(filename)[0])
        np.save(sv,im_raw)

        im_r=np.stack((r,g,b))
        im_r=np.moveaxis(im_r, 0, -1)
        #print(filename+"_im-raw_"+str(im_raw.shape))

        sv=os.path.join(savedir2,os.path.splitext(filename)[0])
        np.save(sv,im_r)

        lsm.append(im_raw)
        lsi.append(im_r)

        completed=completed+1
    else:
        print("cannot open file:"+filename)

sv2=os.path.join(savedir2,'sv')

sv1=os.path.join(savedir1,'sv')

np.save(sv2,np.stack(lsm))
np.save(sv1,np.stack(lsi))

# with open(sv2,'wb') as fp2:
#     pickle.dump(lsm,fp2)
#
# with open(sv1,'wb') as fp1:
#     pickle.dump(lsi,fp1)

print("Completed: "+str(completed))
