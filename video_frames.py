import numpy as np
import os
from PIL import Image
import pickle
import imageio
import pylab
import keras
import cv2

def mosaic(im_ptchs):

    g1 = im_ptchs[::2,::2,1]
    #print(g1.shape)
    r  = im_ptchs[::2,1::2,2]
    #print(r.shape)
    g2 = im_ptchs[1::2,::2,1]
    #print(g2.shape)
    b  = im_ptchs[1::2,1::2,0]
    #print(b.shape)

    return np.stack((g1,b,g2,r),-1), np.stack((r,g1,b),-1)

def generate_video(img):
    for i in xrange(len(img)):
        plt.imshow(img[i], cmap=cm.Greys_r)
        plt.savefig(folder + "/file%02d.png" % i)

    os.chdir("your_folder")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

    return

def gen_vid(frames, filename):

    height, width, chnls = frames[0].shape

    video = cv2.VideoWriter(filename, -1, 30, (width,height))

    for i in range(0, len(frames)):
        video.write(frames[i])

    video.release()

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


opendir = r'C:\Users\buggyr\Mosaic_Experiments\data\external\Trees_4k.mp4'
savedir1 = r'C:\Users\buggyr\Mosaic_Experiments\data\processed\Kodak_test\mos_npy'
model = keras.models.load_model(r'C:\Users\buggyr\Mosaic_Experiments\models\2017-11-02 18-18\DeMos_mod.h5')

vid = imageio.get_reader(opendir,  'ffmpeg')
frames = []
mos_frames = []
mos_vis = []
for num in range(0,60):
    print('Frame: '+str(num))
    frame = np.array(vid.get_data(num))
    frames.append(frame)
    mos_4chn, mos_3chn = mosaic(frame)
    mos_frames.append(mos_4chn)
    mos_vis.append(mos_3chn)

input_mos_frames = (((np.stack(mos_frames,0))/127.5)-1).astype(np.float32)
np.save(r'C:\Users\buggyr\Mosaic_Experiments\data\mosf',input_mos_frames)

input_vis = np.stack(mos_vis,0)
#np.save(r'C:\Users\buggyr\Mosaic_Experiments\data\visf',pred_imgs)

pred_imgs = (model.predict(input_mos_frames, batch_size=1))
np.save(r'C:\Users\buggyr\Mosaic_Experiments\data\predf',pred_imgs)


gen_vid(input_vis,r'input.avi')

gen_vid((((pred_imgs+1)/2)*255).astype(np.uint8),r'predicted.avi')
