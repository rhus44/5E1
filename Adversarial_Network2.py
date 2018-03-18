import datetime
import math
import random

import cv2
import keras
import skimage.measure
from keras.layers import (Conv2D, UpSampling2D, Input, concatenate, Lambda, Flatten,
                          Dense)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import *
from keras.optimizers import Nadam, Adam
from keras.utils import Sequence
from keras.utils.generic_utils import Progbar
from keras.utils.np_utils import to_categorical
from sklearn.feature_extraction import image
from keras import backend as k

k.clear_session()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class Save_predictions(keras.callbacks.Callback):
    def __init__(self, save_file, open_dir):
        self.savf = save_file
        self.predict_gen = predict_generator_rgb(open_dir)
        self.ep_dir = open_dir

    def on_epoch_end(self, epoch, logs={}):
        for i in (os.listdir(self.ep_dir)):
            p_img_gen, orig_gen, img_name = next(self.predict_gen)

            pred = self.model.predict(p_img_gen, batch_size=1)[0]

            pred_img = addBayer(orig_gen, pred)

            sv = os.path.join(self.savf, str(epoch) + '_' + img_name)
            cv2.imwrite(sv, denormalise1(pred_img))


class train_seq_tiled(Sequence):
    def __init__(self, train_dir, patch_size, batch_size, rotate=True, shuffle=True):

        self.train_dir = train_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_names = os.listdir(train_dir)
        self.btch_start = np.zeros(len(os.listdir(train_dir))).astype(np.uint8)
        self.rotate = rotate

    def __len__(self):
        return math.ceil(len(self.img_names))

    def __getitem__(self, idx):

        train_dir = self.train_dir
        batch_size = self.batch_size
        patch_size = self.patch_size

        img = self.img_names[idx]

        n_dim = 128
        m_dim = 64
        tile_n = 8
        orig = np.ones((128, 128, 6))
        input_mos4 = np.zeros((batch_size, patch_size, patch_size, 4))
        input_mos3 = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
        if (self.rotate == True):
            input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 6))
        else:
            input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
        num_btchs = int(64 / batch_size)

        sc = os.path.join(train_dir, img)

        ptch = cv2.imread(sc)

        # ptch_mos4 = mosaic1(ptch, 4)
        # ptch_mos3 = mosaic1(ptch, 3)

        btch_begin = self.btch_start[idx]
        if (btch_begin == 0):
            self.btch_start[idx] = 4
        else:
            self.btch_start[idx] = 0
        x = 0
        for i in range(btch_begin, tile_n):
            for j in range(tile_n):
                i_dim = n_dim * i
                j_dim = n_dim * j

                im_dim = m_dim * i
                jm_dim = m_dim * j

                if (self.rotate == True):

                    ptch_rotated = rotate_random(ptch[i_dim:i_dim + n_dim,
                                                 j_dim:j_dim + n_dim, :], random.randint(0, 90))

                    ptch_mos4 = mosaic1(ptch_rotated, 4)
                    ptch_mos3 = mosaic1(ptch_rotated, 3)

                    orig[:, :, 0:3] = normalise1(ptch_rotated)
                    orig[:, :, 3:6] = (orig[:, :, 0:3] > 0).astype(np.float32)

                    input_orig[x, :, :, :] = orig
                    input_mos4[x, :, :, :] = ptch_mos4
                    input_mos3[x, :, :, :] = ptch_mos3
                else:

                    input_orig[x, :, :, :] = ptch[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
                    input_mos4[x, :, :, :] = ptch_mos4[im_dim:im_dim + m_dim, jm_dim:jm_dim + m_dim, :]
                    input_mos3[x, :, :, :] = ptch_mos3[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
                x += 1
                if (x % batch_size == 0):
                    x = 0

                    return [normalise1(input_mos4), normalise1(input_mos3)], input_orig


class train_seq(Sequence):
    def __init__(self, train_dir, patch_size, batch_size, shuffle=True):

        self.train_dir = train_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_names = os.listdir(train_dir)
        self.btch_start = np.zeros(len(os.listdir(train_dir))).astype(np.uint8)

    def __len__(self):
        return math.ceil(len(self.img_names))

    def __getitem__(self, idx):

        img_name = os.listdir(train_dir)

        input_mos4 = np.zeros((batch_size, patch_size, patch_size, 4))
        input_mos3 = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
        input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))

        img = self.img_names[idx]

        sc = os.path.join(train_dir, img)
        # print(sc)
        try:
            im = cv2.imread(sc)

            ptch = extractPatches(im, patch_size)
            # print(img_ptch.shape)

            ptch_mos4 = mosaic(ptch, 4)
            ptch_mos3 = mosaic(ptch, 3)

            num_ptchs = ptch_mos4.shape[0]
            # print(num_ptchs)
            btchs = int(num_ptchs / batch_size)
            # print(str(btchs))

            b = i * batch_size

            input_mos4 = ptch_mos4[b:b + batch_size]

            input_mos3 = ptch_mos3[0][b:b + batch_size]

            input_orig = ptch[b:b + batch_size]

            return [normalise1(input_mos4), normalise1(input_mos3)], normalise1(input_orig)


        except Exception as e:
            print('file open error: ' + str(e))


def get_unet_input(array):
    y = []
    h, w, _ = array.shape
    for x in range(0, max(h, w)):
        x1 = x
        if (x % 32 == 0):
            y.append(x1)

    h1 = max(int(t) for t in y if h >= t)
    w1 = max(int(t) for t in y if w >= t)
    return array[:h1, :w1:, :]


def extractPatches(im, patch_size, max_patches=96):
    return image.extract_patches_2d(im, (patch_size * 2, patch_size * 2), max_patches)


def rotate_random(img, x):
    rows, cols, _ = img.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), x, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


# batch_mosaic
def mosaic(im_ptchs, num_channels=4):
    if (num_channels == 4):
        g1 = im_ptchs[:, ::2, ::2, 1]
        # print(g1.shape)
        r = im_ptchs[:, ::2, 1::2, 2]
        # print(r.shape)
        g2 = im_ptchs[:, 1::2, 1::2, 1]
        # print(g2.shape)
        b = im_ptchs[:, 1::2, ::2, 0]
        # print(b.shape)
        out = np.stack((g1, b, g2, r), -1)
        # print("Mos4 Shape:" + str(out.shape))
        return out

    elif (num_channels == 3):
        n, h, w, c = im_ptchs.shape

        b = np.zeros((n, h, w))
        g = np.zeros((n, h, w))
        r = np.zeros((n, h, w))

        b[:, 1::2, ::2] = im_ptchs[:, 1::2, ::2, 0]
        # print(b.shape)
        g[:, ::2, ::2] = im_ptchs[:, ::2, ::2, 1]
        g[:, 1::2, 1::2] = im_ptchs[:, 1::2, 1::2, 1]
        # print(g.shape)
        r[:, ::2, 1::2] = im_ptchs[:, ::2, 1::2, 2]
        # print(r.shape)
        out = np.stack((b, g, r), -1), im_ptchs
        # print(out[1].shape)
        return out

    elif (num_channels == 6):
        n, h, w, c = im_ptchs.shape

        b = np.zeros((n, h, w))
        g = np.zeros((n, h, w))
        r = np.zeros((n, h, w))
        bm = np.zeros((n, h, w))
        gm = np.zeros((n, h, w))
        rm = np.zeros((n, h, w))

        b[:, 1::2, ::2] = im_ptchs[:, 1::2, ::2, 0]
        # print(b.shape)
        g[:, ::2, ::2] = im_ptchs[:, ::2, ::2, 1]
        g[:, 1::2, 1::2] = im_ptchs[:, 1::2, 1::2, 1]
        # print(g.shape)
        r[:, ::2, 1::2] = im_ptchs[:, ::2, 1::2, 2]
        # print(r.shape)
        bm[:, 1::2, ::2] = 1
        # print(b.shape)
        gm[:, ::2, ::2] = 1
        gm[:, 1::2, 1::2] = 1
        # print(g.shape)
        rm[:, ::2, 1::2] = 1

        return np.stack((b, g, r, bm, gm, rm), -1), im_ptchs

    elif (num_channels == 1):
        n, h, w, c = im_ptchs.shape

        bgr = np.zeros((n, h, w))

        bgr[:, 1::2, ::2] = im_ptchs[:, 1::2, ::2, 0]
        # print(b.shape)
        bgr[:, ::2, ::2] = im_ptchs[:, ::2, ::2, 1]
        bgr[:, 1::2, 1::2] = im_ptchs[:, 1::2, 1::2, 1]
        # print(g.shape)
        bgr[:, ::2, 1::2] = im_ptchs[:, ::2, 1::2, 2]

        return bgr, im_ptchs

    elif (num_channels == 0):
        n, h, w, c = im_ptchs.shape

        g = np.zeros((n, h, w))

        g[:, ::2, ::2] = im_ptchs[:, ::2, ::2, 1]
        g[:, 1::2, 1::2] = im_ptchs[:, 1::2, 1::2, 1]

        return g, im_ptchs[:, :, :, 1]

    return 0


# mosaic_single_image
def mosaic1(im_ptchs, num_channels=4):
    # print("Input Dims: ",str(im_ptchs.shape))
    if (num_channels == 4):

        g1 = im_ptchs[::2, ::2, 1]
        # print(g1.shape)
        r = im_ptchs[::2, 1::2, 2]
        # print(r.shape)
        g2 = im_ptchs[1::2, 1::2, 1]
        # print(g2.shape)
        b = im_ptchs[1::2, ::2, 0]
        # print(b.shape)
        out = np.stack((g1, b, g2, r), -1)
        # print("Mos4 Shape: " + str(out.shape))
        return out

    elif (num_channels == 3):
        h, w, _ = im_ptchs.shape

        b = np.zeros((h, w))
        g = np.zeros((h, w))
        r = np.zeros((h, w))

        b[1::2, ::2] = im_ptchs[1::2, ::2, 0]
        # print(b.shape)
        g[::2, ::2] = im_ptchs[::2, ::2, 1]
        g[1::2, 1::2] = im_ptchs[1::2, 1::2, 1]
        # print(g.shape)
        r[::2, 1::2] = im_ptchs[::2, 1::2, 2]
        out = np.stack((b, g, r), -1)
        return out

    elif (num_channels == 6):
        h, w, c = im_ptchs.shape

        b = np.zeros((h, w))
        g = np.zeros((h, w))
        r = np.zeros((h, w))
        bm = np.zeros((h, w))
        gm = np.zeros((h, w))
        rm = np.zeros((h, w))

        b[:, 1::2, ::2] = im_ptchs[1::2, ::2, 0]
        # print(b.shape)
        g[:, ::2, ::2] = im_ptchs[::2, ::2, 1]
        g[:, 1::2, 1::2] = im_ptchs[1::2, 1::2, 1]
        # print(g.shape)
        r[:, ::2, 1::2] = im_ptchs[::2, 1::2, 2]
        # print(r.shape)
        bm[:, 1::2, ::2] = 1
        # print(b.shape)
        gm[:, ::2, ::2] = 1
        gm[:, 1::2, 1::2] = 1
        # print(g.shape)
        rm[:, ::2, 1::2] = 1

        return np.stack((b, g, r, bm, gm, rm), -1)

    elif (num_channels == 1):
        h, w, c = im_ptchs.shape

        bgr = np.zeros((h, w))

        bgr[1::2, ::2] = im_ptchs[1::2, ::2, 0]
        # print(b.shape)
        bgr[::2, ::2] = im_ptchs[::2, ::2, 1]
        bgr[1::2, 1::2] = im_ptchs[1::2, 1::2, 1]
        # print(g.shape)
        bgr[::2, 1::2] = im_ptchs[::2, 1::2, 2]

        return bgr, im_ptchs

    elif (num_channels == 0):
        h, w, c = im_ptchs.shape

        g = np.zeros((h, w))
        g[::2, ::2] = im_ptchs[::2, ::2, 1]
        g[1::2, 1::2] = im_ptchs[1::2, 1::2, 1]

        return g

    return 0


def normalise(array):
    return ((array / 127.5) - 1).astype(np.float32)


def denormalise(array):
    return ((array + 1) * 127.5).astype(np.uint8)


def normalise1(array):
    return (array / 255).astype(np.float32)


def denormalise1(array):
    return (np.floor((array) * 255)).astype(np.uint8)


def addBayer(orig, predicted):
    predicted[::2, ::2, 1] = orig[::2, ::2, 1]

    predicted[::2, 1::2, 2] = orig[::2, 1::2, 2]

    predicted[1::2, 1::2, 1] = orig[1::2, 1::2, 1]

    predicted[1::2, ::2, 0] = orig[1::2, ::2, 0]

    print("Bayer Difference:", str(np.average(abs(predicted - orig))))

    return predicted


def clipper(x):
    x = K.clip(x, -1, 1)
    return x


def loss_SSIM(y_true, y_pred):
    y_true = tf.transpose(y_true, [0, 3, 1, 2])
    y_pred = tf.transpose(y_pred, [0, 3, 1, 2])

    u_true = K.mean(y_true, axis=3)
    u_pred = K.mean(y_pred, axis=3)
    var_true = K.var(y_true, axis=3)
    var_pred = K.var(y_pred, axis=3)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    return K.mean(((1.0 - ssim) / 2))


def loss_yuv(y_true, y_pred):
    bgr2yuv = K.constant([[0.11400, 0.436, -0.1],
                          [0.58700, -0.289, -0.515],
                          [0.29900, -0.147, 0.615]], name='bgr2yuv')

    y_true_yuv = K.dot(y_true[:, :, :, 0:3], bgr2yuv)
    y_pred_yuv = K.dot(y_pred, bgr2yuv)

    if (K.int_shape(y_true)[2] is not None):
        if (K.int_shape(y_true)[2] > 3):
            return K.mean(K.square(y_pred_yuv - y_true_yuv) * y_true[:, :, :, 3:6], axis=-1)

    else:
        return K.mean(K.square(y_pred_yuv - y_true_yuv), axis=-1)


def loss_custom(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def train_generator_rgb(train_dir, patch_size, batch_size, scale, num_patches):
    '''
    Python generator that loads imgs and batches
    '''

    while True:
        img_name = os.listdir(train_dir)

        input_mos4 = np.zeros((batch_size, patch_size, patch_size, 4))
        input_mos3 = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
        input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))

        for img in enumerate(img_name):

            sc = os.path.join(train_dir, img[1])
            # print(sc)
            try:
                im = cv2.imread(sc)
                x, y, _ = np.shape(im)
                fx = int(x / scale)
                fy = int(y / scale)
                im = cv2.resize(im, (fx, fy), interpolation=cv2.INTER_LANCZOS4)

                ptch = extractPatches(im, patch_size, num_patches)
                # print(img_ptch.shape)

                ptch_mos4 = mosaic(ptch, 4)
                ptch_mos3 = mosaic(ptch, 3)

                num_ptchs = ptch_mos4.shape[0]
                # print(num_ptchs)
                btchs = int(num_ptchs / batch_size)
                # print(str(btchs))
                for i in range(btchs):
                    b = i * batch_size

                    input_mos4 = ptch_mos4[b:b + batch_size]

                    input_mos3 = ptch_mos3[0][b:b + batch_size]

                    input_orig = ptch[b:b + batch_size]

                    yield [normalise1(input_mos4), normalise1(input_mos3)], normalise1(input_orig)


            except Exception as e:
                print('file open error: ' + str(e))
    return


def train_generator_rgb_patches(train_dir, patch_size, batch_size, input_channels=4):
    '''
    Python generator that loads imgs and batches
    '''

    if (input_channels == 4):
        input_mos = np.zeros((batch_size, patch_size, patch_size, 4))
        input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
    else:
        input_mos = np.zeros((batch_size, patch_size, patch_size, input_channels))
        input_orig = np.zeros((batch_size, patch_size, patch_size, 3))
    read = 0

    while True:
        img_name = os.listdir(train_dir)

        for img in enumerate(img_name):

            # print(img[0])
            sc = os.path.join(train_dir, img[1])

            try:
                ptch = cv2.imread(sc)

                ptch_mos = mosaic1(ptch, input_channels)

                input_mos[read] = ptch_mos
                input_orig[read] = ptch

                read += 1

                if (read == batch_size):
                    yield normalise1(input_mos), normalise1(input_orig)
                    read = 0

            except Exception as e:
                print('file open error: ' + str(e) + str(img[1]))

    return


def train_generator_rgb_tiled(train_dir, patch_size, batch_size, input_channels=4):
    '''
    Python generator that loads imgs and batches
    '''

    while True:
        img_name = os.listdir(train_dir)
        n_dim = 128
        m_dim = 64
        tile_n = 8
        input_mos4 = np.zeros((batch_size, patch_size, patch_size, 4))
        input_mos3 = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
        input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
        num_btchs = int(64 / batch_size)

        for img in enumerate(img_name):

            sc = os.path.join(train_dir, img[1])
            # print(sc)
            try:
                ptch = cv2.imread(sc)

                ptch_mos4 = mosaic1(ptch, 4)
                ptch_mos3 = mosaic1(ptch, 3)

                x = 0
                for i in range(tile_n):
                    for j in range(tile_n):
                        i_dim = n_dim * i
                        j_dim = n_dim * j

                        im_dim = m_dim * i
                        jm_dim = m_dim * j

                        input_orig[x, :, :, :] = ptch[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
                        input_mos4[x, :, :, :] = ptch_mos4[im_dim:im_dim + m_dim, jm_dim:jm_dim + m_dim, :]
                        input_mos3[x, :, :, :] = ptch_mos3[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
                        x += 1
                        if (x % batch_size == 0):
                            x = 0
                            yield [normalise1(input_mos4), normalise1(input_mos3)], normalise1(input_orig)

            except Exception as e:
                print('file open error: ' + str(e))
    return


def train_generator_ad(train_dir, patch_size, batch_size, gen_model):
    while True:
        print("CUDA Devices ", os.environ["CUDA_VISIBLE_DEVICES"])
        img_name = os.listdir(train_dir)
        n_dim = 128
        m_dim = 64
        tile_n = 8
        input_mos4 = np.zeros((batch_size, patch_size, patch_size, 4))
        input_mos3 = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
        input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
        num_btchs = int(64 / batch_size)

        for img in enumerate(img_name):

            sc = os.path.join(train_dir, img[1])
            # print(sc)
            try:
                ptch = cv2.imread(sc)

                ptch_mos4 = mosaic1(ptch, 4)
                ptch_mos3 = mosaic1(ptch, 3)

                x = 0
                for i in range(tile_n):
                    for j in range(tile_n):
                        i_dim = n_dim * i
                        j_dim = n_dim * j

                        im_dim = m_dim * i
                        jm_dim = m_dim * j

                        input_orig[x, :, :, :] = ptch[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
                        input_mos4[x, :, :, :] = ptch_mos4[im_dim:im_dim + m_dim, jm_dim:jm_dim + m_dim, :]
                        input_mos3[x, :, :, :] = ptch_mos3[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
                        x += 1
                        if (x % batch_size == 0):
                            gen_btch = gen_model.predict_on_batch([normalise1(input_mos4), normalise1(input_mos3)])
                            gen_label = [0] * batch_size

                            orig_label = [1] * batch_size
                            x = 0

                            yield np.concatenate((gen_btch, input_orig), axis=0), gen_label + orig_label

            except Exception as e:
                print('file open error: ' + str(e))
    return


def val_generator_rgb(train_dir):
    while True:
        img_name = os.listdir(train_dir)

        for img in enumerate(img_name):

            sc = os.path.join(train_dir, img[1])
            # print(sc)

            try:
                im = cv2.imread(sc)

                im = get_unet_input(im)

                mos4 = np.array([mosaic1(im, 4)])
                mos3 = np.array([mosaic1(im, 3)])
                # print(normalise1(mos).shape)

                yield [normalise1(mos4), normalise1(mos3)], normalise1(np.array([im]))

            except Exception as e:
                print('file open error: ' + str(e))

    return


def predict_generator_rgb(train_dir):
    '''
    Python generator that loads imgs and batches
    '''

    while True:
        img_name = os.listdir(train_dir)

        for img in enumerate(img_name):

            sc = os.path.join(train_dir, img[1])
            # print(sc)

            try:
                im = cv2.imread(sc)

                im = get_unet_input(im)

                mos4 = np.array([mosaic1(im, 4)])
                mos3 = np.array([mosaic1(im, 3)])

                yield [normalise1(mos4), normalise1(mos3)], normalise1(im), img[1]
                # print('Batched Input')

            except Exception as e:
                print('file open error: ' + str(e))

    return


def predict_generator(model, k_gen, steps, data, bp, save_file):
    psnr = []
    ssim = []

    for i in range(steps):
        p_img_gen, orig_gen, img_name = next(k_gen)

        pred_img = (model.predict(p_img_gen, batch_size=1))[0]
        pred_img[(pred_img > 1)] = 1
        # pred_img = denormalise1(pred_img)
        # print("Output max:", str(pred_img.min()))
        # print("Output min", str(pred_img.max()))
        print("Predicted Model Shape: " + str(pred_img.shape))
        # cv2.imshow('image',pred_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        pred_img = addBayer(orig_gen, pred_img)

        pred_psnr = skimage.measure.compare_psnr(pred_img[bp:-bp, bp:-bp, :], orig_gen[bp:-bp, bp:-bp, :], 1)
        print(img_name + " PSNR: " + str(pred_psnr))

        cv2.imshow("Original " + img_name, orig_gen)
        cv2.imshow("Difference " + img_name, (np.floor((abs(orig_gen - pred_img)) * 255)).astype(np.uint8) + 100)
        cv2.imshow("Predicted " + img_name, pred_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(save_file, ('predicted_' + img_name)), denormalise1(pred_img))

        psnr.append(pred_psnr)
        ssim.append(
            skimage.measure.compare_ssim(pred_img[bp:-bp, bp:-bp, :], orig_gen[bp:-bp, bp:-bp, :], multichannel=True))

    psnr_avg = sum(psnr) / steps
    ssim_avg = sum(ssim) / steps
    print('Kodak PSNR Average: ' + str(psnr_avg))
    print('Kodak SSIM Average: ' + str(ssim_avg))

    return psnr, ssim, psnr_avg, ssim_avg


def predict_chal_patchs(model, psnr_threshold, k_gen, steps, save_file):
    target = []
    num_btch = 0
    chlng = 0
    test_chlng = 0

    for i in range(steps):

        print(str(i) + " of " + str(steps))

        input_ptchs, target_patchs = next(k_gen)

        pred_btch = model.predict_on_batch(input_ptchs)

        # print(target_patchs.dtype)
        # print(pred_btch.dtype)

        for i in range(pred_btch.shape[0]):
            pred_psnr = skimage.measure.compare_psnr(pred_btch[i], target_patchs[i])
            test_chlng += 1

            if (pred_psnr < psnr_threshold):
                print(pred_psnr)
                target.append(target_patchs[i])
                chlng += 1

            if (len(target) > 63):
                chnlg_target = np.stack(target, 0)
                tile_image(chnlg_target, save_file, str(num_btch))

                target = []

                num_btch += 1

    return [chlng, test_chlng]


def tile_image(target, save_dir, s_name):
    n_dim = 128
    tile_n = 8

    im_tile = np.zeros((n_dim * tile_n, n_dim * tile_n, 3))

    x = 0

    print("Tiling ")
    for i in range(tile_n):
        for j in range(tile_n):
            im = target[x]
            # cv2.imshow(im_name[x], im)
            i_dim = n_dim * i
            j_dim = n_dim * j
            im_tile[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :] = im
            x += 1
            cv2.imwrite(os.path.join(save_dir, os.path.split(save_dir)[1] + s_name + '.png'), denormalise1(im_tile))

    # cv2.imshow("Tiled ", denormalise1(im_tile))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


###############################################################################
###############################################################################
###############################################################################

def smooth_gan_labels(y):
    assert len(y.shape) == 2, "Needs to be a binary class"
    y = np.asarray(y, dtype='int')
    Y = np.zeros(y.shape, dtype='float32')

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i, j] == 0:
                Y[i, j] = np.random.uniform(0.0, 0.3)
            else:
                Y[i, j] = np.random.uniform(0.7, 1.2)

    return Y


def k_load_model(path):
    return keras.models.load_model(path)


class GenModel:
    def __init__(self):
        self.gen_model = None
        return

    def load_model(self, path):
        self.gen_model = keras.models.load_model(path)

        return self.gen_model

    def pre_train(self):
        return self.gen_model


class AdModel:
    def __init__(self, disc_loss_weight=0.5):
        self.pred_dir = r'C:\Users\buggyr\Mosaic_Experiments\data\interim\Callback_images'
        self.save_pred_dir = None

        self.batch_size = None
        self.disc_loss_weight = disc_loss_weight
        self.save_epoch_models = None
        self.save_file = None
        self.save_test = None

        self.gen_loss_func = 'mse'
        self.disc_loss_func = 'categorical_crossentropy'
        self.disc_optimizer_func = Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

        self.defined_gen_model = None
        self.un_def_gen_model = None
        self.disc_model = None
        self.ad_model = None

        self.test_data = None
        return

    def mk_file(self):
        save_results = r'C:\Users\buggyr\Mosaic_Experiments\models'
        keyname = "_Ad_Model"
        now = datetime.datetime.now()
        self.save_file = os.path.join(save_results, now.strftime("%Y-%m-%d %H-%M") + keyname)
        os.mkdir(self.save_file)
        self.save_pred_dir = os.path.join(self.save_file, 'Epoch_Predictions')
        os.mkdir(self.save_pred_dir)
        self.save_epoch_models = os.path.join(self.save_file, 'Epoch_Models')
        os.mkdir(self.save_epoch_models)
        self.save_test = os.path.join(self.save_file, 'Test_Results')
        os.mkdir(self.save_test)

        return self.save_file

    def load_model(self, path):

        self.ad_model = keras.models.load_model(path)
        print('Loaded AD_model')

        ad_weights = self.ad_model.get_weights()
        print('Loaded Weights')

        self.disc_model.set_weights(ad_weights[32:43])
        print('Set Weights')

        self.disc_model.save('disc.h5')
        print('Saved Model')

        self.un_def_gen_model.set_weights(ad_weights[0:31])
        print('set weights')

        self.un_def_gen_model.save('un_def_gen.h5')
        print('Saved Model')

        return self.ad_model

    def get_defined_gen_model(self, shape):
        chnl4_input = Input(shape=(64, 64, 4))
        chnl3_input = Input(shape=(128, 128, 3))

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(chnl4_input)
        conv2 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv1)

        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv4 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv3)

        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

        up1 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)

        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up2 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv1], axis=-1)
        conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)

        conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)
        conv12 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv11)

        up3 = concatenate([UpSampling2D(size=(2, 2))(conv12), chnl3_input], axis=-1)
        conv13 = Conv2D(67, (3, 3), activation='relu', padding='same')(up3)

        conv14 = Conv2D(67, (3, 3), activation='relu', padding='same')(conv13)
        conv15 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv14)
        conv16 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv15)

        out = Lambda(clipper, name='clipper')(conv16)

        self.defined_gen_model = Model(inputs=[chnl4_input, chnl3_input], outputs=[out], name='Defined_Generator_Model')

        # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
        # model1.compile(optimizer=optimizer_func,loss=loss_func)

        return self.defined_gen_model

    def load_undefined_gen_model(self, shape=(128, 128, 3)):

        chnl4_input = Input(shape=(None, None, 4))
        chnl3_input = Input(shape=(None, None, 3))

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(chnl4_input)
        conv2 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv1)

        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv4 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv3)

        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

        up1 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)

        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        conv9 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up2 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv1], axis=-1)
        conv10 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)

        conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv10)
        conv12 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv11)

        up3 = concatenate([UpSampling2D(size=(2, 2))(conv12), chnl3_input], axis=-1)
        conv13 = Conv2D(67, (3, 3), activation='relu', padding='same')(up3)

        conv14 = Conv2D(67, (3, 3), activation='relu', padding='same')(conv13)
        conv15 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv14)
        conv16 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv15)

        out = Lambda(clipper, name='clipper')(conv16)

        self.un_def_gen_model = Model(inputs=[chnl4_input, chnl3_input], outputs=[out], name='Undefined_Generator_Model')

        # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
        optimizer_func = Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        loss_func = 'logcosh'
        self.un_def_gen_model.compile(optimizer=optimizer_func, loss=loss_func)
        # model1.compile(optimizer=optimizer_func,loss=loss_func)

        return self.un_def_gen_model

    def load_gen_model(self,
                       path=r'C:\Users\buggyr\Mosaic_Experiments\src\un_def_gen.h5'):

        self.un_def_gen_model = k_load_model(path)
        weights = self.un_def_gen_model.get_weights()

        self.get_defined_gen_model(shape=(128, 128, 3))
        self.defined_gen_model.set_weights(weights)

        print('Gen_loaded')
        return self.defined_gen_model

    def load_disc_model(self,
                        path=r'C:\Users\buggyr\Mosaic_Experiments\src\disc.h5'):

        self.disc_model = k_load_model(path)
        self.disc_model.compile(self.disc_optimizer_func, self.disc_loss_func)
        self.disc_model.name = 'Discriminator_Model'

        print('Disc_Loaded')
        return self.disc_model

    def build_disc_model(self):

        inputs = Input(shape=(128, 128, 3))

        conv1 = Conv2D(16, (3, 3), strides=2, padding='same')(inputs)

        conv2 = Conv2D(32, (3, 3), strides=2, padding='same')(conv1)

        conv3 = Conv2D(64, (3, 3), strides=2, padding='same')(conv2)

        conv4 = Conv2D(128, (3, 3), padding='same')(conv3)


        flatten = Flatten(name='flatten')(conv4)

        dense1 = Dense(512, activation='relu')(flatten)

        dense3 = Dense(2, activation='softmax')(dense1)

        self.disc_model = Model(inputs=[inputs], outputs=[dense3])

        optimizer_func = Adam(lr=1e-4)
        loss_func = 'categorical_crossentropy'
        self.disc_model.compile(optimizer=optimizer_func, loss=loss_func)

        self.disc_model.summary()

        return self.disc_model

    def assm_ad_model(self):

        self.load_gen_model()

        self.ad_model = Model(inputs=self.defined_gen_model.inputs,
                              outputs=[self.defined_gen_model.output, self.disc_model(self.defined_gen_model.output)],
                              name='Adversarial_Container')

        print('Ad_Model Assembled')
        return self.ad_model

    def set_trainable(self, value):
        for layer in self.disc_model.layers:
            layer.trainable = value

    def predict_callback(self, epoch):

        for layr in range(len(self.un_def_gen_model.layers)):

            self.un_def_gen_model.layers[layr].set_weights(self.ad_model.layers[layr].get_weights())
            gen_weights = self.un_def_gen_model.get_weights()
        self.un_def_gen_model.compile(optimizer=self.disc_optimizer_func, loss=self.gen_loss_func)
        gen_weights = self.un_def_gen_model.get_weights()
        predict_gen = predict_generator_rgb(self.pred_dir)
        psnr = {}
        for _ in (os.listdir(self.pred_dir)):
            p_img_gen, orig_gen, img_name = next(predict_gen)

            pred = self.un_def_gen_model.predict(p_img_gen, batch_size=1)[0]
            pred_img = addBayer(orig_gen, pred)

            pred_psnr = skimage.measure.compare_psnr(pred_img, orig_gen, 1)
            psnr[img_name] = pred_psnr
            print(img_name + " PSNR: " + str(pred_psnr))

            sv = os.path.join(self.save_pred_dir, str(epoch) + '_' + img_name)
            cv2.imwrite(sv, denormalise1(pred_img))
        print('Written Predicted Images')

    def train(self, load_training, load_validation=r'C:\Users\buggyr\Mosaic_Experiments\data\interim\validation_tiled',
              batch_size=32, epochs=50, initial_epoch=0):

        # Train D -> Train AD -> Train D
        self.batch_size = batch_size
        optimizer_func = Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        self.gen_loss_func = 'mse'
        self.disc_loss_func = 'categorical_crossentropy'

        self.disc_model.compile(optimizer=optimizer_func, loss=self.disc_loss_func)

        self.set_trainable(False)

        self.ad_model.compile(optimizer=optimizer_func, loss=[self.gen_loss_func, self.disc_loss_func],
                              loss_weights=[1 - self.disc_loss_weight, self.disc_loss_weight])
        self.ad_model.stop_training = False

        print('Models Compiled!')

        tbCallBack = keras.callbacks.TensorBoard(log_dir=os.path.join(self.save_file, 'TNSR_BRD'), histogram_freq=0,
                                                 write_graph=True, write_images=True, write_grads=True)
        tbCallBack.set_model(self.ad_model)
        csv_logger = keras.callbacks.CSVLogger(os.path.join(self.save_file, 'training.log'), separator=',',
                                               append=False)
        csv_logger.set_model(self.ad_model)
        csv_logger.on_train_begin()

        fln_training = len(os.listdir(load_training))
        fln_validation = len(os.listdir(load_validation))

        loss_history = {'discriminator_loss': [],
                        'generator_loss': [], }

        validation_history = {'discriminator_loss': [],
                              'generator_loss': [], }

        logs = {'Gen_loss_train': [],
                'Disc_loss_train': [],
                'Gen_loss_val': [],
                'Disc_loss_val': [], }

        print("Begin AD_Model Train")

        for ep in range(initial_epoch, epochs + 1):
            print('Epoch {}/{}'.format(ep, epochs))

            num_batches = int(2 * fln_training)
            progress_bar = Progbar(target=num_batches)

            epoch_gen_loss = []
            epoch_disc_loss = []

            train_generator = train_generator_rgb_tiled(load_training, 64, self.batch_size)
            validation_generator = train_generator_rgb_tiled(load_validation, 64, self.batch_size)
            for i in range(2 * fln_training):
                try:
                    ####### train discriminator ########
                    inputs, input_orig = next(train_generator)

                    gen_batch = self.defined_gen_model.predict_on_batch(inputs)

                    target_label = [0] * self.batch_size + [1] * self.batch_size

                    y_gan = np.asarray(target_label, dtype=np.int).reshape(-1, 1)
                    y_gan = to_categorical(y_gan, num_classes=2)
                    y_gan = smooth_gan_labels(y_gan)

                    input_gen = np.concatenate((gen_batch, input_orig), axis=0)

                    d_weights = self.disc_model.get_weights()
                    d_ad_weights = self.ad_model.get_weights()
                    g_weights = self.defined_gen_model.get_weights()


                    epoch_disc_loss.append(self.disc_model.train_on_batch(input_gen, y_gan))
                    ############################################
                    d_weights = self.disc_model.get_weights()
                    d_ad_weights = self.ad_model.get_weights()
                    g_weights = self.defined_gen_model.get_weights()

                    ########### train generator ################

                    y_gan = [1] * self.batch_size
                    y_gan = np.asarray(y_gan, dtype=np.int).reshape(-1, 1)
                    y_gan = to_categorical(y_gan, num_classes=2)
                    y_gan = smooth_gan_labels(y_gan)

                    epoch_gen_loss.append(self.ad_model.train_on_batch(inputs, [input_orig, y_gan]))

                    d_weights = self.disc_model.get_weights()
                    d_ad_weights = self.ad_model.get_weights()
                    g_weights = self.defined_gen_model.get_weights()

                    progress_bar.update(i + 1)

                except KeyboardInterrupt:
                    print("Keyboard interrupt detected. Stopping early and Saving Out.")
                    self.ad_model.save(os.path.join(self.save_epoch_models, 'Adversarial_2.Epoch.' + str(ep) + '.h5'))
                    self.disc_model.save(
                        os.path.join(self.save_epoch_models, 'Discriminator_2.Epoch.' + str(ep) + '.h5'))
                    self.un_def_gen_model.save(
                        os.path.join(self.save_epoch_models, 'Generator_undefined_2.Epoch.' + str(ep) + '.h5'))
                    break

            # Epoch Loss History #####################
            hist_val_ad = 0
            hist_val_disc = 0
            for i in range(2 * fln_validation):
                inputs, input_orig = next(validation_generator)

                y_gan = [1] * self.batch_size
                y_gan = np.asarray(y_gan, dtype=np.int).reshape(-1, 1)
                y_gan = to_categorical(y_gan, num_classes=2)

                # test AD_Model {Generator}
                hist_val_ad = self.ad_model.test_on_batch(inputs, [input_orig, y_gan])

                # test Disc_model
                gen_batch = self.defined_gen_model.predict_on_batch(inputs)

                target_label = [0] * self.batch_size + [1] * self.batch_size

                y_gan = np.asarray(target_label, dtype=np.int).reshape(-1, 1)
                y_gan = to_categorical(y_gan, num_classes=2)
                y_gan = smooth_gan_labels(y_gan)

                input_gen = np.concatenate((gen_batch, input_orig), axis=0)

                hist_val_disc = self.disc_model.test_on_batch(input_gen, y_gan)

            validation_history['generator_loss'].append(hist_val_ad)
            validation_history['discriminator_loss'].append(hist_val_disc)

            discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0).item()
            generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0).tolist()

            loss_history['discriminator_loss'].append(discriminator_train_loss)
            loss_history['generator_loss'].append(generator_train_loss)

            logs['Gen_loss_train'] = generator_train_loss
            logs['Disc_loss_train'] = discriminator_train_loss
            logs['Gen_loss_val'] = hist_val_ad
            logs['Disc_loss_val'] = hist_val_disc

            l = {'Discriminator_Loss': np.mean(np.array(epoch_disc_loss), axis=0),
                 'Generator_Loss': np.mean(np.array(epoch_gen_loss), axis=0)[0]}

            # Write Epoch Results ##############

            tbCallBack.on_epoch_end(epoch=ep, logs=l)
            csv_logger.on_epoch_end(epoch=ep, logs=logs)

            # Write Epoch Models
            self.ad_model.save(os.path.join(self.save_epoch_models, 'Adversarial_1.Epoch.' + str(ep) + '.h5'))
            self.disc_model.save(os.path.join(self.save_epoch_models, 'Discriminator_1.Epoch.' + str(ep) + '.h5'))
            self.un_def_gen_model.save(os.path.join(self.save_epoch_models, 'Generator_undefined_1.Epoch.' + str(ep) + '.h5'))

            # Write Epoch Predictions
            self.predict_callback(epoch=ep)

        # Write Epoch Histories


        with open(os.path.join(self.save_file, 'History.txt'), 'w') as fh:
            json.dump(loss_history, fh, indent=4)

        # Write Final Model

        return self.ad_model

    def model_test(self):
        data = {}
        # Test Kodak
        kodak_dir = r'C:\Users\buggyr\Mosaic_Experiments\data\interim\Kodak'
        ls = len(os.listdir(kodak_dir))
        kodak_generator = predict_generator_rgb(kodak_dir)
        res_Kodak = predict_generator(self.ad_model, kodak_generator, ls, data, 5, self.save_test)
        data['Kodak_IMGS_PSNR'] = res_Kodak[0]
        data['Kodak_IMGS_SSIM'] = res_Kodak[1]
        data['Kodak_AVG_PSNR'] = res_Kodak[2]
        data['Kodak_AVG_SSIM'] = res_Kodak[3]
        # Test McManus
        McM_dir = r'C:\Users\buggyr\Mosaic_Experiments\data\interim\McM'
        ls = len(os.listdir(McM_dir))
        McM_generator = predict_generator_rgb(McM_dir)
        res_McM = predict_generator(self.ad_model, McM_generator, ls, data, 5, self.save_test)
        data['McM_IMGS_PSNR'] = res_McM[0]
        data['McM_IMGS_SSIM'] = res_McM[1]
        data['McM_AVG_PSNR'] = res_McM[2]
        data['McM_AVG_SSIM'] = res_McM[3]
        # Write Results
        data['Parameters'] = {
            'Disc_Loss_Function': self.disc_loss_func,
            'Gen_Loss_Function': self.gen_loss_func,
            'Optimizer': str(type(self.disc_optimizer_func))
        }
        data['Training Set'] = {
            'Training Path': load_training,
        }
        with open(os.path.join(self.save_file, 'results.txt'), 'w') as outfile:
            json.dump(data, outfile, indent=4)

        return


class DiscModel:
    def __init__(self):
        self.save_file = None
        self.load_training = None
        self.disc_model = None
        self.batch_size = None
        self.save_epoch_models = None
        return

    def mk_file(self):
        save_results = r'C:\Users\buggyr\Mosaic_Experiments\models'
        keyname = "_descriminator1"
        now = datetime.datetime.now()
        self.save_file = os.path.join(save_results, now.strftime("%Y-%m-%d %H-%M") + keyname)
        os.mkdir(self.save_file)
        save_pred = os.path.join(self.save_file, 'Epoch_Predictions')
        os.mkdir(save_pred)
        self.save_epoch_models = os.path.join(self.save_file, 'Epoch_Models')
        os.mkdir(self.save_epoch_models)
        save_test = os.path.join(self.save_file, 'Test_Results')
        os.mkdir(save_test)

        return self.save_file

    def load_model(self, path):

        self.disc_model = keras.models.load_model(path)

        return self.disc_model

    def set_trainable(self, value=False):
        for layer in self.disc_model.layers:
            layer.trainable = value

    def build_model(self):

        inputs = Input(shape=(128, 128, 3))

        conv1 = Conv2D(16, (3, 3), activation='relu', strides=2, padding='same')(inputs)

        conv2 = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(conv1)

        conv3 = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(conv2)

        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)


        flatten = Flatten(name='flatten')(conv4)

        dense1 = Dense(512, activation='relu')(flatten)

        dense3 = Dense(2, activation='softmax')(dense1)

        self.disc_model = Model(inputs=[inputs], outputs=[dense3])

        optimizer_func = Adam(lr=1e-4)
        loss_func = 'categorical_crossentropy'
        self.disc_model.compile(optimizer=optimizer_func, loss=loss_func)

        self.disc_model.summary()

        return self.disc_model

    def pre_train(self, training_path, gen_model_path,
                  load_validation=r'C:\Users\buggyr\Mosaic_Experiments\data\interim\validation_tiled',
                  batch_size=32, epochs=50):
        self.batch_size = batch_size
        self.load_training = training_path
        gen_model = keras.models.load_model(gen_model_path)
        print("Loaded Gen-Model")
        fls = len(os.listdir(self.load_training))
        train_generator = train_generator_rgb_tiled(self.load_training, 64, self.batch_size)
        validation_generator = train_generator_rgb_tiled(load_validation, 64, self.batch_size)

        csv_logger = keras.callbacks.CSVLogger(os.path.join(self.save_file, 'training.log'), separator=',',
                                               append=False)
        csv_logger.set_model(self.disc_model)
        csv_logger.on_train_begin()

        loss_history = {'discriminator_loss': [],
                        'discriminator_acc': [], }

        print("Begin Disc_Model Pre-Train")
        num_batches = int(2 * fls)
        progress_bar = Progbar(target=num_batches)

        for ep in range(epochs):
            try:
                for i in range(2*fls):
                    inputs, input_orig = next(train_generator)

                    gen_batch = gen_model.predict_on_batch(inputs)

                    target_label = [0] * self.batch_size + [1] * self.batch_size

                    y_gan = np.asarray(target_label, dtype=np.int).reshape(-1, 1)
                    y_gan = to_categorical(y_gan, num_classes=2)
                    y_gan = smooth_gan_labels(y_gan)

                    input_gen = np.concatenate((gen_batch, input_orig), axis=0)

                    hist_loss = self.disc_model.fit(input_gen, y_gan, batch_size=self.batch_size, verbose=1)

                    loss_history['discriminator_loss'].append(hist_loss)

                    progress_bar.update(i + 1)

                print('Epoch: ', ep)
                self.disc_model.save(os.path.join(self.save_epoch_models, 'Epoch.' + str(ep) + '.h5'))

                csv_logger.on_epoch_end(epoch=ep, logs=loss_history)

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Stopping early and Saving Out.")

                self.disc_model.save(
                    os.path.join(self.save_epoch_models, 'Discriminator_2.Epoch.' + str(ep) + '.h5'))
                break
        with open(os.path.join(self.save_file, 'Model_Summary.txt'), 'w') as fh:
            json.dump(loss_history, fh)

        return self.disc_model, loss_history


if __name__ == "__main__":
    ########### Pre-Train Discriminator  #################
    load_training = r'C:\Users\buggyr\Mosaic_Experiments\data\external\Gharbi_tiled'
    mod_gen_path = r'C:\Users\buggyr\Mosaic_Experiments\models\2018-02-01 18-56_UNET_2_layer_64x64_mse_normal1_Patterns+Gharbi_2_input\DeMos_mod.h5'
    # disc_mod = DiscModel()
    #
    # disc_mod.mk_file()
    #
    # disc_mod.build_model()
    #
    #d_mod = disc_mod.pre_train(load_training, mod_gen_path, 50)
    ######################################################
    ########### Train AD #################################

    adm = AdModel()

    # adm.build_disc_model()
    #
    # adm.load_undefined_gen_model()
    #
    # adm.load_model(path=r'C:\Users\buggyr\Mosaic_Experiments\models\2018-03-16 20-55_Ad_Model\Epoch_Models\Adversarial_1.Epoch.10.h5')

    adm.load_disc_model()

    adm.assm_ad_model()

    adm.mk_file()

    adm.ad_model = adm.train(load_training=load_training, batch_size=32, epochs=200, initial_epoch=11)

    adm.model_test()