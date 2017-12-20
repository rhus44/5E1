import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
import scipy.misc
import sklearn.feature_extraction
import numpy as np
import scipy.ndimage
from skimage import measure, io
from skimage import transform
import train_rgb
import skimage
import json
import datetime
import pickle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

save_results = r'C:\Users\buggyr\Mosaic_Experiments\models'
load_training = r'C:\Users\buggyr\Mosaic_Experiments\data\external\Training Images 3'
load_validation = r'C:\Users\buggyr\Mosaic_Experiments\data\external\Val_data'


import keras
from keras.models import *
from keras.models import Sequential
from keras.layers import Conv2D, Activation, UpSampling2D,Reshape, MaxPooling2D, Dropout, Cropping2D, merge, Input, concatenate, Conv2DTranspose
from keras.optimizers import Adadelta, Nadam, RMSprop
from keras.models import load_model


f1 = 64
f2 = 32
f3 = 32
ff = 3
k1 = (9,9)
k2 = (1,1)
k3 = (5,5)
k4 = (3,3)

inputs = Input(shape = (None, None, 4))

conv1 = Conv2D(f1, k4, activation='relu', padding='same')(inputs)

conv2 = Conv2D(f1, k4, activation='relu', padding='same')(conv1)

up1 = Conv2DTranspose(f1, k4 , strides=(2, 2), padding='same')(conv2)

conv3 = Conv2D(f1, k4, activation='relu', padding='same')(up1)

conv4 = Conv2D(f1, k4, activation='relu', padding='same')(conv3)

conv5 = Conv2D(f1, k4, activation='relu', padding='same')(conv4)

conv6 = Conv2D(f1, k4, activation='relu', padding='same')(conv5)

conv7 = Conv2D(f1, k4, activation='relu', padding='same')(conv6)

conv8 = Conv2D(f1, k4, activation='relu', padding='same')(conv7)

conv9 = Conv2D(f1, k4, activation='relu', padding='same')(conv8)

conv10 = Conv2D(ff, k4, activation='relu', padding='same')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])

modsum = model.summary()


keyname = "_10_layer_Transpose"
now=datetime.datetime.now()
save_file=os.path.join(save_results,now.strftime("%Y-%m-%d %H-%M")+keyname)
os.mkdir(save_file)
save_pred = os.path.join(save_file,'Epoch_Predictions')
os.mkdir(save_pred)
save_model = os.path.join(save_file,'Epoch_Models')
os.mkdir(save_model)
save_test = os.path.join(save_file,'Test_Results')
os.mkdir(save_test)

with open(os.path.join(save_file,'Model_Summary.txt'),'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

#rmsprop = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer_func = Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
loss_func='mse'

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
tbCallBack = keras.callbacks.TensorBoard(log_dir=os.path.join(save_file,'TNSR_BRD'), histogram_freq=0, write_graph=True, write_images=True)
csv_logger = keras.callbacks.CSVLogger(os.path.join(save_file,'training.log'), separator=',', append=False)
epoch_predict = train_rgb.Save_predictions(save_pred)
model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(save_model,'model.{epoch:02d}-{loss:.2f}.hdf5'), monitor='loss')

model.compile(optimizer=optimizer_func,loss=loss_func)


fls = len(os.listdir(load_training))
val_steps = len(os.listdir(load_validation))
                            #train_dir, patch_size, batch_size
train_generator = train_rgb.train_generator_rgb(load_training,64,32)
val_generator = train_rgb.val_generator_rgb(load_validation)


history = model.fit_generator(generator = train_generator,steps_per_epoch=3*fls,
                              verbose = 1, epochs = 200,callbacks = [tbCallBack,csv_logger,
                                epoch_predict, model_checkpoint],
                                    validation_data = val_generator, validation_steps = val_steps)
print(history.history)

model.save(os.path.join(save_file,'DeMos_mod.h5'))

#Test Kodak
data ={}

kodak_dir = r'C:\Users\buggyr\Mosaic_Experiments\data\interim\Kodak'
ls = len(os.listdir(kodak_dir))

kodak_generator = train_rgb.predict_generator_rgb(kodak_dir)

# k_pred = model.predict_generator(kodak_generator, steps = ls)

res_Kodak = train_rgb.predict_generator(model,kodak_generator,ls,data,5,save_test)

data['Kodak_IMGS_PSNR'] = res_Kodak[0]
data['Kodak_IMGS_SSIM'] = res_Kodak[1]
data['Kodak_AVG_PSNR']  = res_Kodak[2]
data['Kodak_AVG_SSIM']  = res_Kodak[3]

#Test McManus
McM_dir = r'C:\Users\buggyr\Mosaic_Experiments\data\interim\McM'
ls = len(os.listdir(McM_dir))

McM_generator = train_rgb.predict_generator_rgb(McM_dir)

# k_pred = model.predict_generator(kodak_generator, steps = ls)

res_McM = train_rgb.predict_generator(model,McM_generator,ls,data,5,save_test)

data['McM_IMGS_PSNR'] = res_McM[0]
data['McM_IMGS_SSIM'] = res_McM[1]
data['McM_AVG_PSNR']  = res_McM[2]
data['McM_AVG_SSIM']  = res_McM[3]

#Write Results
data['Parameters'] = {
    'Loss Function': loss_func,
    'Optimizer':str(type(optimizer_func))
}
data['Training Set'] = {
    'Training Path': load_training,
}

with open(os.path.join(save_file,'results.txt'), 'w') as outfile:
    json.dump(data, outfile,indent=4)
