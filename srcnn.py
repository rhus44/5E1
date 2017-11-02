import os
import matplotlib.pyplot as plt
import  scipy.misc
import sklearn.feature_extraction
import numpy as np
import scipy.ndimage
sub_image_res = 33

def lowRes(high_res):
    #blur  = scipy.ndimage.gaussian_filter(high_res, sigma=3)
    resized  = scipy.misc.imresize(high_res, (sub_image_res // 3, sub_image_res //3))
    return scipy.misc.imresize(resized, (sub_image_res,sub_image_res), interp='bicubic')

def extractPatches(img_path):
    img  = scipy.misc.imread(img_path)
    return sklearn.feature_extraction.image.extract_patches_2d(img, (sub_image_res,sub_image_res), max_patches=272)
def getSubImages(images_dir):
    image_paths = [os.path.join(images_dir,filename) for filename in os.listdir(images_dir)]
    sub_images = np.array( map(extractPatches, image_paths))
    shape = np.shape(sub_images)
    sub_images = sub_images.reshape( (shape[0]*shape[1], shape[2],shape[3],shape[4]))

    low_res_sub_images = np.array( map(lowRes, sub_images))
    
    return sub_images, low_res_sub_images

project_dir  =  os.getcwd()
training_images_dir = os.path.join(project_dir, "Training Images")
test_images_dir = os.path.join(project_dir, "Test Images")
validation_images_dir = os.path.join(project_dir, "Validation Images")
print("Pre processing images")

#sub_images_train , low_res_sub_images_train = getSubImages(training_images_dir)
#sub_images_test , low_res_sub_images_test = getSubImages(test_images_dir)


#np.save('sub_images_train.npy', sub_images_train)
#np.save('low_res_sub_images_train.npy', low_res_sub_images_train)

#np.save('sub_images_test.npy', sub_images_test)
#np.save('low_res_sub_images_test.npy', low_res_sub_images_test)


sub_images_train = np.load('sub_images_train.npy')
low_res_sub_images_train = np.load('low_res_sub_images_train.npy')
sub_images_test = np.load('sub_images_test.npy')
low_res_sub_images_test = np.load('low_res_sub_images_test.npy')
print(np.shape(sub_images_train))




f1 = 9
f2 = 1
f3 = 5
n1 = 64
n2 = 32
c = 3

from keras.models import Sequential
from keras.layers import Convolution2D, Activation
from keras.optimizers import Adadelta, Nadam, RMSprop
from keras.models import load_model

model = Sequential()
##keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
model.add(Convolution2D(n1, f1, f1, border_mode='same', input_shape=(None, None,3), activation='relu')) 
model.add(Convolution2D(n2, f2, f2, border_mode='same', activation='relu'))

model.add(Convolution2D(c, f3, f3, border_mode='same'))

rmsprop = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
nadam = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)




model = load_model('super-res-model4.h5')
model.compile(optimizer=nadam,loss='mse')
history = model.fit(low_res_sub_images_train, sub_images_train, nb_epoch=350, batch_size=128)
print(history.history.keys())
model.save('super-res-model5.h5')
print(model.evaluate(low_res_sub_images_test, sub_images_test, verbose=1))

plt.imshow(low_res_sub_images_test[10])
plt.show()
pred_image = model.predict(np.array([low_res_sub_images_test[10]]), batch_size=1)[0]
print(np.shape(pred_image))
plt.imshow(pred_image.astype(np.uint8))
plt.show()
plt.imshow(sub_images_test[10])
plt.show()

mse = model.evaluate(np.array([low_res_sub_images_test[10]]), np.array([sub_images_test[10]]),batch_size=1)
print(mse)