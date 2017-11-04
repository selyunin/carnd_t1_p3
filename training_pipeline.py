'''
Created on Nov 4, 2017
@author: selyunin
'''

import cv2
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

lines = []
training_data_path = "../training_data"
log_filename = "driving_log.csv"
img_folder = os.path.join(training_data_path, 'IMG/')
log_file = os.path.join(training_data_path, log_filename)

#read the data and remove abs paths
df = pd.read_csv(log_file, header=None,  names=['center_img', 'left_img', 'right_img', 'steering', 'throttle', 'brake', 'speed'])
df['center_img'] = pd.core.series.Series(map(lambda x : re.search('[a-zA-Z0-9_.]+$', x).group(0), df['center_img'] ))
df['left_img'] = pd.core.series.Series(map(lambda x : re.search('[a-zA-Z0-9_.]+$', x).group(0), df['left_img'] ))
df['right_img'] = pd.core.series.Series(map(lambda x : re.search('[a-zA-Z0-9_.]+$', x).group(0), df['right_img'] ))

df['center_img'] = img_folder + df['center_img']
df['left_img'] = img_folder + df['left_img']
df['right_img'] = img_folder + df['right_img']

np_images = np.array(list(map(lambda x : cv2.imread(x), df['center_img'])))
np_steering = np.array(df['steering'])

X_train = np_images
y_train = np_steering

#augment by flipping image 
X_train_augment = np.array(list(map(lambda x : cv2.flip(x,1), X_train)))
y_train_augment = np.array(list(map(lambda x : -1*x, y_train)))

X_train_new = np.concatenate((X_train, X_train_augment), axis=0)
y_train_new = np.concatenate((y_train, y_train_augment), axis=0)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Cropping2D

model = Sequential()
#pre-processing steps -- cropping & downsampling the image
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3) ))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4,4), padding='valid', data_format='channels_last'))
model.add(Lambda(lambda x : x/255. - 0.5 ))

#create a CNN
#input size (16 , 80, 24) 
model.add(Conv2D(filters=24, kernel_size=(3,3), 
                 strides=(1, 1), padding='valid', 
                 data_format='channels_last',
                 activation='relu'))
model.add(Dropout(rate=0.5))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=None, 
#                        padding='valid', data_format='channels_last'))
#input size (14 , 78 , 24) 
model.add(Conv2D(filters=36, kernel_size=(3,3), 
                 strides=(1, 1), padding='valid', 
                 data_format='channels_last',
                 activation='relu'))
model.add(Dropout(rate=0.75))
#input size (12 , 76 , 36)
model.add(Conv2D(filters=48, kernel_size=(3,3), 
                 strides=(1, 1), padding='valid', 
                 data_format='channels_last',
                 activation='relu'))
model.add(Dropout(rate=0.5))
#input size (10 , 74 , 48)
model.add(Conv2D(filters=64, kernel_size=(3,3), 
                 strides=(1, 1), padding='valid', 
                 data_format='channels_last',
                 activation='relu'))
model.add(Dropout(rate=0.5))
#input size (8, 72 , 64) 
model.add(Conv2D(filters=64, kernel_size=(5,5), 
                 strides=(1, 1), padding='valid', 
                 data_format='channels_last',
                 activation='relu'))
model.add(Dropout(rate=0.5))

# model.add(MaxPooling2D(pool_size=(2, 2), strides=None, 
#                        padding='valid', data_format='channels_last'))
#input size (4, 68 , 64) 
model.add(Flatten())

model.add(Dense(2048))
model.add(Dropout(rate=0.5))

model.add(Dense(512))
model.add(Dropout(rate=0.5))

model.add(Dense(32))
model.add(Dropout(rate=0.5))


#return steering
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

#train and save the model
model.fit(X_train_new, y_train_new, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')
model.save_weight('model_weights.h5')
model_architecture_json = model.to_json() 
with open('model_arch.json', 'w+') as f:
    f.write(model_architecture_json)
model_architecture_yaml = model.to_yaml()
with open('model_arch.yaml', 'w+') as f:
    f.write(model_architecture_yaml)