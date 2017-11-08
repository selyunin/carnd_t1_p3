'''
Created on Nov 6, 2017
@author: selyunin
'''

import re
import math
import pandas as pd
import numpy as np
from os.path import join
from cnn_model import CNN
from sklearn.model_selection import train_test_split
from generators import training_generator, validation_generator
from hyperparams import HyperParams
from keras.optimizers import Adam

hyper_params = HyperParams()
cnn = CNN()
cnn.create_model()
cnn.model.compile(loss='mse', optimizer = Adam(lr=hyper_params.LEARNING_RATE))
training_data_path = "../new_training_data"
log_filename = "driving_log.csv"
img_folder = join(training_data_path, 'IMG/')
log_file = join(training_data_path, log_filename)

df = pd.read_csv(log_file, header=None,  names=['center_img', 'left_img', 'right_img', 'steering', 'throttle', 'brake', 'speed'])
df['center_img'] = pd.core.series.Series(map(lambda x : re.search('[a-zA-Z0-9_.]+$', x).group(0), df['center_img'] ))
df['left_img'] = pd.core.series.Series(map(lambda x : re.search('[a-zA-Z0-9_.]+$', x).group(0), df['left_img'] ))
df['right_img'] = pd.core.series.Series(map(lambda x : re.search('[a-zA-Z0-9_.]+$', x).group(0), df['right_img'] ))
df['center_img'] = img_folder + df['center_img']
df['left_img'] = img_folder + df['left_img']
df['right_img'] = img_folder + df['right_img']
df_train, df_valid = train_test_split(df, test_size=0.2)
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

TRAINING_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 32
STEPS_PER_EPOCH = math.ceil(df_train.shape[0]*6 / TRAINING_BATCH_SIZE)
VALIDATION_STEPS = math.ceil(df_valid.shape[0]*6 / VALIDATION_BATCH_SIZE)

train_gen = training_generator(df_train, batch_size=TRAINING_BATCH_SIZE)
valid_gen = validation_generator(df_valid, batch_size=VALIDATION_BATCH_SIZE)


history_object = cnn.model.fit_generator(generator=train_gen,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=7,
                        validation_data=valid_gen,
                        validation_steps=VALIDATION_STEPS)
 
#get directory with the training data
#read the training data 
#read previously saved model
#launch training on the 
