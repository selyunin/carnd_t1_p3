#!/usr/bin/env python

'''
Created on Nov 8, 2017
@author: selyunin
'''
import sys
import argparse
import datetime
import os
from cnn_model import CNN
from keras.models import load_model
from keras.optimizers import Adam
from os.path import join
import pandas as pd
from sklearn.model_selection import train_test_split
from generators import training_generator, validation_generator
import re
import math
from keras.callbacks import ModelCheckpoint


def restore_model(args):
    cnn = CNN()
    if os.path.isfile(args.input_model):
        try:
            cnn.model = load_model(args.input_model)
        except:
            print("Model not loaded, exit..")
            sys.exit()
        print("Model {} restored".format(args.input_model))
    else:
        print("Model not found, creating new model")
        cnn.create_model_v1_2()
    return cnn.model

def read_training_data(args):
    training_data_path = args.img_folder
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
    return df

def train_model(model, args, df):
    df_train, df_valid = train_test_split(df, test_size=0.2)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    model.compile(loss='mse', optimizer = Adam(lr=args.learning_rate))
    TRAINING_BATCH_SIZE = args.batch_size
    VALIDATION_BATCH_SIZE = 32
#     STEPS_PER_EPOCH = math.ceil(df_train.shape[0]*6 / TRAINING_BATCH_SIZE)
#     VALIDATION_STEPS = math.ceil(df_valid.shape[0]*6 / VALIDATION_BATCH_SIZE)
    train_gen = training_generator(df_train, batch_size=TRAINING_BATCH_SIZE)
    valid_gen = validation_generator(df_valid, batch_size=VALIDATION_BATCH_SIZE)
    filepath="weights-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    TRAINING_SAMPLES = df_train.shape[0]*6
    VALIDATION_SAMPLES = df_valid.shape[0]*6
    history_object = model.fit_generator(generator=train_gen,
                            samples_per_epoch=TRAINING_SAMPLES,
                            nb_epoch=7,
                            validation_data=valid_gen,
                            nb_val_samples=VALIDATION_SAMPLES,
                            callbacks=callbacks_list, 
                            )

def save_model(model, args):
    model.save_model(args.output_model)


def _main():
    description = "Train CNN to steer the car simulator\n" + \
    "takes an existing model and image folder, and trains a model on new images\n"+ \
    "./model.py -in_m oldmodel.h5 -out_m newmodel.h5 -f /image/folder -b 64 -l_r 5e-4"
    time_now = datetime.datetime.now().strftime("_%y_%m_%d_%H_%M_%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-in_m", "--input_model", type=str, default='model.h5')
    parser.add_argument("-out_m", "--output_model", type=str, default='model' + time_now + '.h5')
    parser.add_argument("-f", "--img_folder", type=str, default='../training_data')
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-l_r", "--learning_rate", type=float, default=1e-4)
    args = parser.parse_args(sys.argv[1:])
    
    for key, value in (vars(args)).items():
        print("{:15s} -> {}".format(key, value))
    
    model = restore_model(args)
    df = read_training_data(args)
    train_model(model, args, df)
    save_model(model, args)
    
if __name__ == '__main__':
    _main()
