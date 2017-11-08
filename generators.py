'''
Created on Nov 8, 2017
@author: selyunin
'''

import numpy as np
from PIL import Image
import cv2
from sklearn.utils import shuffle

def training_generator(df, batch_size=128):
    num_images = df.shape[0]
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_images, batch_size):
#             print("offset: {}".format(offset))
            batch_df = df.loc[offset:offset+batch_size]
#             X_train_new, y_train_new = get_data(batch_df) 
            X_train_center = np.array(list(map(lambda x : np.asarray(Image.open(x)), batch_df['left_img'])))
            y_train_center = np.array(batch_df['steering'])
            X_train_right = np.array(list(map(lambda x : np.asarray(Image.open(x)), batch_df['right_img'])))
            y_train_right = np.array(batch_df['steering']) - 0.115
            X_train_left = np.array(list(map(lambda x : np.asarray(Image.open(x)), batch_df['left_img'])))
            y_train_left = np.array(batch_df['steering']) + 0.115
            
            X_train_lrc = np.concatenate((X_train_center, X_train_left, X_train_right), axis=0)
            y_train_lrc = np.concatenate((y_train_center, y_train_left, y_train_right), axis=0)
            
            X_train_flip = np.array(list(map(lambda x : cv2.flip(x,1), X_train_lrc)))
            y_train_flip = np.array(list(map(lambda x : -1.*x, y_train_lrc)))
            #concatenate left, center, right and flipped img
            X_train_new = np.concatenate((X_train_lrc, X_train_flip), axis=0)
            y_train_new = np.concatenate((y_train_lrc, y_train_flip), axis=0)

            yield shuffle(X_train_new, y_train_new)


def validation_generator(df, batch_size=32):
    num_images = df.shape[0]
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_images, batch_size):
#             print("offset: {}".format(offset))
            batch_df = df.loc[offset:offset+batch_size]
#             X_valid_new, y_valid_new = get_data(batch_df) 
            X_train_center = np.array(list(map(lambda x : np.asarray(Image.open(x)), batch_df['left_img'])))
            y_train_center = np.array(batch_df['steering'])
            X_train_right = np.array(list(map(lambda x : np.asarray(Image.open(x)), batch_df['right_img'])))
            y_train_right = np.array(batch_df['steering']) - 0.115
            X_train_left = np.array(list(map(lambda x : np.asarray(Image.open(x)), batch_df['left_img'])))
            y_train_left = np.array(batch_df['steering']) + 0.115
            
            X_train_lrc = np.concatenate((X_train_center, X_train_left, X_train_right), axis=0)
            y_train_lrc = np.concatenate((y_train_center, y_train_left, y_train_right), axis=0)
            
            X_train_flip = np.array(list(map(lambda x : cv2.flip(x,1), X_train_lrc)))
            y_train_flip = np.array(list(map(lambda x : -1.*x, y_train_lrc)))
            #concatenate left, center, right and flipped img
            X_valid_new = np.concatenate((X_train_lrc, X_train_flip), axis=0)
            y_valid_new = np.concatenate((y_train_lrc, y_train_flip), axis=0)
            yield shuffle(X_valid_new, y_valid_new)
