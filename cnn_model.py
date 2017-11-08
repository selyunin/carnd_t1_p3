'''
Created on Nov 7, 2017
@author: selyunin
'''

from os.path import join
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Cropping2D
from keras.optimizers import Adam
from generators import training_generator, validation_generator

class CNN:
    def __init__(self):
        self.model_dir = './'
        self.golden_model_name = 'model.h5'
        self.model_json = join(self.model_dir, 'model_arch.json')
        self.model_yaml = join(self.model_dir, 'model_arch.yaml')
        self.golden_model_file = join(self.model_dir, self.golden_model_name)
        self.model_checkpoint = ''
    
    def create_model(self):
        self.model = Sequential()
        #pre-processing steps -- cropping & downsampling the image
        self.model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3) ))
        self.model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format='channels_last'))
        self.model.add(Lambda(lambda x : x/255. - 0.5 ))
        
        #create a CNN
        #input size (16 , 80, 24) 
        self.model.add(Conv2D(filters=24, kernel_size=(3,3), 
                         strides=(1, 1), padding='valid', 
                         data_format='channels_last',
                         activation='elu',
                         name='Conv2D_l1'))
        self.model.add(Dropout(rate=0.5))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=None, 
        #                        padding='valid', data_format='channels_last'))
        #input size (14 , 78 , 24) self.
        self.model.add(Conv2D(filters=36, kernel_size=(3,3), 
                         strides=(1, 1), padding='valid', 
                         data_format='channels_last',
                         activation='elu',
                         name='Conv2D_l2'))
        self.model.add(Dropout(rate=0.5))
        #input size (12 , 76 , 36)
        self.model.add(Conv2D(filters=48, kernel_size=(3,3), 
                         strides=(1, 1), padding='valid', 
                         data_format='channels_last',
                         activation='elu',
                         name='Conv2D_l3'))
        self.model.add(Dropout(rate=0.5))
        #input size (10 , 74 , 48)
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), 
                         strides=(1, 1), padding='valid', 
                         data_format='channels_last',
                         activation='elu',
                         name='Conv2D_l4'))
        self.model.add(Dropout(rate=0.5))
        #input size (8, 72 , 64) 
        self.model.add(Conv2D(filters=64, kernel_size=(5,5), 
                         strides=(1, 1), padding='valid', 
                         data_format='channels_last',
                         activation='elu',
                         name='Conv2D_l5'))
        self.model.add(Dropout(rate=0.5))
        
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=None, 
                               padding='valid', data_format='channels_last'))
        #input size (4, 68 , 64) 
        #input size (2, 34 , 64) 
        self.model.add(Flatten())

        self.model.add(Dense(100, activation='elu'))
        self.model.add(Dropout(rate=0.5))
        
        self.model.add(Dense(50, activation='elu'))
        self.model.add(Dropout(rate=0.5))
        
        self.model.add(Dense(10, activation='elu'))
        self.model.add(Dropout(rate=0.5))
        #return steering
        self.model.add(Dense(1))
    
    def restore_golden_model(self):
        self.load(self.model.load_weights(self.golden_model_file))
    
    def save_golden_model(self):
        self.model.save(self.golden_model_file)
    
    def save_json_model(self):
        model_arch_json = self.model.to_json() 
        with open(self.model_json, 'w+') as f:
            f.write(model_arch_json)
            
    def save_yaml_model(self):
        model_arch_yaml = self.model.to_yaml() 
        with open(self.model_yaml, 'w+') as f:
            f.write(model_arch_yaml)
