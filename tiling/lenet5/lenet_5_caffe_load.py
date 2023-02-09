# -*- coding: utf-8 -*-
"""
#################################################################################
    Paper ID     : 12076
    Title        : Dynamic Probabilistic Pruning: Training sparse networks based on stochastic and dynamic masking
#################################################################################
        
    Source Name  : lenet_5_caffe.py
    Description  : Main file for MNIST using Lenet5-Caffe. The network is pruned based on Dynamic Probabilistic
                   Pruning (DPP). 

#################################################################################
"""

from __future__ import print_function, absolute_import, division, unicode_literals
from __future__ import absolute_import
from __future__ import division


import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)

########################################################################################################
########################################################################################################
# Activate the following lines for GPU's usage

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Currently, memory growth needs to be the same across GPUs
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.experimental.set_visible_devices(gpus[1],'GPU')


########################################################################################################
########################################################################################################
#%% Load data  
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

x_val = x_train[-5000:,:,:,:]
x_train = x_train[:-5000,:,:,:]


# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
y_val = y_train[-5000:,:]
y_train = y_train[:-5000,:]

########################################################################################################
########################################################################################################
#%% Define sparseConnect Model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sparseconnect import sparseconnect_layer, sparseconnect_CNN_layer

# Parameters
n_epochs = 40

### Fully-connected layer
n_nodes = 500 # units, nr of active output nodes
n_connect = 25 # nr of active input connections per output node

### CNN parameters (convolutional layer 1)
n_filters_1 = 20 # number of filters
n_connect_CNN_1 = 20 # number of active inputs per element of kernel
kernel_size_1 = 20  # kernel size
channel_size_1 = 1 # channel size

### CNN parameters (convolutional layer 2)
n_filters_2 = 50 # number of filters
n_connect_CNN_2 =20 # number of active inputs per element of kernel
kernel_size_2 = 5  # kernel size
channel_size_2 = 20 # channel size


N_in = np.shape(x_train)[-2]
N_channel = np.shape(x_train)[-1]
N_out = np.size(y_train,-1)

x = Input(shape=(N_in, N_in, N_channel))
x_1 = sparseconnect_CNN_layer( 
                              n_connect = n_connect_CNN_1, 
                              filters = n_filters_1, 
                              kernel_size = kernel_size_1, 
                              channel_size = channel_size_1, 
                              activation='relu',
                              n_epochs=n_epochs, 
                              tempIncr=5)(x)
x_2 = sparseconnect_CNN_layer(
                              n_connect = n_connect_CNN_2, 
                              filters = n_filters_2, 
                              kernel_size = kernel_size_2, 
                              channel_size = channel_size_2, 
                              activation='relu',
                              n_epochs=n_epochs, 
                              tempIncr=5)(x_1)
x_3 = tf.keras.layers.Flatten()(x_2)

x_4 = sparseconnect_layer(units=500,n_connect=10,activation='relu',n_epochs=n_epochs, tempIncr=5)(x_3)

y = Dense(10,activation='softmax')(x_4)
model = Model(inputs=x, outputs=y)

model.summary()
  
########################################################################################################
########################################################################################################
#%% Start training

import callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


model.compile(optimizer = optimizer,
             loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

### loading best model for testing
model = tf.keras.models.load_model('weights.13-0.9924.hdf5',  custom_objects={'sparseconnect_layer':sparseconnect_layer,'sparseconnect_CNN_layer':sparseconnect_CNN_layer})

model.summary()


results = model.evaluate(x_test, y_test, batch_size=64)
print(results)
