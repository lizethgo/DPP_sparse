# -*- coding: utf-8 -*-
"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : ResNet_pre_trained_ImageNet.py
                    File contains the pre-trained model ResNet50 for imageNet
                    
    Author        : Lizeth Gonzalez Carabarin
    Date          : nov/2020
    Reference     : https://keras.io/api/applications/#classify-imagenet-calsses-with-resnet50
==============================================================================
"""
from __future__ import print_function, absolute_import, division, unicode_literals
from __future__ import absolute_import
from __future__ import division

#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#import numpy as np
#
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Currently, memory growth needs to be the same across GPUs
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.experimental.set_visible_devices(gpus[2],'GPU')
#
#model = ResNet50(weights='imagenet')
#
#img_path = 'elephant.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#
#preds = model.predict(x)
## decode the results into a list of tuples (class, description, probability)
## (one such list for each sample in the batch)
#print('Predicted:', decode_predictions(preds, top=3)[0])
## Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
#
####################
### saving model
#import h5py
#
#weight = model.get_weights()
#
#file = h5py.File('ResNet_ImageNet.h5py','w')
#for i in range(len(weight)):
#    file.create_dataset('weight'+str(i),data=weight[i])    
#file.close()
#
#
#
#

from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

from numpy.random import seed
import numpy as np
seed(1)

### Loading cifar10
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
#x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

# normalize inputs from 0-255 to 0-1
#x_train = x_train / 255
#x_test = x_test / 255

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

x_val = x_train[-5000:,:,:,:]
x_train = x_train[:-5000,:,:,:]

#x_train = (x_train -((0.4914, 0.4822, 0.4465)))/(0.2023, 0.1994, 0.2010)
#x_test = (x_test -((0.4914, 0.4822, 0.4465)))/(0.2023, 0.1994, 0.2010)

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_val = y_train[-5000:,:]
y_train = y_train[:-5000,:]



# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, AveragePooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D
from sparseconnect_filters import sparseconnect_layer, sparseconnect_CNN_layer_filter, sparseconnect_mask

def res_identity(x, filters, stride): 
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters

  #first block 

  
 
  #first block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(stride, stride), padding='same', use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
  x = BatchNormalization()(x)
  #x = tf.keras.layers.Dropout(0.3)(x)
  x = Activation('relu')(x)
  #x = tf.keras.layers.Dropout(0.3)(x)

  
    #third block # bottleneck (but size kept same with padding)
    #kernel_regularizer=regularizers.l2(0.0005),
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
  x = BatchNormalization()(x)
  #x = tf.keras.layers.Dropout(0.3)(x)





  # add the input 
  x = Add()([x_skip, x])
  #x = tf.keras.layers.Dropout(0.3)(x)
  x = Activation('relu')(x)
  #x = tf.keras.layers.Dropout(0.3)(x)

  return x
  
  
def res_identity_exp(x, filters, stride): 
  #expand filter to match output

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters

  #first block 

  
 
  #first block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(stride, stride), padding='same',  use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
  x = BatchNormalization()(x)
  #x = tf.keras.layers.Dropout(0.3)(x)
  x = Activation('relu')(x)
  
  #x = tf.keras.layers.Dropout(0.3)(x)

  
    #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',  use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
  x = BatchNormalization()(x)
  #x = tf.keras.layers.Dropout(0.3)(x)
  
  
  x_skip = Conv2D(f1, kernel_size=(1, 1), strides=(stride, stride), padding='same',  use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x_skip)
  x_skip = BatchNormalization()(x_skip)



  # add the input 
  x = Add()([x_skip, x])
  #x = tf.keras.layers.Dropout(0.3)(x)
  x = Activation('relu')(x)
  #x = tf.keras.layers.Dropout(0.3)(x)

  return x

def res_identity_sparse(x, filters, channel_in, stride, ratio): 
  #renet block where dimension does not change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters
  size = np.shape(x)


  #first block 
  # bottleneck (but size kept same with padding)
  #x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0005))(x)
  x = sparseconnect_CNN_layer_filter(n_connect_ch = round(ratio*f1), 
                              filters = f1, 
                              kernel_size = 3,
                              padding = 'SAME',
                              channel_size = channel_in, 
                              strides = (stride, stride),
                              activation='relu',
                              n_epochs=250, 
                              tempIncr=5)(x)
                              

  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  size = np.shape(x)

  # third block activation used after adding the input
  x = sparseconnect_CNN_layer_filter(n_connect_ch = round(ratio*f1), 
                              filters = f1, 
                              kernel_size = 3,
                              padding = 'SAME',
                              channel_size = f1, 
                              strides = (1,1),
                              activation='relu',
                              n_epochs=250, 
                              tempIncr=5)(x)
  #x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.0005))(x)
  x = BatchNormalization()(x)


  # add the input 
  x = Add()([x_skip, x])
  x = Activation('relu')(x)
  #x = tf.keras.layers.Dropout(0.3)(x)

  return x

def res_identity_exp_sparse(x, filters, channel_in, stride, ratio): 
  #renet block where dimension does not change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters
  size = np.shape(x)


  #first block 
  # bottleneck (but size kept same with padding)
  #x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0005))(x)
  x = sparseconnect_CNN_layer_filter(n_connect_ch = round(ratio*f1), 
                              filters = f1, 
                              kernel_size = 3,
                              padding = 'SAME',
                              channel_size = channel_in, 
                              strides = (stride, stride),
                              activation='relu',
                              n_epochs=250, 
                              tempIncr=5)(x)
                              

  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  size = np.shape(x)

  # third block activation used after adding the input
  x = sparseconnect_CNN_layer_filter(n_connect_ch = round(ratio*f1), 
                              filters = f1, 
                              kernel_size = 3,
                              padding = 'SAME',
                              channel_size = f1, 
                              strides = (1,1),
                              activation='relu',
                              n_epochs=250, 
                              tempIncr=5)(x)
  #x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.0005))(x)
  x = BatchNormalization()(x)
  
  
  x_skip = sparseconnect_CNN_layer_filter(n_connect_ch = round(ratio*f1), 
                              filters = f1, 
                              kernel_size = 1,
                              padding = 'SAME',
                              channel_size = channel_in, 
                              strides = (stride,stride),
                              activation='relu',
                              n_epochs=250, 
                              tempIncr=5)(x)
  #x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(0.0005))(x)
  x_skip = BatchNormalization()(x_skip)


  # add the input 
  x = Add()([x_skip, x])
  x = Activation('relu')(x)
  #x = tf.keras.layers.Dropout(0.3)(x)

  return x










N_in = np.shape(x_train)[-2]
N_channel = np.shape(x_train)[-1]
N_out = np.size(y_train,-1)


x_ = Input(shape=(N_in, N_in, N_channel))

x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), use_bias=True)(x_)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = MaxPooling2D((3, 3), strides=(2, 2))(x)

#2nd stage 
# frm here on only conv block and identity block, no pooling

x = res_identity_exp(x, filters=(64, 64), stride=1)
x = res_identity(x, filters=(64, 64), stride=1)


# 3rd stage
x = res_identity_exp(x, filters=(128, 128), stride=2)
x = res_identity(x, filters=(128, 128), stride=1)
#x = res_identity_exp_sparse(x, filters=(128, 128), channel_in = 64, stride=2, ratio = 0.6)
#x = res_identity_sparse(x, filters=(128, 128), channel_in = 128, stride=1,  ratio = 0.6)


# 4th stage
x = res_identity_exp(x, filters=(256, 256), stride=2)
x = res_identity(x, filters=(256, 256), stride=1)
#x = res_identity_exp_sparse(x, filters=(256, 256), channel_in = 128, stride=2,  ratio = 0.5)
#x = res_identity_sparse(x, filters=(256, 256), channel_in = 256, stride=1,  ratio = 0.5)


# 5th stage
x = res_identity_exp(x, filters=(512, 512), stride=2)
x = res_identity(x, filters=(512, 512), stride=1)
#x = res_identity_exp_sparse(x, filters=(512, 512), channel_in = 256, stride=2,  ratio = 0.5)
#x = res_identity_sparse(x, filters=(512, 512), channel_in = 512, stride=1,  ratio = 0.5)


# ends with average pooling and dense connection

x = GlobalAveragePooling2D()(x)

x = Flatten()(x)


#x = tf.keras.layers.Dropout(0.4)(x)
#x = sparseconnect_layer(units=1000,n_connect=40,activation='relu', tempIncr=5)(x)
#y = sparseconnect_layer(units=10,n_connect=100,activation='softmax', tempIncr=5)(x)
#x = tf.keras.layers.Dropout(0.3)(x)
#x = Dense(512, kernel_initializer='he_normal')(x) #multi-class
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = tf.keras.layers.Dropout(0.3)(x)
y = Dense(10, activation='softmax', kernel_initializer='he_normal')(x) #multi-class

model = Model(inputs=x_, outputs=y)
model.summary()

from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    learning_rate = 1e-1
    if epoch < 150:
        return learning_rate * (0.1 ** (epoch // 100))
    if ((epoch >= 150) and (epoch < 200)):
        return learning_rate * (0.01 ** (epoch // 150))
    
    if epoch >= 200:
        return learning_rate * (0.001 ** (epoch // 200))
    
    
    
    #return learning_rate 



lr_scheduler = LearningRateScheduler(lr_schedule)

lr_decay = 1e-5
lr_drop = 50
    
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule(0), decay=lr_decay, momentum=0.9)

#callbacks = [
             #callbacks.training_vis()]

model.compile(optimizer = optimizer,
             loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

callback_model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
  filepath='weights.{epoch:02d}-{val_categorical_accuracy:.4f}.hdf5',
  #filepath='weights.{epoch:02d}.hdf5',
  monitor = 'val_categorical_accuracy',
  verbose = 0,
  save_best_only = True,
  save_weights_only = False,
  save_freq = 'epoch'
)

callbacks = [callback_model_checkpoint,
             lr_scheduler]



model = tf.keras.models.load_model('weights.108-0.9382.hdf5', custom_objects={'sparseconnect_mask':sparseconnect_mask})

results = model.evaluate(x_test, y_test, batch_size=128)
print(results)


