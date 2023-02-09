from __future__ import print_function, absolute_import, division, unicode_literals
from __future__ import absolute_import
from __future__ import division

import time
import scipy.ndimage as nd
import numpy as np
from imageio import imread

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

tf.config.experimental.set_visible_devices(gpus[3],'GPU')

path = 'H:\\liz_datasets\\tiny-imagenet-200\\tiny-imagenet-200\\'

def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
  
def get_class_to_id_dict():
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open( path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result

def get_data(id_dict):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB') for i in range(500)]
        train_labels_ = np.array([[0]*200]*500)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()

    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(imread( path + 'val/images/{}'.format(img_name) ,pilmode='RGB'))
        test_labels_ = np.array([[0]*200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)
  
x_train, y_train, x_test, y_test = get_data(get_id_dictionary())

print( "train data shape: ",  x_train.shape )
print( "train label shape: ", y_train.shape )
print( "test data shape: ",   x_test.shape )
print( "test_labels.shape: ", y_test.shape )





from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import ZeroPadding2D
tf.compat.v1.disable_eager_execution()


mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)






from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, AveragePooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D
from sparseconnect_filters import sparseconnect_layer, sparseconnect_CNN_layer_filter
import tensorflow_addons as tfa


l2 = 0.0001
def res_identity(x, filters, stride): 
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters

  #first block 

  
 
  #first block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(stride, stride), padding='same', use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = BatchNormalization()(x)
  #x = tf.keras.layers.Dropout(0.3)(x)
  x = Activation('relu')(x)
  #x = tf.keras.layers.Dropout(0.3)(x)

  
    #third block # bottleneck (but size kept same with padding)
    #kernel_regularizer=regularizers.l2(0.0005),
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
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
  x = Conv2D(f1, kernel_size=(3, 3), strides=(stride, stride), padding='same',  use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = BatchNormalization()(x)
  #x = tf.keras.layers.Dropout(0.3)(x)
  x = Activation('relu')(x)
  
  #x = tf.keras.layers.Dropout(0.3)(x)

  
    #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',  use_bias=True, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = BatchNormalization()(x)
  #x = tf.keras.layers.Dropout(0.3)(x)
  
  
  x_skip = Conv2D(f1, kernel_size=(1, 1), strides=(stride, stride), padding='same',  use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(l2))(x_skip)
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
#x = tf.keras.layers.experimental.preprocessing.RandomCrop(56, 56)(x_)
x=tf.keras.layers.experimental.preprocessing.CenterCrop(56,56)(x_)
x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),  use_bias=False, kernel_regularizer=regularizers.l2(l2))(x)
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

x = AveragePooling2D((7,7), strides=(1,1))(x)
#x = tfa.layers.AdaptiveAveragePooling2D(x)


x = Flatten()(x)


x = tf.keras.layers.Dropout(0.1)(x)
#x = sparseconnect_layer(units=1000,n_connect=40,activation='relu', tempIncr=5)(x)
#y = sparseconnect_layer(units=10,n_connect=100,activation='softmax', tempIncr=5)(x)
#x = tf.keras.layers.Dropout(0.3)(x)
#x = Dense(512, kernel_initializer='he_normal')(x) #multi-class
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = tf.keras.layers.Dropout(0.3)(x)
x = Dense(200,  kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2))(x) #multi-class
#x = BatchNormalization()(x)
y = Activation('softmax')(x)

model = Model(inputs=x_, outputs=y)
model.summary()

from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    learning_rate = 1e-1
    if epoch < 55:
        return learning_rate * (0.2 ** (epoch // 40))
    if ((epoch >= 55) and (epoch < 65)):
        return learning_rate * (0.04 ** (epoch // 55))
    
    if epoch >= 65:
        return learning_rate * (0.008 ** (epoch // 65))
    
    
    
    
    
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
    #brightness_range = (0.2,0.8),
    rotation_range=12,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)
    #preprocessing_function = random_crop)  # randomly flip images
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



history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=128),
                            steps_per_epoch=x_train.shape[0] // 128,
                            epochs=65,
                            validation_data=(x_test, y_test),callbacks=callbacks,verbose=1)



plt.plot(history.history['val_categorical_accuracy'])
plt.plot(history.history['categorical_accuracy'])
plt.legend(['validation', 'train'])
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()
