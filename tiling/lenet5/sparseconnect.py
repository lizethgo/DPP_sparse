# -*- coding: utf-8 -*-
"""

#################################################################################
    Paper ID: 12076
    Title: Dynamic Probabilistic Pruning: Training sparse networks based on stochastic and dynamic masking
#################################################################################
    
    Source Name   :  sparseconnect.py
    Description   :  This file contains the sparse layers (fully-connected and convolutional) and the main algorithm of 
                     Dynamic Probabilistic Pruning. 

#################################################################################          

"""

import tensorflow as tf
import temperatureUpdate
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda, Activation
from tensorflow.keras import regularizers
import numpy as np


# Entropy penalization fr logits
class entropy_reg(tf.keras.regularizers.Regularizer):

    def __init__(self, entropyMult):
        self.entropyMult = entropyMult

    def __call__(self, logits):
        normDist = tf.nn.softmax(logits,1)
        logNormDist = tf.math.log(normDist+1e-20)
        
        rowEntropies = -tf.reduce_sum(tf.multiply(normDist, logNormDist),1)
        sumRowEntropies = tf.reduce_sum(rowEntropies)
        
        multiplier = self.entropyMult
        return multiplier*sumRowEntropies

    def get_config(self):
        return {'entropyMult': float(self.entropyMult)}




##########################################################################################################################
############################################################################################################################
# This class includes the forward and backward part to optimize logits and generate the masks for creating the sparse net 
    
class DPS_topk(Layer):
    
    """
    - DPS_topK optimizes logits, and gets k samples from each categorical distribution 
    - It returns hardSamples during forwardpass
    - It uses SoftSamples during backward pass
    
    """
    
    def __init__(self, is_CNN, BS, k, batchPerEpoch=1, n_epochs=2,  tempIncr=5, name=None, **kwargs):
        self.is_CNN = is_CNN
        self.BS = BS                     # The dynamic batch_size parameter
        self.k = k                       # Define the number of weights per output node that should be used. (k <= input_nodes)
        self.batchPerEpoch = batchPerEpoch  # Amount of batch updates per epoch
        self.n_epochs = n_epochs            # Total number of epochs used for training
        self.tempIncr = tempIncr        # Value with which the temperature in the softmax is multiplied at the beginning of training
        
        #self.outShape = (self.BS, self., self.k, self.input_nodes)

        super(DPS_topk, self).__init__(name=name,**kwargs) 

    def build(self, input_shape):
        self.step = K.variable(0)
        super(DPS_topk, self).build(input_shape)  
      
    def call(self, inp):

               
        if self.is_CNN == True:
            ## if kernels are sparsifying, inputs are just flattened, after same method as the one for the fully-connected case is followed
            #inp = tf.reshape(inp, [inp.shape[1], (inp.shape[0])*(inp.shape[0]), inp.shape[-1]])
            data_shape = tf.shape(inp)
            logits = inp  #[output_nodes,input_nodes]
            
            ### Forwards ###
            GN = -0.01*tf.math.log(-tf.math.log(tf.random.uniform((self.BS,data_shape[0], data_shape[1], data_shape[2]),0,1)+1e-20)+1e-20) #[BS,output_nodes,input_nodes]
            perturbedLog = logits+GN #[BS,output_nodes, ch, input_nodes]            
            # Find the top-k indices. Apply top_k second time to sort them from high to low
            ind =  tf.nn.top_k(tf.nn.top_k(perturbedLog, k=self.k).indices,k=self.k).values  #[BS,output_nodes, ch, k]
            # Reverse the sorting to have the indices from low to high
            topk = tf.reverse(tf.expand_dims(ind,-1), axis=[3]) #[BS,output_nodes,ch, k]          
            hardSamples = tf.squeeze(tf.one_hot(topk,depth=data_shape[-1]),axis=-2) #[BS,output_nodes,ch,k,input_nodes]
          
            ### Backwards ###
            epoch = self.step/self.batchPerEpoch
            Temp = temperatureUpdate.temperature_update_tf(self.tempIncr, epoch, self.n_epochs)
            updateSteps = []
            updateSteps.append((self.step, self.step+1))
            self.add_update(updateSteps,inp)   
            prob_exp = tf.tile(tf.expand_dims(tf.expand_dims(tf.exp(logits),0),3),(self.BS,1,1,self.k,1)) #[BS,output_nodes,ch,k,input_nodes]  
            cumMask = tf.cumsum(hardSamples,axis=-2, exclusive=True) #[BS,output_nodes,ch,k,input_nodes]
            softSamples = tf.nn.softmax((tf.math.log(tf.multiply(prob_exp,1-cumMask+1e-20))+tf.tile(tf.expand_dims(GN,-2),(1,1,1,self.k,1)))/Temp, axis=-1)  #[BS,output_nodes,k,input_nodes]
   
            return tf.stop_gradient(hardSamples - softSamples) + softSamples
            
        else: # This is for the fully connected case   
            
            data_shape = tf.shape(inp)
            logits = inp  #[output_nodes,input_nodes]
            ### Forwards ###
            GN = -0.1*tf.math.log(-tf.math.log(tf.random.uniform((self.BS,data_shape[0], data_shape[1]),0,1)+1e-20)+1e-20) #[BS,output_nodes,input_nodes]
            perturbedLog = logits+GN #[BS,output_nodes,input_nodes]     
            # Find the top-k indices. Apply top_k second time to sort them from high to low
            ind =  tf.nn.top_k(tf.nn.top_k(perturbedLog, k=self.k).indices,k=self.k).values  #[BS,output_nodes,k]
            # Reverse the sorting to have the indices from low to high
            topk = tf.reverse(tf.expand_dims(ind,-1), axis=[2]) #[BS,output_nodes,k]           
            hardSamples = tf.squeeze(tf.one_hot(topk,depth=data_shape[-1]),axis=-2) #[BS,output_nodes,k,input_nodes]

            
            ### Backwards ###
            epoch = self.step/self.batchPerEpoch
            Temp = temperatureUpdate.temperature_update_tf(self.tempIncr, epoch, self.n_epochs)
            updateSteps = []
            updateSteps.append((self.step, self.step+1))
            self.add_update(updateSteps,inp)

            prob_exp = tf.tile(tf.expand_dims(tf.expand_dims(tf.exp(logits),0),2),(self.BS,1,self.k,1)) #[BS,output_nodes,k,input_nodes]  
            cumMask = tf.cumsum(hardSamples,axis=-2, exclusive=True) #[BS,output_nodes,k,input_nodes]
            softSamples = tf.nn.softmax((tf.math.log(tf.multiply(prob_exp,1-cumMask+1e-20))+tf.tile(tf.expand_dims(GN,-2),(1,1,self.k,1)))/Temp, axis=-1)  #[BS,output_nodes,k,input_nodes]    
            return tf.stop_gradient(hardSamples - softSamples) + softSamples
        
    

    
##########################################################################################################################
############################################################################################################################

# This class creates sparse layers for fully-connected case. 
class sparseconnect_layer(Layer):  
    """
    Sparse fully-connected layer
    - Generates trainable logits (D)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify W matrix
    """
    def __init__(self,units, n_connect, activation=None, n_epochs=10, tempIncr=5, name=None, one_per_batch=True, **kwargs):
        self.units = units
        self.n_connect = n_connect
        self.activation = activation
        self.n_epochs = n_epochs
        self.tempIncr = tempIncr
        self.one_per_batch = one_per_batch 
        super(sparseconnect_layer, self).__init__(name=name, **kwargs) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units':self.units, 
            'n_connect':self.n_connect, 
            'activation':self.activation,
            'n_epochs':self.n_epochs,
            'tempIncr': self.tempIncr,
            'one_per_batch':self.one_per_batch
        })
        return config

    def build(self, input_shape): 
        # Define weight matrix and bias vector
        self.W = self.add_weight(shape=[self.units,int(input_shape[-1])],
                                 initializer='glorot_uniform',
                                 #regularizer=regularizers.l2(0.0005),
                                 trainable=True, name='w_bin', dtype='float32')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='glorot_normal',
                                 trainable=True, name='b_bin',  dtype='float32')
        
        # Define sampling logits
        self.D = self.add_weight(name='TrainableLogits', 
                              shape=(self.units, input_shape[-1]),
                              initializer = tf.random_normal_initializer(mean=0, stddev=0.5, seed=None),
                              regularizer = entropy_reg(5e-5),
                              #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None), #TODO Liz: choose which initializer might be suitable for you
                              trainable=True)

        super(sparseconnect_layer, self).build(input_shape) 
 
    def call(self, x):
        units = self.units


        batch_size = K.shape(x)[0]
        # Produce sparse sampling matrix
        A = DPS_topk(is_CNN = False, BS=1, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D)
        A = tf.reduce_sum(A,axis=-2)
        A = tf.reduce_sum(A,axis=0)
        
        print(K.int_shape(A))
        
        # Produce sparse weight matrix
        AW = Lambda(lambda inp: tf.multiply(inp[0],inp[1]), output_shape = (units,units))([A,self.W])

        print(K.int_shape(AW))

        # Produce layer output
        y = Lambda(lambda inp: K.dot(inp[1],tf.transpose(inp[0],(1,0)))+inp[2], output_shape = (units))([AW,x,self.b])
                               
        if not self.activation == None:
            y = Activation(self.activation)(y)
        
        return y

##########################################################################################################################
############################################################################################################################

# This class creates sparse layers for convolutional case.    
class sparseconnect_CNN_layer(Layer): 
    """
    Sparse convolutional layer
    - Generates trainable logits (D)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify k matrix (kernels)
    """
    def __init__(self,
                 n_connect,
                 filters,
                 kernel_size,
                 channel_size,
                 cnn_sparse = True, 
                 activation=None, 
                 n_epochs=10, 
                 tempIncr=5, 
                 name=None, 
                 one_per_batch=True,
                 strides = (1,1,1,1),
                 padding = 'SAME',
                 **kwargs):
        self.n_connect = n_connect
        self.filters = filters
        self.kernel_size = kernel_size
        self.channel_size = channel_size
        self.cnn_sparse = cnn_sparse
        self.activation = activation
        self.n_epochs = n_epochs
        self.tempIncr = tempIncr
        self.one_per_batch = one_per_batch 
        self.strides = strides
        self.padding = padding
        
        super(sparseconnect_CNN_layer, self).__init__(name=name, **kwargs) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_connect':self.n_connect, 
            'filters':self.filters, 
            'kernel_size':self.kernel_size,
            'channel_size':self.channel_size,
            'cnn_sparse': self.cnn_sparse,
            'activation':self.activation,
            'n_epochs':self.n_epochs,
            'tempIncr':self.tempIncr,
            'strides':self.one_per_batch,
            'padding':self.padding,
            'strides':self.strides
        })
        return config

    def build(self, input_shape): 

        self.k = self.add_weight(shape=[self.kernel_size,self.kernel_size, self.channel_size, self.filters],
                              initializer='glorot_uniform',
                              #regularizer=regularizers.l2(0.0005),
                              trainable=True,  dtype='float32')
        self.b = self.add_weight(shape=[int(self.filters)]
                                 ,initializer='glorot_uniform',trainable=True,  dtype='float32')

        self.D = self.add_weight(name='TrainableLogits_CNN', 
                              shape=[self.channel_size, self.kernel_size*self.kernel_size, self.filters],
                              initializer = tf.random_normal_initializer(mean=0, stddev=0.5, seed=None),
                              regularizer = entropy_reg(5e-5),
                              #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None), #TODO Liz: choose which initializer might be suitable for you
                              trainable=True)

        super(sparseconnect_CNN_layer, self).build(input_shape) 
 
    def call(self, x):
        print('x shape', np.shape(x))

        batch_size = K.shape(x)[0]
        A = tf.ones(shape=(self.k.shape))
        
        if self.cnn_sparse == True:
            # Produce sparse sampling matrix
            A_ = DPS_topk(is_CNN = True, BS=1, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D)
            A_ = tf.squeeze(A_, [0])  ### This is to match the size of w matrix, the BS dimension should be 0
            A_ = tf.reduce_sum(A_,axis=-2)
            print('A test SHAPE', np.shape(A_))
            # after returning hardSamples, they must be reshaped to their kernel dimensions
            A = tf.reshape(A_,[self.kernel_size*self.kernel_size, self.channel_size ,self.filters])
            A = tf.reshape(A_,[self.kernel_size,self.kernel_size,self.channel_size, self.filters])
            print('A SHAPE', np.shape(A))

                                
        AW =tf.multiply(A, self.k)  ## aplying masks
        print('x shape', np.shape(x))
        print('AW shape', np.shape(AW))
        y = tf.nn.conv2d(x, AW, self.strides, padding = 'SAME')
        y = tf.nn.bias_add(y,self.b)      
            

            
        if not self.activation == None:
            y = Activation(self.activation)(y)
        y = tf.nn.max_pool2d(y, ksize=2, strides=2, padding ='SAME' )
        
        return y

#



class sparseconnect_CNN_layer_filter(Layer): 
    """
    Sparse convolutional layer (filter level)
    - Generates trainable logits (D_ch)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify the number of filters
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 channel_size,
                 n_connect_ch,
                 activation=None, 
                 n_epochs=10, 
                 tempIncr=5, 
                 name=None, 
                 one_per_batch=True,
                 strides = (1,1,1,1),
                 padding = 'SAME',
                 **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.channel_size = channel_size
        self.n_connect_ch = n_connect_ch
        self.activation = activation
        self.n_epochs = n_epochs
        self.tempIncr = tempIncr
        self.one_per_batch = one_per_batch 
        self.strides = strides
        self.padding = padding

		
        super(sparseconnect_CNN_layer_filter, self).__init__(name=name, **kwargs) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({ 
            'filters':self.filters, 
            'kernel_size':self.kernel_size,
            'channel_size':self.channel_size,
            'n_connect_ch': self.n_connect_ch,
            'activation':self.activation,
            'n_epochs':self.n_epochs,
            'tempIncr':self.tempIncr,
            'strides':self.one_per_batch,
            'padding':self.padding,
            'strides':self.strides
        })
        return config

    def build(self, input_shape): 

        self.k = self.add_weight(shape=[self.kernel_size,self.kernel_size, self.channel_size, self.filters],
                              initializer='he_uniform',
                              regularizer=regularizers.l2(5e-4),
                              trainable=True,  dtype='float32')
        self.b = self.add_weight(shape=[int(self.filters)]
                                  # #,regularizer=regularizers.l2(1e-4)
                                  ,initializer='zeros',trainable=True,  dtype='float32')

        self.D_ch = self.add_weight(name='TrainableLogits_channel',
                               shape=[self.filters, self.channel_size],
                               initializer = tf.random_normal_initializer(mean=0, stddev=1, seed=None),
                               #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None),
                               regularizer = entropy_reg(1e-5),
                               trainable=True)

        super(sparseconnect_CNN_layer_filter, self).build(input_shape) 

    def call(self, x):
        filters = self.filters
        print('x shape', np.shape(x))

        if self.one_per_batch:
            batch_size = K.shape(x)[0]
            #########
            # filter prunning
            A_ch = DPS_topk(is_CNN = False, BS=1, k = self.n_connect_ch, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D_ch)
            print('A CH SHAPE', A_ch.shape)
            A_ch = tf.reduce_sum(A_ch,axis=-2)
            A_ch = tf.reshape(A_ch,[1,1,A_ch.shape[-1], A_ch.shape[-2]])
            #aux = tf.ones(shape=(y.shape[1], y.shape[2], y.shape[3]))
            ########
            
            A_ch=tf.multiply(A_ch, self.k)
            
            #########
            y = tf.nn.conv2d(x, A_ch, self.strides, padding = 'SAME')
            y = tf.nn.bias_add(y,self.b)  

            
           
 
    
        return y


class sparseconnect_CNN_layer_filter_tile_2(Layer): 
    """
    Sparse convolutional layer (filter level)
    - Generates trainable logits (D_ch)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify the number of filters
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 channel_size,
                 n_connect_ch,
                 activation=None, 
                 n_epochs=10, 
                 tempIncr=5, 
                 name=None, 
                 one_per_batch=True,
                 strides = (1,1,1,1),
                 padding = 'SAME',
                 **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.channel_size = channel_size
        self.n_connect_ch = n_connect_ch
        self.activation = activation
        self.n_epochs = n_epochs
        self.tempIncr = tempIncr
        self.one_per_batch = one_per_batch 
        self.strides = strides
        self.padding = padding

		
        super(sparseconnect_CNN_layer_filter_tile_2, self).__init__(name=name, **kwargs) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({ 
            'filters':self.filters, 
            'kernel_size':self.kernel_size,
            'channel_size':self.channel_size,
            'n_connect_ch': self.n_connect_ch,
            'activation':self.activation,
            'n_epochs':self.n_epochs,
            'tempIncr':self.tempIncr,
            'strides':self.one_per_batch,
            'padding':self.padding,
            'strides':self.strides
        })
        return config

    def build(self, input_shape): 

        self.k = self.add_weight(shape=[self.kernel_size,self.kernel_size, self.channel_size, self.filters],
                              initializer='he_uniform',
                              regularizer=regularizers.l2(5e-4),
                              trainable=True,  dtype='float32')
        self.b = self.add_weight(shape=[int(self.filters)]
                                  # #,regularizer=regularizers.l2(1e-4)
                                  ,initializer='zeros',trainable=True,  dtype='float32')

        self.D_ch_1 = self.add_weight(name='TrainableLogits_channel',
                               shape=[self.filters, int(self.channel_size/2)],
                               initializer = tf.random_normal_initializer(mean=0, stddev=1, seed=None),
                               #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None),
                               regularizer = entropy_reg(1e-5),
                               trainable=True)
                               
        self.D_ch_2 = self.add_weight(name='TrainableLogits_channel',
                               shape=[self.filters, int(self.channel_size/2)],
                               initializer = tf.random_normal_initializer(mean=0, stddev=1, seed=None),
                               #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None),
                               regularizer = entropy_reg(1e-5),
                               trainable=True)

        super(sparseconnect_CNN_layer_filter_tile_2, self).build(input_shape) 

    def call(self, x):
        filters = self.filters
        print('x shape', np.shape(x))

        if self.one_per_batch:
            batch_size = K.shape(x)[0]
            #########
            # filter prunning
            A_ch_1 = DPS_topk(is_CNN = False, BS=1, k = self.n_connect_ch, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D_ch_1)
            
            A_ch_1 = tf.reduce_sum(A_ch_1,axis=-2)
            A_ch_1 = tf.reshape(A_ch_1,[1,1,A_ch_1.shape[-1], A_ch_1.shape[-2]])
            print('A CH SHAPE', A_ch_1.shape)
            A_ch_2 = DPS_topk(is_CNN = False, BS=1, k = self.n_connect_ch, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D_ch_2)
            
            A_ch_2 = tf.reduce_sum(A_ch_2,axis=-2)
            A_ch_2 = tf.reshape(A_ch_2,[1,1,A_ch_2.shape[-1], A_ch_2.shape[-2]])
            print('A CH SHAPE', A_ch_2.shape)
            #aux = tf.ones(shape=(y.shape[1], y.shape[2], y.shape[3]))
            ########
            
            

            A_ch = tf.concat([A_ch_1, A_ch_2], 2)
            
            A_ch=tf.multiply(A_ch, self.k)
            
            #########
            y = tf.nn.conv2d(x, A_ch, self.strides, padding = 'SAME')
            y = tf.nn.bias_add(y,self.b)  

        return y
        
        

class sparseconnect_CNN_layer_filter_tile_4(Layer): 
    """
    Sparse convolutional layer (filter level)
    - Generates trainable logits (D_ch)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify the number of filters
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 channel_size,
                 n_connect_ch,
                 activation=None, 
                 n_epochs=10, 
                 tempIncr=5, 
                 name=None, 
                 one_per_batch=True,
                 strides = (1,1,1,1),
                 padding = 'SAME',
                 **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.channel_size = channel_size
        self.n_connect_ch = n_connect_ch
        self.activation = activation
        self.n_epochs = n_epochs
        self.tempIncr = tempIncr
        self.one_per_batch = one_per_batch 
        self.strides = strides
        self.padding = padding

		
        super(sparseconnect_CNN_layer_filter_tile_4, self).__init__(name=name, **kwargs) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({ 
            'filters':self.filters, 
            'kernel_size':self.kernel_size,
            'channel_size':self.channel_size,
            'n_connect_ch': self.n_connect_ch,
            'activation':self.activation,
            'n_epochs':self.n_epochs,
            'tempIncr':self.tempIncr,
            'strides':self.one_per_batch,
            'padding':self.padding,
            'strides':self.strides
        })
        return config

    def build(self, input_shape): 

        self.k = self.add_weight(shape=[self.kernel_size,self.kernel_size, self.channel_size, self.filters],
                              initializer='he_uniform',
                              regularizer=regularizers.l2(5e-4),
                              trainable=True,  dtype='float32')
        self.b = self.add_weight(shape=[int(self.filters)]
                                  # #,regularizer=regularizers.l2(1e-4)
                                  ,initializer='zeros',trainable=True,  dtype='float32')

        self.D_ch_1 = self.add_weight(name='TrainableLogits_channel',
                               shape=[self.filters, int(self.channel_size/2)],
                               initializer = tf.random_normal_initializer(mean=0, stddev=1, seed=None),
                               #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None),
                               regularizer = entropy_reg(1e-5),
                               trainable=True)
                               
        self.D_ch_2 = self.add_weight(name='TrainableLogits_channel',
                               shape=[self.filters, int(self.channel_size/2)],
                               initializer = tf.random_normal_initializer(mean=0, stddev=1, seed=None),
                               #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None),
                               regularizer = entropy_reg(1e-5),
                               trainable=True)
                               
        self.D_ch_3 = self.add_weight(name='TrainableLogits_channel',
                               shape=[self.filters, int(self.channel_size/2)],
                               initializer = tf.random_normal_initializer(mean=0, stddev=1, seed=None),
                               #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None),
                               regularizer = entropy_reg(1e-5),
                               trainable=True)
                               
        self.D_ch_4 = self.add_weight(name='TrainableLogits_channel',
                               shape=[self.filters, int(self.channel_size/2)],
                               initializer = tf.random_normal_initializer(mean=0, stddev=1, seed=None),
                               #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None),
                               regularizer = entropy_reg(1e-5),
                               trainable=True)

        super(sparseconnect_CNN_layer_filter_tile_4, self).build(input_shape) 

    def call(self, x):
        filters = self.filters
        print('x shape', np.shape(x))

        if self.one_per_batch:
            batch_size = K.shape(x)[0]
            #########
            # filter prunning
            A_ch_1 = DPS_topk(is_CNN = False, BS=1, k = self.n_connect_ch, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D_ch_1)
            A_ch_1 = tf.reduce_sum(A_ch_1,axis=-2)
            A_ch_1 = tf.reshape(A_ch_1,[1,1,A_ch_1.shape[-1], A_ch_1.shape[-2]])
            print('A CH SHAPE', A_ch_1.shape)
            
            A_ch_2 = DPS_topk(is_CNN = False, BS=1, k = self.n_connect_ch, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D_ch_2)
            A_ch_2 = tf.reduce_sum(A_ch_2,axis=-2)
            A_ch_2 = tf.reshape(A_ch_2,[1,1,A_ch_2.shape[-1], A_ch_2.shape[-2]])
            print('A CH SHAPE', A_ch_2.shape)
            
            A_ch_3 = DPS_topk(is_CNN = False, BS=1, k = self.n_connect_ch, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D_ch_3)
            A_ch_3 = tf.reduce_sum(A_ch_3,axis=-2)
            A_ch_3 = tf.reshape(A_ch_3,[1,1,A_ch_3.shape[-1], A_ch_3.shape[-2]])
            
            A_ch_4 = DPS_topk(is_CNN = False, BS=1, k = self.n_connect_ch, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D_ch_4)
            A_ch_4 = tf.reduce_sum(A_ch_4,axis=-2)
            A_ch_4 = tf.reshape(A_ch_4,[1,1,A_ch_4.shape[-1], A_ch_4.shape[-2]])
            #aux = tf.ones(shape=(y.shape[1], y.shape[2], y.shape[3]))
            ########
            
            

            A_ch = tf.concat([A_ch_1, A_ch_2, A_ch_3, A_ch_4], 2)
            
            A_ch=tf.multiply(A_ch, self.k)
            
            #########
            y = tf.nn.conv2d(x, A_ch, self.strides, padding = 'SAME')
            y = tf.nn.bias_add(y,self.b)  

        return y 

class sparseconnect_layer_tile_2(Layer):  
    
    """
    Sparse fully-connected layer
    - Generates trainable logits (D)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify W matrix
    """
    
    def __init__(self,units, n_connect, activation=None, n_epochs=10, tempIncr=5, name=None, one_per_batch=True, tiles=2, **kwargs):
        self.units = units
        self.n_connect = n_connect
        self.activation = activation
        self.n_epochs = n_epochs
        self.tempIncr = tempIncr
        self.one_per_batch = one_per_batch 
        self.tiles = tiles
        super(sparseconnect_layer_tile_2, self).__init__(name=name, **kwargs) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units':self.units, 
            'n_connect':self.n_connect, 
            'activation':self.activation,
            'n_epochs':self.n_epochs,
            'tempIncr': self.tempIncr,
            'one_per_batch':self.one_per_batch,
            'tiles':self.tiles
        })
        return config

    def build(self, input_shape): 
        # Define weight matrix and bias vector
        self.W = self.add_weight(shape=[self.units,int(input_shape[-1])],
                                 initializer='glorot_uniform',
                                 trainable=True, name='w_bin', dtype='float32')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='glorot_normal',
                                 trainable=True, name='b_bin',  dtype='float32')
        
        # Define sampling logits
        self.D1 = self.add_weight(name='TrainableLogits', 
                              shape=(self.units, int(input_shape[-1]/self.tiles)),
                              initializer = tf.random_normal_initializer(mean=0, stddev=0.5, seed=None),
                              regularizer = entropy_reg(5e-6),
                              #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None), #TODO Liz: choose which initializer might be suitable for you
                              trainable=True)
                              
        self.D2 = self.add_weight(name='TrainableLogits', 
                              shape=(self.units, int(input_shape[-1]/self.tiles)),
                              initializer = tf.random_normal_initializer(mean=0, stddev=0.5, seed=None),
                              regularizer = entropy_reg(5e-6),
                              #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None), #TODO Liz: choose which initializer might be suitable for you
                              trainable=True)

        super(sparseconnect_layer_tile_2, self).build(input_shape) 
 
    def call(self, x):
        units = self.units

        if self.one_per_batch:
            batch_size = K.shape(x)[0]
            # Produce sparse sampling matrix
            A1 = DPS_topk(is_CNN=False, BS=1, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D1)
            A1 = tf.reduce_sum(A1,axis=-2)
            A1 = tf.reduce_sum(A1,axis=0)

            A2 = DPS_topk(is_CNN=False, BS=1, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D2)
            A2 = tf.reduce_sum(A2,axis=-2)
            A2 = tf.reduce_sum(A2,axis=0) 
            print('A shape', A1.shape)
            A = tf.concat([A1, A2], 1)
            
            
            # Produce sparse weight matrix
            AW = Lambda(lambda inp: tf.multiply(inp[0],inp[1]), output_shape = (units,units))([A,self.W])  
            # Produce layer output
            y = Lambda(lambda inp: K.dot(inp[1],tf.transpose(inp[0],(1,0)))+inp[2], output_shape = (units))([AW,x,self.b])
            #y = tf.keras.layers.BatchNormalization()(y)            
        else:
            batch_size = K.shape(x)[0]
            # Produce sparse sampling matrix
            A = DPS_topk(BS=batch_size, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D)
            A = tf.reduce_sum(A,axis=-2)          
            # Produce sparse weight matrix
            AW = Lambda(lambda inp: tf.multiply(inp[0],inp[1]), output_shape = (units,units))([A,self.W])  
            # Produce layer output
            y = Lambda(lambda inp: K.batch_dot(inp[1],tf.transpose(inp[0],(0,2,1)))+inp[2], output_shape = (units))([AW,x,self.b])
        if not self.activation == None:
            y = Activation(self.activation)(y)
        
        return y

class sparseconnect_layer_tile_4(Layer):  
    
    """
    Sparse fully-connected layer
    - Generates trainable logits (D)
    - Call DPS_topK to perform optimization
    - Generates a mask based on hardSamples to sparsify W matrix
    """
    
    def __init__(self,units, n_connect, activation=None, n_epochs=10, tempIncr=5, name=None, one_per_batch=True, tiles=2, **kwargs):
        self.units = units
        self.n_connect = n_connect
        self.activation = activation
        self.n_epochs = n_epochs
        self.tempIncr = tempIncr
        self.one_per_batch = one_per_batch 
        self.tiles = tiles
        super(sparseconnect_layer_tile_2, self).__init__(name=name, **kwargs) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units':self.units, 
            'n_connect':self.n_connect, 
            'activation':self.activation,
            'n_epochs':self.n_epochs,
            'tempIncr': self.tempIncr,
            'one_per_batch':self.one_per_batch,
            'tiles':self.tiles
        })
        return config

    def build(self, input_shape): 
        # Define weight matrix and bias vector
        self.W = self.add_weight(shape=[self.units,int(input_shape[-1])],
                                 initializer='glorot_uniform',
                                 trainable=True, name='w_bin', dtype='float32')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='glorot_normal',
                                 trainable=True, name='b_bin',  dtype='float32')
        
        # Define sampling logits
        self.D1 = self.add_weight(name='TrainableLogits', 
                              shape=(self.units, int(input_shape[-1]/self.tiles)),
                              initializer = tf.random_normal_initializer(mean=0, stddev=0.5, seed=None),
                              regularizer = entropy_reg(5e-6),
                              #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None), #TODO Liz: choose which initializer might be suitable for you
                              trainable=True)
                              
        self.D2 = self.add_weight(name='TrainableLogits', 
                              shape=(self.units, int(input_shape[-1]/self.tiles)),
                              initializer = tf.random_normal_initializer(mean=0, stddev=0.5, seed=None),
                              regularizer = entropy_reg(5e-6),
                              #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None), #TODO Liz: choose which initializer might be suitable for you
                              trainable=True)
                             
        self.D3 = self.add_weight(name='TrainableLogits', 
                              shape=(self.units, int(input_shape[-1]/self.tiles)),
                              initializer = tf.random_normal_initializer(mean=0, stddev=0.5, seed=None),
                              regularizer = entropy_reg(5e-6),
                              #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None), #TODO Liz: choose which initializer might be suitable for you
                              trainable=True)
                              
        self.D4 = self.add_weight(name='TrainableLogits', 
                              shape=(self.units, int(input_shape[-1]/self.tiles)),
                              initializer = tf.random_normal_initializer(mean=0, stddev=0.5, seed=None),
                              regularizer = entropy_reg(5e-6),
                              #initializer = tf.initializers.RandomNormal(minval=-1.0, maxval=1.0, seed=None), #TODO Liz: choose which initializer might be suitable for you
                              trainable=True)

        super(sparseconnect_layer_tile_2, self).build(input_shape) 
 
    def call(self, x):
        units = self.units

        if self.one_per_batch:
            batch_size = K.shape(x)[0]
            # Produce sparse sampling matrix
            A1 = DPS_topk(is_CNN=False, BS=1, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D1)
            A1 = tf.reduce_sum(A1,axis=-2)
            A1 = tf.reduce_sum(A1,axis=0)

            A2 = DPS_topk(is_CNN=False, BS=1, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D2)
            A2 = tf.reduce_sum(A2,axis=-2)
            A2 = tf.reduce_sum(A2,axis=0) 
            
            A3 = DPS_topk(is_CNN=False, BS=1, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D3)
            A3 = tf.reduce_sum(A3,axis=-2)
            A3 = tf.reduce_sum(A3,axis=0)


            A4 = DPS_topk(is_CNN=False, BS=1, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D4)
            A4 = tf.reduce_sum(A4,axis=-2)
            A4 = tf.reduce_sum(A4,axis=0) 
            
            print('A shape', A1.shape)
            A = tf.concat([A1, A2, A3, A4], 1)
            
            
            # Produce sparse weight matrix
            AW = Lambda(lambda inp: tf.multiply(inp[0],inp[1]), output_shape = (units,units))([A,self.W])  
            # Produce layer output
            y = Lambda(lambda inp: K.dot(inp[1],tf.transpose(inp[0],(1,0)))+inp[2], output_shape = (units))([AW,x,self.b])
            #y = tf.keras.layers.BatchNormalization()(y)            
        else:
            batch_size = K.shape(x)[0]
            # Produce sparse sampling matrix
            A = DPS_topk(BS=batch_size, k = self.n_connect, n_epochs=self.n_epochs,  tempIncr=self.tempIncr)(self.D)
            A = tf.reduce_sum(A,axis=-2)          
            # Produce sparse weight matrix
            AW = Lambda(lambda inp: tf.multiply(inp[0],inp[1]), output_shape = (units,units))([A,self.W])  
            # Produce layer output
            y = Lambda(lambda inp: K.batch_dot(inp[1],tf.transpose(inp[0],(0,2,1)))+inp[2], output_shape = (units))([AW,x,self.b])
        if not self.activation == None:
            y = Activation(self.activation)(y)
        
        return y
