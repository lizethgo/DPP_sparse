# -*- coding: utf-8 -*-
"""
#################################################################################
    Paper ID   : 12076
    Title      : Dynamic Probabilistic Pruning: Training sparse networks based on stochastic and dynamic masking
#################################################################################
    Source Name    : callbacks.py
    Description    : This file containsthe callbacks to visualize validation 
                     loss and accuracy plots (training_vis).
.
#################################################################################
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class training_vis(tf.keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))


    def on_train_end(self, epoch, logs={}):

        N = np.arange(0, len(self.losses))

         # Plot train loss, train acc, val loss and val acc against epochs passed
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.losses, label = "train_loss")
        axs[0].plot(self.val_losses, label = "val_loss")
        axs[0].set_title("Training: Validation Loss and Accuracy".format(epoch))
        axs[0].set_xlabel("Epoch #")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        
        axs[1].plot(self.acc, label = "train_acc")
        axs[1].plot(self.val_acc, label = "val_acc")
        axs[1].set_xlabel("Epoch #")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        plt.show()
        






