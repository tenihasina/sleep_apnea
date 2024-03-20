import math
import numpy as np
import torch
import os
import datetime
import tensorflow as tf
import keras.callbacks as Callback
import keras_tuner as kt
# example of import to start creating your model
from keras_applications.resnet_1d import ResNet18
from segmentation_models.segmentation_models.models.unet_1d import Unet1D

checkpoint_dir = os.path.join("physionet.org/checkpoint_dir", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
logdir = os.path.join("physionet.org/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard = tf.keras.callbacks.TensorBoard(
    logdir, histogram_freq=1
    )
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}", save_freq=128
    )
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss', factor=0.2, patience=5, min_lr=0.001
#     )
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=3
    )

callbacks = [tensorboard, checkpoint, early_stop]

def build_model(hp):

  model = Unet1D(backbone_name = "resnet18_1d", classes = 1, input_shape = (batch_train.shape[1], batch_train.shape[2]))
  model.compile(
    optimizer = hp.Choice("loss", ["adam", "adamax", "SGD", "rmsprop"]),
    loss = hp.Choice("optimizer", values=["binary_crossentropy", "binary_focal_crossentropy"]),
    metrics=['accuracy']    
  )
  return model



def tuner_fit_predict(tuner, saving_path, x_train, y_train):
    
    # Get the top 2 hyperparameters.
    best_hps = tuner.get_best_hyperparameters(5)
    # Build the model with the best hp.
    model = build_model(best_hps[0])
    model.fit(x=x_train, y=y_train, epochs=10, callbacks = callbacks)
    # Save model
    model.save(saving_path)

    # Return a single float as the objective value.
    # You may also return a dictionary
    # of {metric_name: metric_value}.
    

    return None