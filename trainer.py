from dataloader import *
from model import deepEdge

import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import os

#Instantiate the model
model = deepEdge()

version = 1 #Version number
model_name = 'deepEdge_best_val_loss_v1.hdf5'

start_from_scratch = False


if not start_from_scratch:
    if os.path.exists(model_name):
        print("Model Exists! Resuming fine tuning")
        try:
            model.load_weights('./deepEdge_best_val_loss_v1.hdf5')
        except:
            print("Error Loading model: Can happen in the event of a change in model arch/file getting corrupt!")
            print("Starting from scratch")
    else:
        print("Given version model does not exists, starting from scratch")

num_training_samples = 240
batch_size = 10
steps_per_epoch = 40
num_epochs = 1000

checkpoint_filepath = './'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True)

model.fit(datastreamer_BIPED(), steps_per_epoch=steps_per_epoch,epochs=num_epochs,validation_data = datastreamer_BIPED(mode = 'test'),validation_steps = 1, callbacks = [model_checkpoint])


model.save('deepEdge_end_of_training.h5', model)
