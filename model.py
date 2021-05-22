import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import backend as K

from losses import *



#Handmade - PSPNet
def deepEdge(input_size = (512,512,3), training = True):
    #conv nets for unsupervised feature extraction
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    #Pooling Module
    #We will use pooling sizes = 2,4,8,16,32

    #Bin1 pooling size = 2
    bin1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    flat_bin1 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(bin1)

    #Bin2 pooling size = 4
    bin2 = MaxPooling2D(pool_size=(4, 4))(conv1)
    flat_bin2 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(bin2)

    #Bin3 pooling size = 8
    bin3 = MaxPooling2D(pool_size=(8, 8))(conv1)
    flat_bin3 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(bin3)

    #Bin4 pooling size = 16
    bin4 = MaxPooling2D(pool_size=(16, 16))(conv1)
    flat_bin4 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(bin4)

    #Bin5 pooling size = 32
    bin5 = MaxPooling2D(pool_size=(32, 32))(conv1)
    flat_bin5 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(bin5)


    #Upsample each flattened bins to match the size equal to the original input size

    up_bin1 = UpSampling2D(size = (2,2))(flat_bin1)
    up_bin2 = UpSampling2D(size = (4,4))(flat_bin2)
    up_bin3 = UpSampling2D(size = (8,8))(flat_bin3)
    up_bin4 = UpSampling2D(size = (16,16))(flat_bin4)
    up_bin5 = UpSampling2D(size = (32,32))(flat_bin5)

    #Merge upsampled layers together one after the other along with the unsupervised feature extraction layer
    merged_layers = concatenate([conv1,up_bin1], axis = 3)
    merged_layers = concatenate([merged_layers,up_bin2], axis = 3)
    merged_layers = concatenate([merged_layers,up_bin3], axis = 3)
    merged_layers = concatenate([merged_layers,up_bin4], axis = 3)
    merged_layers = concatenate([merged_layers,up_bin5], axis = 3)

    #Finally generate the binary mask
    binary_masks = Conv2D(1, 1, activation = 'sigmoid')(merged_layers)

    model = Model(inputs = inputs, outputs = binary_masks)

    model.compile(optimizer = Adam(lr = 1e-7), loss = dice_coef_loss, metrics = ['accuracy'])

    return model
