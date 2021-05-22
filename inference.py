from dataloader import *
from model import deepEdge
import numpy as np
import cv2
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import os
import matplotlib.pyplot as plt

from model import deepEdge

DE = deepEdge()
DE.load_weights('deepEdge_best_val_loss_v1.hdf5')

print(DE.summary())

inference_edges_src = './BIPED/edges/imgs/test/rgbr/'
inference_edges_dst = './DE_inference/'

if not os.path.exists(inference_edges_dst):
    os.mkdir(inference_edges_dst)

for filename in os.listdir(inference_edges_src):
    print(filename)
    try:
    #if 1:
        img = plt.imread(inference_edges_src + filename)[:,:,:3] #avoid 4 channel input

    except:
    #else:
        print("Error reading file! Continuing...")
        continue

    #make sure to normalise the img
    img_max = np.max(img)
    if img_max > 0: #To avoid division by zero error
        img = img/img_max

    img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_CUBIC)

    #Note: matplotlib automatically takes in a normalised img while read

    #predict
    edges_pred = np.squeeze(DE(np.reshape(img, (1,) + img.shape)))

    plt.imsave(inference_edges_dst + filename, edges_pred)
