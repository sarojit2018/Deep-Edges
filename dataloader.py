import numpy as np
import cv2
from random import random
from random import shuffle
from random import randint
from random import uniform
import os
import matplotlib.pyplot as plt

def datastreamer_BIPED(batch_size = 10 , target_shape = (512,512), mode = 'train', base_path = '/Users/sarojitauddya/edges_detect/BIPED/edges/', debug = False):
    #print(mode)
    #print("Entered")
    #print(debug)
    target_x, target_y = target_shape

    edge_base_path = base_path + 'edge_maps/'
    imgs_base_path = base_path + 'imgs/'

    edge_path = ''
    imgs_path = ''
    if mode=='train':
        edge_path = edge_base_path + 'train/rgbr/real/'
        imgs_path = imgs_base_path + 'train/rgbr/real/'

    elif mode=='test':
        edge_path = edge_base_path + 'test/rgbr/'
        imgs_path = imgs_base_path + 'test/rgbr/'

    list_edge = os.listdir(edge_path)
    list_imgs = os.listdir(imgs_path)

    #print(list_edge)
    #print(list_imgs)

    num_images = 0
    while 1:
        shuffle(list_edge)
        imgs_batch = []
        edge_batch = []

        for filename in list_edge:
            if filename[-3:]!='png' and filename[-3:]!='jpg':
                continue
            if num_images == batch_size:
                num_images = 0
                imgs_batch = np.array(imgs_batch)
                #print(type(edge_batch))
                edge_batch = np.array(edge_batch)

                if imgs_batch.shape[0] == 0:
                    pass

                else:
                    yield imgs_batch, edge_batch


                imgs_batch = []
                edge_batch = []

            image_path = imgs_path + filename[:-3] + 'jpg'
            edges_path = edge_path + filename

            image = plt.imread(image_path)
            edges = plt.imread(edges_path)

            if debug:
                print("Max Image pixel: "+str(np.max(image)))
                print("Min Image pixel: "+str(np.min(image)))


            patch_image = np.zeros((target_x,target_y,3))
            patch_edges = np.zeros((target_x,target_y,1))

            size_x, size_y,_ = image.shape

            edges = np.reshape(edges, (size_x, size_y, 1))

            if size_x <= target_x or size_y <= target_y: #May not be the case for BIPED dataset
                print("Oddity in the BIPED dataset")
                patch_image = cv2.resize(image, (target_x, target_y), interpolation = cv2.INTER_CUBIC)
                patch_edges = cv2.resize(edges, (target_x, target_y), interpolation = cv2.INTER_CUBIC)

            #Generally the size of images in BIPED dataset is 720 by 1280
            #Random patch based training
            start_x, start_y = randint(0, size_x - target_x), randint(0, size_y - target_y)
            patch_image = image[start_x:start_x + target_x, start_y:start_y + target_y,:]
            patch_edges = edges[start_x:start_x + target_x, start_y:start_y + target_y,:]

            #Random rotations (0,90,180,270)
            #Randomly rotate/flip
            rn_un = uniform(0,1)
            cv2_object = None

            if rn_un <= 0.25:
                cv2_object = cv2.ROTATE_90_COUNTERCLOCKWISE #counterclockwise 90

            elif rn_un  > 0.25 and rn_un<=0.5:
                cv2_object = cv2.ROTATE_90_CLOCKWISE #clockwise 90

            elif rn_un >0.5 and rn_un<=0.75:
                cv2_object = cv2.ROTATE_180 #flip

            else:
                cv2_object = None

            if False:
                patch_image = cv2.rotate(patch_image, cv2_object)
                patch_edges = cv2.rotate(patch_edges, cv2_object)


            #Colour based augmentation

            #Randomly choose channels
            ch1 = randint(0,2)
            ch2 = randint(0,2)
            ch3 = randint(0,2)

            if debug:
                print("*****Chosen channels*****")
                print(ch1, ch2, ch3)
                print("*************************")


            patch_image_colour_aug = np.zeros(patch_image.shape)
            patch_image_colour_aug[:,:,0] = patch_image[:,:,ch1]
            patch_image_colour_aug[:,:,1] = patch_image[:,:,ch2]
            patch_image_colour_aug[:,:,2] = patch_image[:,:,ch3]

            patch_image = np.uint8(patch_image_colour_aug)

            #Thicken edges
            kernel = np.ones((2,2))
            patch_edges = cv2.dilate(patch_edges, kernel, iterations = 2)


            if debug:
                plt.imshow(patch_image)
                plt.title("Patch Image")
                plt.show()
                plt.imshow(patch_edges)
                plt.title("Patch Edge")
                plt.show()

            patch_image = np.float32(patch_image)
            patch_edges = np.float32(patch_edges)

            imgs_batch.append(patch_image/255)
            edge_batch.append(patch_edges/np.max(patch_edges))

            num_images += 1
