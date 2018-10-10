###############################################################################
# test_net.py
# This file trains a squeezenet neural net to detect traffic light state in 
# a single shot. 
###############################################################################
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import (
        Input, Dense, Dropout, Convolution2D, Flatten,
        Add, Activation, Concatenate, Conv2D, 
        GlobalAveragePooling2D, MaxPooling2D
)
import keras.backend as K
from keras.models import model_from_yaml

from glob import glob
import os
import random
import cv2
import yaml
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import tensorflow as tf

def readFile(filename):
    handle=open(filename,'r')
    txt=handle.read()
    handle.close()
    return txt
#
# initialize arrays that contain image files and labels
#
img_paths = []
labels = []

#DATA_ROOT = "/home/ubuntu/work/projects/term3/project3-capstone/ilgen-classifier/data"
DATA_ROOT = "../data"
SIM_DATA_DIR = os.path.join(DATA_ROOT, 'simulator_dataset_rgb')
SITE_DATA_DIR = os.path.join(DATA_ROOT, 'udacity_testarea_rgb')

SIM_DIR_GREEN = os.path.join(SIM_DATA_DIR,'Green')
SIM_DIR_YELLOW = os.path.join(SIM_DATA_DIR,'Yellow')
SIM_DIR_RED = os.path.join(SIM_DATA_DIR,'Red')
SIM_DIR_UNKNOWN = os.path.join(SIM_DATA_DIR,'Unknown')

#
# The data is stored in Green, Yellow, Red, and Unknown folders.
# The images I train on are all jpg files, while the eval images are all png files
#
sim_train_file_paths_green = glob(os.path.join(SIM_DIR_GREEN, 'left*.jpg'))
sim_train_file_paths_yellow = glob(os.path.join(SIM_DIR_YELLOW, 'left*.jpg'))
sim_train_file_paths_red = glob(os.path.join(SIM_DIR_RED, 'left*.jpg'))
sim_train_file_paths_none = glob(os.path.join(SIM_DIR_UNKNOWN, 'left*.jpg'))

sim_train2_file_paths_green = glob(os.path.join(SIM_DIR_GREEN, '*.png'))
sim_train2_file_paths_yellow = glob(os.path.join(SIM_DIR_YELLOW, '*.png'))
sim_train2_file_paths_red = glob(os.path.join(SIM_DIR_RED, '*.png'))
sim_train2_file_paths_none = glob(os.path.join(SIM_DIR_UNKNOWN, '*.png'))

print('Simulator training images Set 1 - Green: {}, Yellow: {}, Red: {}, None: {}'.format(
    len(sim_train_file_paths_green), len(sim_train_file_paths_yellow), len(sim_train_file_paths_red), len(sim_train_file_paths_none)))
print('Simulator training images Set 2 - Green: {}, Yellow: {}, Red: {}, None: {}'.format(
    len(sim_train2_file_paths_green), len(sim_train2_file_paths_yellow), len(sim_train2_file_paths_red), len(sim_train2_file_paths_none)))

#
# set image paths  
#
img_paths.extend(sim_train_file_paths_green)
img_paths.extend(sim_train_file_paths_yellow)
img_paths.extend(sim_train_file_paths_red)
img_paths.extend(sim_train_file_paths_none)

img_paths.extend(sim_train2_file_paths_green)
img_paths.extend(sim_train2_file_paths_yellow)
img_paths.extend(sim_train2_file_paths_red)
img_paths.extend(sim_train2_file_paths_none)
#
# Now handle the labels
#
labels.extend([[1.0, 0.0, 0.0, 0.0] for i in range(len(sim_train_file_paths_green))])
labels.extend([[0.0, 1.0, 0.0, 0.0] for i in range(len(sim_train_file_paths_yellow))])
labels.extend([[0.0, 0.0, 1.0, 0.0] for i in range(len(sim_train_file_paths_red))])
labels.extend([[0.0, 0.0, 0.0, 1.0] for i in range(len(sim_train_file_paths_none))])

labels.extend([[1.0, 0.0, 0.0, 0.0] for i in range(len(sim_train2_file_paths_green))])
labels.extend([[0.0, 1.0, 0.0, 0.0] for i in range(len(sim_train2_file_paths_yellow))])
labels.extend([[0.0, 0.0, 1.0, 0.0] for i in range(len(sim_train2_file_paths_red))])
labels.extend([[0.0, 0.0, 0.0, 1.0] for i in range(len(sim_train2_file_paths_none))])

print('No. images: {}, No. Labels: {}'.format(len(img_paths), len(labels)))

#
# shuffle the data
#
img_paths, labels = shuffle(img_paths, labels)
print('Total - imgs: {}, labels: {}'.format(len(img_paths), len(labels)))
img_paths_train, img_paths_test, labels_train, labels_test = train_test_split(img_paths, labels, test_size=0.2)

print('Train - imgs: {}, labels: {}'.format(len(img_paths_train), len(labels_train)))
print('Test - imgs: {}, labels: {}'.format(len(img_paths_test), len(labels_test)))

##########################
# read classifier model
##########################
yaml_file = 'models/sim/classifier_model.yaml'
weights_file = 'models/sim/classifier_model_weights.h5'
model_yaml_txt = readFile(yaml_file)
classifier_model = model_from_yaml(model_yaml_txt)
classifier_model.load_weights(weights_file)
graph = tf.get_default_graph()

######################
# test here
######################

NUM_TEST = 100
with graph.as_default():
    num=0
    num_correct=0
    for k in range(NUM_TEST):
        i = (int)(random.random() * len(img_paths))
        img_path = img_paths[i]
        print('Image Path: {}'.format(img_path))

        img = cv2.resize(cv2.imread(img_path), (224, 224))
        label = labels[i]
        label_cls = np.argmax(label)
        print('Image Shape: {}'.format(img.shape))
        print('Image Label: {}'.format(label))
        print('Image Class: {}'.format(
            'green' if label[0] else ('yellow' if label[1] else ('red' if label[2] else 'none'))))

        pred = classifier_model.predict(img.reshape(1,224,224,3))[0]
        print('Prediction: {}'.format(pred))
        img_cls = np.argmax(pred)
        print('Predicted Class: {}'.format(
            'green' if img_cls==0 else ('yellow' if img_cls==1 else ('red' if img_cls==2 else 'none'))))
        if label_cls == img_cls:
            num_correct += 1
            print("CORRECT!")
        else:
            print("WRONG")

        num += 1
        print("************************")

pct_correct = (100.0*num_correct)/num

print("Accuracy: %f percent" % pct_correct)
print("Done!")



