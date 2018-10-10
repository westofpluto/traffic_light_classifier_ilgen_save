###############################################################################
# train_site.py
# This file trains a squeezenet neural net to detect traffic light state in 
# a single shot. The dataset used is site data
###############################################################################
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import (
        Input, Dense, Dropout, Convolution2D, Flatten,
        Add, Activation, Concatenate, Conv2D, 
        GlobalAveragePooling2D, MaxPooling2D
)
import keras.backend as K

from glob import glob
import os
import random
import cv2
import yaml
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from keras.models import model_from_yaml

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
SITE_DATA_DIR = os.path.join(DATA_ROOT, 'real_carla_data_aug')

DIR_GREEN = os.path.join(SITE_DATA_DIR,'Green')
DIR_YELLOW = os.path.join(SITE_DATA_DIR,'Yellow')
DIR_RED = os.path.join(SITE_DATA_DIR,'Red')
DIR_UNKNOWN = os.path.join(SITE_DATA_DIR,'Unknown')

#
# The data is stored in Green, Yellow, Red, and Unknown folders.
# The images I train on are all jpg files, while the eval images are all png files
#
train_file_paths_green = glob(os.path.join(DIR_GREEN, '*.jpg'))
train_file_paths_yellow = glob(os.path.join(DIR_YELLOW, '*.jpg'))
train_file_paths_red = glob(os.path.join(DIR_RED, '*.jpg'))
train_file_paths_none = glob(os.path.join(DIR_UNKNOWN, '*.jpg'))

train2_file_paths_green = glob(os.path.join(DIR_GREEN, '*.png'))
train2_file_paths_yellow = glob(os.path.join(DIR_YELLOW, '*.png'))
train2_file_paths_red = glob(os.path.join(DIR_RED, '*.png'))
train2_file_paths_none = glob(os.path.join(DIR_UNKNOWN, '*.png'))

print('Simulator training images Set 1 - Green: {}, Yellow: {}, Red: {}, None: {}'.format(
    len(train_file_paths_green), len(train_file_paths_yellow), len(train_file_paths_red), len(train_file_paths_none)))
print('Simulator training images Set 2 - Green: {}, Yellow: {}, Red: {}, None: {}'.format(
    len(train2_file_paths_green), len(train2_file_paths_yellow), len(train2_file_paths_red), len(train2_file_paths_none)))

#
# set image paths  
#
img_paths.extend(train_file_paths_green)
img_paths.extend(train_file_paths_yellow)
img_paths.extend(train_file_paths_red)
img_paths.extend(train_file_paths_none)

img_paths.extend(train2_file_paths_green)
img_paths.extend(train2_file_paths_yellow)
img_paths.extend(train2_file_paths_red)
img_paths.extend(train2_file_paths_none)
#
# Now handle the labels
#
labels.extend([[1.0, 0.0, 0.0, 0.0] for i in range(len(train_file_paths_green))])
labels.extend([[0.0, 1.0, 0.0, 0.0] for i in range(len(train_file_paths_yellow))])
labels.extend([[0.0, 0.0, 1.0, 0.0] for i in range(len(train_file_paths_red))])
labels.extend([[0.0, 0.0, 0.0, 1.0] for i in range(len(train_file_paths_none))])

labels.extend([[1.0, 0.0, 0.0, 0.0] for i in range(len(train2_file_paths_green))])
labels.extend([[0.0, 1.0, 0.0, 0.0] for i in range(len(train2_file_paths_yellow))])
labels.extend([[0.0, 0.0, 1.0, 0.0] for i in range(len(train2_file_paths_red))])
labels.extend([[0.0, 0.0, 0.0, 1.0] for i in range(len(train2_file_paths_none))])

print('No. images: {}, No. Labels: {}'.format(len(img_paths), len(labels)))

#
# shuffle the data
#
img_paths, labels = shuffle(img_paths, labels)
print('Total - imgs: {}, labels: {}'.format(len(img_paths), len(labels)))
img_paths_train, img_paths_test, labels_train, labels_test = train_test_split(img_paths, labels, test_size=0.2)

print('Train - imgs: {}, labels: {}'.format(len(img_paths_train), len(labels_train)))
print('Test - imgs: {}, labels: {}'.format(len(img_paths_test), len(labels_test)))

#
# Now set up the Squeezenet classifier. 
#
NO_CLASSES = 4
# From https://github.com/cmasch/squeezenet/blob/master/squeezenet.py

def SqueezeNet_11(input_shape, nb_classes, dropout_rate=None, compression=1.0):
    """
    Creating a SqueezeNet of version 1.1
    
    2.4x less computation over SqueezeNet 1.0 implemented above.
    
    Arguments:
        input_shape  : shape of the input images e.g. (224,224,3)
        nb_classes   : number of classes
        dropout_rate : defines the dropout rate that is accomplished after last fire module (default: None)
        compression  : reduce the number of feature-maps
        
    Returns:
        Model        : Keras model instance
    """
    
    input_img = Input(shape=input_shape)

    x = Conv2D(int(64*compression), (3,3), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
    x = create_fire_module(x, int(16*compression), name='fire2')
    x = create_fire_module(x, int(16*compression), name='fire3')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool3')(x)
    
    x = create_fire_module(x, int(32*compression), name='fire4')
    x = create_fire_module(x, int(32*compression), name='fire5')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool5')(x)
    
    x = create_fire_module(x, int(48*compression), name='fire6')
    x = create_fire_module(x, int(48*compression), name='fire7')
    x = create_fire_module(x, int(64*compression), name='fire8')
    x = create_fire_module(x, int(64*compression), name='fire9')

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    # Creating last conv10
    x = output(x, nb_classes)

    return Model(inputs=input_img, outputs=x)


def output(x, nb_classes):
    x = Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv10')(x)
    x = GlobalAveragePooling2D(name='avgpool10')(x)
    x = Activation("softmax", name='softmax')(x)
    return x


def create_fire_module(x, nb_squeeze_filter, name, use_bypass=False):
    """
    Creates a fire module
    
    Arguments:
        x                 : input
        nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
        use_bypass        : if True then a bypass will be added
        name              : name of module e.g. fire123
    
    Returns:
        x                 : returns a fire module
    """
    nb_expand_filter = 4 * nb_squeeze_filter
    squeeze    = Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
    expand_1x1 = Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
    expand_3x3 = Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
    
    axis = get_axis()
    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
    
    if use_bypass:
        x_ret = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
        
    return x_ret


def get_axis():
    axis = -1 if K.image_data_format() == 'channels_last' else 1
    return axis

#
# Now actucally create the Squeezenet classifier model
#
classifier_model = SqueezeNet_11((224, 224, 3), 4, dropout_rate=0.5, compression=2.0)
classifier_model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
print(classifier_model.summary())

#
# data sample generator function, used to get and resize images
#
def generate_sample(batch_size, img_paths_sample, labels_sample):
    while True:
        for batch_i in range(0, len(img_paths_sample), batch_size):
            imgs = []
            lbs = labels_sample[batch_i:batch_i+batch_size]
            for img_path in img_paths_sample[batch_i:batch_i+batch_size]:
                imgs.append(cv2.resize(cv2.imread(img_path), (224, 224)))
            yield np.array(imgs), np.array(lbs)

#
# Now set up and train classifier
#
BATCH_SIZE = 128
STEEPS_PER_EPOCH = len(img_paths_train) / BATCH_SIZE
TEST_STEEPS_PER_EPOCH = len(img_paths_test) / BATCH_SIZE

def train_classifier(epochs):

    classifier_model.fit_generator(
        generate_sample(BATCH_SIZE, img_paths_train, labels_train),
        steps_per_epoch=STEEPS_PER_EPOCH, 
        epochs=epochs,
        validation_data = generate_sample(BATCH_SIZE, img_paths_test, labels_test),
        validation_steps = TEST_STEEPS_PER_EPOCH)

############################################
# load existing model to keep training
############################################
try:
    yaml_file = 'models/site/classifier_model.yaml'
    weights_file = 'models/site/classifier_model_weights.h5'
    model_yaml_txt = readFile(yaml_file)
    classifier_model = model_from_yaml(model_yaml_txt)
    classifier_model.load_weights(weights_file)
    classifier_model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
except Exception as ex:
    print("Exception: %s" % ex)


######################
# training runs here
######################
train_classifier(100)


######################
# test here
######################
i = (int)(random.random() * len(img_paths))
img_path = img_paths[i]
print('Image Path: {}'.format(img_path))

img = cv2.resize(cv2.imread(img_path), (224, 224))
label = labels[i]

print('Image Shape: {}'.format(img.shape))
print('Image Label: {}'.format(label))
print('Image Class: {}'.format(
    'green' if label[0] else ('yellow' if label[1] else ('red' if label[2] else 'none'))))

pred = classifier_model.predict(img.reshape(1,224,224,3))[0]
print('Prediction: {}'.format(pred))
img_cls = np.argmax(pred)
print('Predicted Class: {}'.format(
    'green' if img_cls==0 else ('yellow' if img_cls==1 else ('red' if img_cls==2 else 'none'))))

#########################
# Finally, save model
#########################
CLASSIFIER_MODEL_WEIGHTS_FILE = os.path.join('models', 'site', 'classifier_model_weights.h5')
CLASSIFIER_MODEL_YAML_FILE = os.path.join('models', 'site', 'classifier_model.yaml')
# Saving the weights
classifier_model.save_weights(CLASSIFIER_MODEL_WEIGHTS_FILE)

# Saving the architecture
classifier_model_yaml = classifier_model.to_yaml()
with open(CLASSIFIER_MODEL_YAML_FILE, "w") as classifier_yaml_file:
    classifier_yaml_file.write(classifier_model_yaml)

print("Done!")



