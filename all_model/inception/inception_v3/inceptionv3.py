import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import keras
import numpy as np
import math

from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Dropout,BatchNormalization,Activation, Convolution2D
from keras.models import Model
from keras.layers import Input, concatenate
from keras import optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 200
iterations         = 782
USE_BN=True
LRN2D_NORM = True
DROPOUT=0.2
CONCAT_AXIS=3
weight_decay=1e-4
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'

log_filepath  = './inception_v3'

def save_txt(file_name='',object=[]):
    file = open(file_name, 'w')
    file.write(str(object))
    file.close()

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test

def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 200:
        return 0.001
    return 0.0001

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)


def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
    x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias,
                      init="he_normal", dim_ordering='tf', W_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x


def inception_module1(x, params, concat_axis, padding='same', data_format=DATA_FORMAT, use_bias=True,
                      kernel_initializer="he_normal", bias_initializer='zeros', kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                      lrn2d_norm=LRN2D_NORM, weight_decay=weight_decay):
    (branch1, branch2, branch3, branch4) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    # 1x1
    pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
    # 1x1->3x3
    pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
    pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway2)
    pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
    # 1x1->3x3+3x3
    pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
    pathway3 = Conv2D(filters=branch3[1], kernel_size=(3, 3), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway3)
    pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
    pathway3 = Conv2D(filters=branch3[1], kernel_size=(3, 3), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway3)
    pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
    # 3x3->1x1
    pathway4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding=padding, data_format=DATA_FORMAT)(x)
    pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway4)
    pathway4 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway4))
    return concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)


def inception_reduce1(x, params, concat_axis, padding='same', data_format=DATA_FORMAT, use_bias=True,
                      kernel_initializer="he_normal", bias_initializer='zeros', kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                      lrn2d_norm=LRN2D_NORM, weight_decay=weight_decay):
    (branch1, branch2) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    # 1x1
    pathway1 = Conv2D(filters=branch1[0], kernel_size=(3, 3), strides=2, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
    # 1x1->3x3+3x3
    pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
    pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway2)
    pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
    pathway2 = Conv2D(filters=branch2[1], kernel_size=(3, 3), strides=2, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway2)
    pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
    # 3x3->1x1
    pathway3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding=padding, data_format=DATA_FORMAT)(x)
    return concatenate([pathway1, pathway2, pathway3], axis=concat_axis)


def inception_module2(x, params, concat_axis, padding='same', data_format=DATA_FORMAT, use_bias=True,
                      kernel_initializer="he_normal", bias_initializer='zeros', kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                      lrn2d_norm=LRN2D_NORM, weight_decay=weight_decay):
    (branch1, branch2, branch3, branch4) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    # 1x1
    pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
    # 1x1->1x7->7x1
    pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
    pathway2 = conv_block(pathway2, branch2[1], 1, 7)
    pathway2 = conv_block(pathway2, branch2[2], 7, 1)
    # 1x1->7x1->1x7->7x1->1x7
    pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
    pathway3 = conv_block(pathway3, branch3[1], 7, 1)
    pathway3 = conv_block(pathway3, branch3[2], 1, 7)
    pathway3 = conv_block(pathway3, branch3[3], 7, 1)
    pathway3 = conv_block(pathway3, branch3[4], 1, 7)
    # 3x3->1x1
    pathway4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding=padding, data_format=DATA_FORMAT)(x)
    pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway4)
    pathway4 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway4))
    return concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)


def inception_reduce2(x, params, concat_axis, padding='same', data_format=DATA_FORMAT, use_bias=True,
                      kernel_initializer="he_normal", bias_initializer='zeros', kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                      lrn2d_norm=LRN2D_NORM, weight_decay=weight_decay):
    (branch1, branch2) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    # 1x1->3x3
    pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
    pathway1 = Conv2D(filters=branch1[1], kernel_size=(3, 3), strides=2, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway1)
    pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
    # 1x1->1x7->7x1->3x3
    pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
    pathway2 = conv_block(pathway2, branch2[1], 1, 7)
    pathway2 = conv_block(pathway2, branch2[2], 7, 1)
    pathway2 = Conv2D(filters=branch2[3], kernel_size=(3, 3), strides=2, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway2)
    pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
    # 3x3->1x1
    pathway3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding=padding, data_format=DATA_FORMAT)(x)
    return concatenate([pathway1, pathway2, pathway3], axis=concat_axis)


def inception_module3(x, params, concat_axis, padding='same', data_format=DATA_FORMAT, use_bias=True,
                      kernel_initializer="he_normal", bias_initializer='zeros', kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                      lrn2d_norm=LRN2D_NORM, weight_decay=weight_decay):
    (branch1, branch2, branch3, branch4) = params
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    # 1x1
    pathway1 = Conv2D(filters=branch1[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway1))
    # 1x1->1x3+3x1
    pathway2 = Conv2D(filters=branch2[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway2))
    pathway2_1 = conv_block(pathway2, branch2[1], 1, 3)
    pathway2_2 = conv_block(pathway2, branch2[2], 3, 1)

    # 1x1->3x3->1x3+3x1
    pathway3 = Conv2D(filters=branch3[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(x)
    pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
    pathway3 = Conv2D(filters=branch3[1], kernel_size=(3, 3), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway3)
    pathway3 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway3))
    pathway3_1 = conv_block(pathway3, branch3[2], 1, 3)
    pathway3_2 = conv_block(pathway3, branch3[3], 3, 1)
    # 3x3->1x1
    pathway4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding=padding, data_format=DATA_FORMAT)(x)
    pathway4 = Conv2D(filters=branch4[0], kernel_size=(1, 1), strides=1, padding=padding, data_format=data_format,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)(pathway4)
    pathway4 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(pathway4))
    return concatenate([pathway1, pathway2_1, pathway2_2, pathway3_1, pathway3_2, pathway4], axis=concat_axis)


# def create_model(img_input):
#     x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same',
#                kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(img_input)
#     x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
#     x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
#                kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
#     x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
#                kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
#     x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=DATA_FORMAT)(x)
#     x = Conv2D(80, kernel_size=(1, 1), strides=(1, 1), padding='same',
#                kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
#     x = Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same',
#                kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
#     x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', data_format=DATA_FORMAT)(x)
#
#     x = inception_module1(x, params=[(64,), (48, 64), (64, 96), (32,)], concat_axis=CONCAT_AXIS)  # 3a 256
#     x = inception_module1(x, params=[(64,), (48, 64), (64, 96), (64,)], concat_axis=CONCAT_AXIS)  # 3b 288
#     x = inception_module1(x, params=[(64,), (48, 64), (64, 96), (64,)], concat_axis=CONCAT_AXIS)  # 3c 288
#     x = inception_reduce1(x, params=[(384,), (64, 96)], concat_axis=CONCAT_AXIS)  # 768
#
#     x = inception_module2(x, params=[(192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)],
#                           concat_axis=CONCAT_AXIS)  # 4a 768
#     x = inception_module2(x, params=[(192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)],
#                           concat_axis=CONCAT_AXIS)  # 4b 768
#     x = inception_module2(x, params=[(192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)],
#                           concat_axis=CONCAT_AXIS)  # 4c 768
#     x = inception_module2(x, params=[(192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)],
#                           concat_axis=CONCAT_AXIS)  # 4d 768
#     x = inception_module2(x, params=[(192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)],
#                           concat_axis=CONCAT_AXIS)  # 4e 768
#     x = inception_reduce2(x, params=[(192, 320), (192, 192, 192, 192)], concat_axis=CONCAT_AXIS)  # 1280
#
#     x = inception_module3(x, params=[(320,), (384, 384, 384), (448, 384, 384, 384), (192,)],
#                           concat_axis=CONCAT_AXIS)  # 4e 2048
#     x = inception_module3(x, params=[(320,), (384, 384, 384), (448, 384, 384, 384), (192,)],
#                           concat_axis=CONCAT_AXIS)  # 4e 2048
#
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(DROPOUT)(x)
#     x = Dense(num_classes, activation='softmax', kernel_initializer="he_normal",
#               kernel_regularizer=regularizers.l2(weight_decay))(x)
#     return x

def create_model(img_input):
    x = Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    x = inception_module1(x, params=[(64,), (48, 64), (64, 96), (32,)], concat_axis=CONCAT_AXIS)  # 3a 256
    x = inception_module1(x, params=[(64,), (48, 64), (64, 96), (64,)], concat_axis=CONCAT_AXIS)  # 3b 288
    x = inception_module1(x, params=[(64,), (48, 64), (64, 96), (64,)], concat_axis=CONCAT_AXIS)  # 3c 288
    x = inception_reduce1(x, params=[(384,), (64, 96)], concat_axis=CONCAT_AXIS)  # 768

    x = inception_module2(x, params=[(192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)],
                          concat_axis=CONCAT_AXIS)  # 4a 768
    x = inception_module2(x, params=[(192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)],
                          concat_axis=CONCAT_AXIS)  # 4b 768
    x = inception_module2(x, params=[(192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)],
                          concat_axis=CONCAT_AXIS)  # 4c 768
    x = inception_module2(x, params=[(192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)],
                          concat_axis=CONCAT_AXIS)  # 4d 768
    x = inception_module2(x, params=[(192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)],
                          concat_axis=CONCAT_AXIS)  # 4e 768
    x = inception_reduce2(x, params=[(192, 320), (192, 192, 192, 192)], concat_axis=CONCAT_AXIS)  # 1280

    x = inception_module3(x, params=[(320,), (384, 384, 384), (448, 384, 384, 384), (192,)],
                          concat_axis=CONCAT_AXIS)  # 4e 2048
    x = inception_module3(x, params=[(320,), (384, 384, 384), (448, 384, 384, 384), (192,)],
                          concat_axis=CONCAT_AXIS)  # 4e 2048

    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x


img_input=Input(shape=(32,32,3))
output = create_model(img_input)
model=Model(img_input,output)
model.summary()


# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# set callback
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]

# dump checkpoint if you need.(add it to cbks)
# ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

# set data augmentation
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)

datagen.fit(x_train)

# start training
hist=model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))
model.save('inception_v3.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
save_txt('./loss.txt',loss)
save_txt('./val_loss.txt',val_loss)
save_txt('./acc.txt',acc)
save_txt('./val_acc.txt',val_acc)