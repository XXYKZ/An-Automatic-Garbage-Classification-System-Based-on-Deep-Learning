import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import keras
from keras.datasets import cifar10
from keras import backend as K
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model
from keras.layers import concatenate,Dropout,Flatten

from keras import optimizers,regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 200
iterations         = 782
DROPOUT=0.5 # keep 50%
CONCAT_AXIS=3
weight_decay=1e-4
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
log_filepath  = './alexnet'


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


def alexnet(img_input, classes=10):
    x = Conv2D(96, (11, 11), strides=(4, 4), padding='same',
               activation='relu', kernel_initializer='uniform')(img_input)  # valid
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='uniform')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(classes, activation='softmax')(x)
    return out

img_input=Input(shape=(32,32,3))
output = alexnet(img_input)
model=Model(img_input,output)
model.summary()

# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# set callback
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]

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
model.save('alexnet.h5')


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
save_txt('./loss.txt', loss)
save_txt('./val_loss.txt', val_loss)
save_txt('./acc.txt', acc)
save_txt('./val_acc.txt', val_acc)