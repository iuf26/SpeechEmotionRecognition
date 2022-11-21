import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def AlexNet(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(96, (11, 11), strides=4, name="conv0")(X_input)
    X = BatchNormalization(axis=3, name="bn0")(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max0')(X)

    X = Conv2D(256, (5, 5), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max1')(X)

    X = Conv2D(384, (3, 3), padding='same', name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    X = Conv2D(384, (3, 3), padding='same', name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)

    X = Conv2D(256, (3, 3), padding='same', name='conv4')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=2, name='max2')(X)

    X = Flatten()(X)

    X = Dense(4096, activation='relu', name="fc0")(X)

    X = Dense(4096, activation='relu', name='fc1')(X)

    X = Dense(6, activation='softmax', name='fc2')(X)

    return  Model(inputs=X_input, outputs=X, name='AlexNet')