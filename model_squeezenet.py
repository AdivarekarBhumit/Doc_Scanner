import keras
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, BatchNormalization, Concatenate
from keras.layers import GlobalAveragePooling2D, Flatten, Input, Activation, InputLayer
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.initializers import he_normal, zeros
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from preprocess import *

def conv_2d(inputs, filters, kernel_size, strides, activation, name):
    x = Conv2D(filters=filters, 
               kernel_size=(kernel_size,kernel_size), 
               strides=(strides,strides), 
               padding='same', 
               kernel_initializer=he_normal(seed=4242),
               bias_initializer=zeros(),
               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
               trainable=True,
               activation=activation,
               name=name)(inputs)
    return BatchNormalization()(x)

def fire_module(x, nb_squeeze_filters, nb_expand_filters, name, activation):
    squeeze = conv_2d(x, nb_squeeze_filters, 1, 1, activation, name + '/squeeze')
    expand1x1 = conv_2d(squeeze, nb_expand_filters, 1, 1, activation, name + '/e1x1')
    expand3x3 = conv_2d(squeeze, nb_expand_filters, 3, 1, activation, name + '/e3x3')

    return Concatenate(name = name + '/concat', axis=3)([expand1x1, expand3x3])

def Squeezenet(features=(32,32,1), classes=None, training=True, activation='relu'):
    ip = InputLayer(input_tensor=features)

    x = conv_2d(features, 96, 7, 2, activation, name='conv1')(ip)

    x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)

    x = fire_module(x, 16, 64, name='fire2', activation=activation)
    x = fire_module(x, 16, 64, name='fire3', activation=activation)
    x = fire_module(x, 32, 128, name='fire4', activation=activation)

    x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool4')(x)

    x = fire_module(x, 32, 128, name='fire5', activation=activation)
    x = fire_module(x, 48, 192, name='fire6', activation=activation)
    x = fire_module(x, 48, 192, name='fire7', activation=activation)
    x = fire_module(x, 64, 256, name='fire8', activation=activation)

    x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool8')(x)

    x = fire_module(x, 64, 128, name='fire9', activation=activation)

    if training:
        x = Dropout(0.5)(x)
    
    x = conv_2d(x, classes, 1, 1, activation=activation, name='conv10')

    x = GlobalAveragePooling2D()(x)
    logits = Activation('softmax', name='loss')(x)

    model = Model(inputs=[ip], outputs=logits, name='squeezenet')
    return model

print(Squeezenet(classes=36).summary())