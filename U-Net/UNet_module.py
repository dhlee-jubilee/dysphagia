from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.activations as activation
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Add, Activation, Multiply


def Attention_block(G, X, F_int):
    W_g = Conv2D(F_int, kernel_size=1)(G)
    W_g = BatchNormalization()(W_g)

    W_x = Conv2D(F_int, kernel_size=1)(X)
    W_x = BatchNormalization()(W_x)

    W_g_W_x = Add()([W_g, W_x])
    psi = Activation('relu')(W_g_W_x)
    psi = Conv2D(1, kernel_size=1)(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    result = Multiply()([X, psi])
    return result


""" Sequential Model """
def conv_block(ch_in, ch_out): ## ch_in: input,  ch_out: Constatnt
    model = Sequential()
    model.add(Conv2D(ch_out, (3, 3), strides=1, padding='same', input_shape=ch_in))
    model.add(BatchNormalization(axis = 3))
    model.add(ReLU(shared_axes=[1, 2]))
    model.add(Conv2D(ch_out, (3, 3), strides=1, padding='same'))
    model.add(BatchNormalization(axis = 3))
    model.add(ReLU(shared_axes=[1, 2]))
    
    return model

                  
def single_conv(ch_in, ch_out):
      model = Sequential()
      model.add(Conv2D(ch_out, (3, 3), strides=1, padding='same', input_shape=ch_in))
      model.add(BatchNormalization(axis = 3))
      model.add(ReLU(shared_axes=[1, 2]))

                  
def Recurrent_block(ch_out, t=2):
    for i in range(t):
        if i == 0:
            x1 = single_conv(ch_out, ch_out)
        x1 = single_conv(ch_out + x1)

    return x1

def RRCNN_block(ch_in, ch_out, t=2):
    model = Sequential()
    model.add(Recurrent_block(ch_out, t=t))
    model.add(Recurrent_block(ch_out, t=t))
                  
    conv = Conv2D(ch_out, (1, 1), padding='valid')(ch_in)
    x1 = model(conv)

    return ch_in + x1
