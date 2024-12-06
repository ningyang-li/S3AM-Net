# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:36:42 2022

@author: fes_map
"""


from keras.layers import Add, Activation, Conv3D, BatchNormalization
import tensorflow.keras.backend as K

from .Network import Network


class ResBlock_3D(Network):
    '''
    build the resnet for feature extraction
    parameters:
        n: the number of the residual blocks
        filters_1: the number of the filters of the convolutional layer in the first residual block,  
                   and that of the subsequent residual blocks are (filters_1 * 2,  filters_1 * 4,  ...)
    '''
    def __init__(self, x, filters=32, kernel_size=(3, 3, 7), strides=(1, 1, 1)):
        super().__init__("ResBlock_3D")
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = "relu"
        self.output = self.residual_block(x)
        
        print("ResBlock_3D build success")


    def residual_block(self, x):
        f = Conv3D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                   padding="same", data_format=self.DT, activation=self.activation)(x)
        f = Conv3D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
                   padding="same", data_format=self.DT)(f)
        
        if K.int_shape(x)[1] != self.filters:
            x = Conv3D(filters=self.filters, kernel_size=1, strides=1, padding="valid", data_format=self.DT)(x)
        
        o = Add()([f, x])
        o = BatchNormalization(axis=1)(o)
        o = Activation(self.activation)(o)
        
        return o

