# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:36:41 2022

@author: fes_map
"""


from keras.layers import Input, Multiply, Reshape, Permute, Conv3D, Concatenate, Lambda, Flatten, Dense
from keras.layers import MaxPooling3D, MaxPooling2D, AveragePooling3D, AveragePooling2D
from keras.models import Model
from keras import initializers
import tensorflow.keras.backend as K

from .Network import Network
from .ResBlock_3D import ResBlock_3D
from .layers.Adaptive_Sum import Adaptive_Sum
from .activations.Scalable_Gaussian import Scalable_Gaussian


class S3AM_Net(Network):
    '''
    paper title:    Spectral similarity-based spatial attention module for hyperspectral image classification
    input:          5-dimensional tensor
    '''
    def __init__(self, input_shape, n_category, stack=3):
        super().__init__("S3AM-Net", input_shape, n_category)
        
        self.stack = 3
        
        self.build_model()
        
    def extract(self, x, b_expand=True):
        '''
        x: 5-d tensor (n_sp, n_channel, n_row, n_col, n_band)
        extract the spectral on the position of "pos"
        
        if 'b_expand' is True,  the output will has the same shape as 'x'
        '''
        pos = (self.n_row // 2, self.n_col // 2)
        
        center = x[:, :, pos[0]:pos[0] + 1, pos[1]:pos[1] + 1, :]
        if b_expand:
            center = K.tile(center, (1, 1, self.n_row, self.n_col, 1))
        
        return center
    
    
    def similarity_cos(self, x):
        center_x = x[:, 0:1]
        center = x[:, 1:2]
        x = x[:, 2:]
        
        center_sqrt = K.sqrt(center)
        x_sqrt = K.sqrt(x)
        sim = center_x / (center_sqrt * x_sqrt)
        sim = 1. - sim
        
        return sim
    
    
    def WCD(self, x):
        '''
        spectral cos-similarity-based spatial attention module
        '''        
        # extract
        c = Lambda(self.extract, output_shape=(1, self.n_row, self.n_col, self.n_band), arguments={"b_expand":True}, name="extract_c")(x)
        
        # element-wise operations
        ewo_cx = Multiply()([c, x])
        ewo_cc = Multiply()([c, c])
        ewo_xx = Multiply()([x, x])
        
        # full-band convolution
        W = Conv3D(filters=1, kernel_size=(1, 1, self.n_band), strides=(1, 1, self.n_band), data_format=self.DT, name="fbc_o", activation="relu", 
                        kernel_initializer=initializers.random_normal(mean=1.0, stddev=0.1), use_bias=False)
        fbc_cx = W(ewo_cx)
        fbc_cc = W(ewo_cc)
        fbc_xx = W(ewo_xx)
        
        # similarity
        cx_c_x = Concatenate(axis=1)([fbc_cx, fbc_cc, fbc_xx])
        sim = Lambda(self.similarity_cos, output_shape=(1, self.n_row, self.n_col, 1), name="sim_c")(cx_c_x)
        
        return sim
    
    
    def substract_power(self, x, power=1):
        x1 = x[:, 0:1]
        x2 = x[:, 1:]
        
        return K.pow(K.abs(x1 - x2), power)
    
    
    def similarity_Minkowski(self, x, power=2):
        sim = K.pow(x, power)
        
        return sim
    
    
    def WED(self, x, power=2):
        '''
        spectral cos-similarity-based spatial attention module
        '''
        # extract
        c = Lambda(self.extract, output_shape=(1, self.n_row, self.n_col, self.n_band), arguments={"b_expand":True}, name="extract_e")(x)
        
        # element-wise operations
        cx = Concatenate(axis=1)([c, x])
        ewo = Lambda(self.substract_power, output_shape=(1, self.n_row, self.n_col, self.n_band), arguments={"power":power}, name="ewo_d")(cx)
        
        # full-band convolution
        W = Conv3D(filters=1, kernel_size=(1, 1, self.n_band), strides=(1, 1, self.n_band), data_format=self.DT, name="fbc_d", activation="relu", 
                        kernel_initializer=initializers.random_normal(mean=1.0, stddev=0.1), use_bias=False)
        fbc = W(ewo)
        
        # similarity
        sim = Lambda(self.similarity_Minkowski, output_shape=(1, self.n_row, self.n_col, 1), arguments={"power":1. / power}, name="sim_e")(fbc)
        
        return sim
    
    
    def S3AM(self, x, power=2):
        n_sp = K.int_shape(x)[0]
        sim_e = self.WED(x, power)
        sim_c = self.WCD(x)
        sim = Concatenate(axis=-1)([sim_e, sim_c])
        sim = Adaptive_Sum(output_dim=(n_sp, 1, self.n_row, self.n_col, 1), name="sim")(sim)
        mask = Scalable_Gaussian(name="mask")(sim)
        
        return mask
    
    
    def S3AM_WED(self, x, power=2):
        sim_e = self.WED(x, power)
        mask = Scalable_Gaussian(name="mask")(sim_e)
        
        return mask
    
        
    def S3AM_WCD(self, x):
        sim_c = self.WCD(x)
        mask = Scalable_Gaussian(name="mask")(sim_c)
        
        return mask
    

    def S3AM_SG_advanced(self, x, power=2):
        '''
        put the SG activation function after the WED/WCD parts
        '''
        n_sp = x.shape[0]
        
        sim_e = self.WED(x, power)
        mask_e = Scalable_Gaussian(name="mask_e")(sim_e)
        
        sim_c = self.WCD(x)
        mask_c = Scalable_Gaussian(name="mask_c")(sim_c)
        
        sim = Concatenate(axis=-1)([mask_e, mask_c])
        sim = Adaptive_Sum(output_dim=(n_sp, self.n_channel, self.n_row, self.n_col, 1), name="adaptive")(sim)
        
        return sim
    
    
    def ResNet_3D(self, x):
        c = ResBlock_3D(x, filters=8, kernel_size=(3, 3, 7)).output
        
        for i in range(1, self.stack):
            if K.int_shape(c)[2] >= 3: 
                p = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format=self.DT)(c)
            c = ResBlock_3D(p, filters=8 * 2 * i, kernel_size=(3, 3, 7)).output
        
        o = AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format=self.DT)(c)
        
        return o
     

    def build_model(self):
        # input
        I = Input(shape=self.input_shape, name="input_0")
        
        # Attention
        mask = self.S3AM(I)
        wx = Multiply(name="multiply")([mask, I])
        
        f = self.ResNet_3D(wx)
        f = Flatten()(f)
        P = Dense(self.n_category, activation="softmax")(f)
        
        self.model = Model(inputs=I, outputs=P, name="S3AM-Net")
        print(self.name + " build success")
    
        
        
        
        
        
