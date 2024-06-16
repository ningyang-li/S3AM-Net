# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:08:22 2021

@author: fes_map
"""

from keras import initializers,regularizers,constraints
from keras.engine.base_layer import Layer
from keras import backend as K
from keras.utils.generic_utils import to_list

class Scalable_Gaussian(Layer):
    '''Scalable_Gaussian

    It follows:
    `f(x) = exp(-alpha*(x^2)))
   
    where `alpha` is the learn-able weight with the shapes of (batch_size,1,1,1,1). (default)

    # Input shape
        five-dimensional tensor

    # Output shape
        Same shape as the input.
    '''
    def __init__(self,alpha_initializer=initializers.Constant(1.0),alpha_regularizer=None,alpha_constraint=constraints.NonNeg(),
                 shared_axes=(1,2,3,4),**kwargs):
        super(Scalable_Gaussian,self).__init__(**kwargs)
        self.alpha_initializer=initializers.get(alpha_initializer)
        self.alpha_regularizer=regularizers.get(alpha_regularizer)
        self.alpha_constraint=constraints.get(alpha_constraint)
        if shared_axes is None:
            self.shared_axes = None
        else:
            self.shared_axes = to_list(shared_axes, allow_tuple=True)
        
    
    def build(self,input_shape):
        #create a trainable weight
        print("SG:",input_shape)
        param_shape=list(input_shape[1:])
        self.param_broadcast=[False]*len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i-1]=1
                self.param_broadcast[i-1]=True
        # self.alpha=1.0
        self.alpha=self.add_weight(name="alpha",shape=param_shape,
                                    initializer=self.alpha_initializer,
                                    regularizer=self.alpha_regularizer,
                                    constraint=self.alpha_constraint,
                                    trainable=True)
        super(Scalable_Gaussian,self).build(input_shape)
        
        
    def call(self,x):
        if K.backend()=="theano":
            return K.exp(-K.pattern_broadcast(self.alpha,self.param_broadcast)*K.pow(x,2))
        else:
            # self.alpha=1./(K.mean(x,axis=0,keepdims=True)+K.epsilon())
            return K.exp(-self.alpha*(K.pow(x,2)))


