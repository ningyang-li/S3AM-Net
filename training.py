# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:10:40 2023

@author: fes_map
"""

# system packages
import numpy as np
from keras.optimizers import RMSprop
import tensorflow as tf
import time
import winsound as ws
import os

# customized packages
from Parameter import args
from networks.S3AM_Net import S3AM_Net

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "S3AM-Net"

# optimizers
rmsprop = RMSprop(learning_rate=0.001)

# here prepare your own 5-d samples
X, X_train, X_test, X_val
y, y_train, y_test, y_val, y_train_1hot, y_test_1hot, y_val_1hot





# build models
s3am = S3AM_Net(input_shape=sp.input_shape, n_category=sp.n_category)
model = s3am.model
model.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

# waiting user to continue
input("model has been built, please input any key to continnue ...\n")





print("training ...\n")
hist = model.fit(x=sp.X_train, y=sp.y_train_1hot, batch_size=args.bs, epochs=args.epochs,
                    shuffle=True, validation_data=(sp.X_val, sp.y_val_1hot), verbose=1)

# play sound
if args.env == 0:
    ws.PlaySound("C:\\Windows\\Media\\Alarm02.wav", ws.SND_ASYNC)
    
# test
print("testing ...\n")
_, OA = model.evaluate(x=sp.X_test, y=sp.y_test_1hot)


print("\n", time.ctime(time.time()))

