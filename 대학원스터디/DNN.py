# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 19:05:05 2021

@author: as_th
"""

# 2021 03 10 

import numpy as np
from tensorflow.keras import datasets, utils
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation



def test1(): 
    (x_train, y_train),(x_test,y_test) = datasets.mnist.load_data()
    
    
    y = utils.to_categorical(y_train) # one-hot encoding 
    
    model = Sequential([
        Dense(28, input_shape=(28,28)),
        Activation('relu'),
        Activation('softmax'),
        ])
    
    
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    
    model.fit(x_train, y_train, epochs=1000, batch_size=100)
    
    scores = model.evaluate(x_test,y_test)



def test2():
# =============================================================================
#     model2 = models.load_model()
# =============================================================================
    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
    
    x = x_train/255
    
    x = x.reshpae(1,28,28) # 
    
test2()