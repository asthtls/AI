# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:42:09 2021

@author: as_th
"""

# 
import clcclib_v2 as tools
import numpy as np
from tensorflow.keras import utils, datasets
from tensorflow.keras import layers, callbacks, models, optimizers

np.random.seed(2021) # 랜덤 고정 

def learning():
    (x_train,y_train),(x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train /255.0
    x_test = x_test /255.0
    
    
    # 데이터 처리 모델 만들기
    model = models.Sequential()
    
    model.add(layers.Flatten(input_shape=(28,28))) # 28,28
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2)) # Dropout 중간에 무작위로 노드 뺌
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    my_adam = optimizers.Adam(lr=0.1) # default = 0.0001
    
    model.compile(loss='sparse_categorical_crossentropy', # sparse 사용시 one-hot encondig 사용 안해도 괜찮다.
                  optimizer=my_adam,
                  metrics=['accuracy'])
    
    log = model.fit(x_train, y_train,
        epochs = 10, batch_size = 100,
        validation_split=0.2
        )
    
    tools.show_log(log)
    
    score = model.evaluate(x_test, y_test)
    print(score)
    
    model.save('my0317_2.h5')
    
    
learning()



