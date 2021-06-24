# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:14:38 2021

@author: as_th
"""

import numpy as np
from tensorflow.keras import models,layers,callbacks,utils
import clcclib_v2 as tools

code2_idx = {"c4":0, "d4":1, "e4":2, "f4":3,"g4":4,"a4":5,
             "c8":6, "d8":7, "e8":8, "f8":9, "g8":10}

idx2_code = {0:"c4", 1:"d4", 2:"e4", 3:"f4", 4:"g4", 5:"a4",
             6:"c8", 7:"d8", 8:"e8", 9:"f8", 10:"g8"}

seq = ["g8","e8","e4","f8","d8","d4","c8","d8","e8","f8","g8","g8","g4",
        "g8","e8","e8","e8","f8","d8","d4","c8","e8","g8","g8","e8","e8","e4",
        "d8","d8","d8","d8","d8","e8","f4","d8","d8","d8","d8","d8","f8","f4",
        "f8","e8","e4","f8","d8","d4","c8","e8","f8","f8","e8","e8","e4"]


def get_data(window_size):
    
    dataset = []
    
    for i in range(len(seq)-window_size):
        subset = seq[i:i+window_size+1]
        dataset.append([code2_idx[a] for a in subset])
        
    dataset = np.array(dataset)
    
    # print(dataset) 
    print(dataset.shape) # 49,6
    
    x_train = dataset[:,:window_size]
    # print(x_train)
    print(x_train.shape) # 49,5
    
    y_train = dataset[:,[window_size]] 
    
    print(y_train.shape) # 49,1
    
    
    return x_train, y_train


# get_data(5)

def do_learn_dnn():
    window_size = 4 # 
    
    x_train,y_train = get_data(window_size)
    
    # 모델 생성
    model = models.Sequential() 
    
    model.add(layers.Dense(128, input_shape=(window_size,),
                           activation ='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(11, activation='softmax'))
    
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    
    log = model.fit(x_train,y_train,
                    epochs = 1000, batch_size=10,
                    validation_split=0.2)
    
    tools.show_log(log)
    
    model.save('navi1.h5')
    
    
# do_learn_dnn()
    

def do_predict_dnn():
    
    model = models.load_model('navi1.h5')
    
    seq_in = ["g8","e8","e4","f8"] # window_size 만큼 가져오기
    
    song_size = len(seq) - 4 #window_size
    
    # print(song_size) # 54
    
    x_test = [code2_idx[a] for a in seq_in] # [10, 8, 2, 9] windows_size
    print(x_test) #[10, 8, 2, 9]
    x_test = np.array(x_test)
    x_test = x_test.reshape(1,4) # 4,~ -> 
    y_predict = model.predict(x_test)
    print(y_predict)    
    
    y_pre = np.argmax(y_predict)
    print(y_pre)
    
    y_pre_num = y_pre
    
    # code 변환
    y_pre = idx2_code[y_pre_num] 
    print(y_pre) # d8
    
    
do_predict_dnn()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

