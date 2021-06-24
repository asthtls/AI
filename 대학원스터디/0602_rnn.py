# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:55:51 2021

@author: as_th
"""

import numpy as np
from tensorflow.keras import models,layers, utils,callbacks
import clcclib_v2 as tools
import cv2

code2_idx = {"c4":0, "d4":1, "e4":2, "f4":3,"g4":4,"a4":5,
             "c8":6, "d8":7, "e8":8, "f8":9, "g8":10}

idx2_code = {0:"c4", 1:"d4", 2:"e4", 3:"f4", 4:"g4", 5:"a4",
             6:"c8", 7:"d8", 8:"e8", 9:"f8", 10:"g8"}

seq = ["g8","e8","e4","f8","d8","d4","c8","d8","e8","f8","g8","g8","g4",
        "g8","e8","e8","e8","f8","d8","d4","c8","e8","g8","g8","e8","e8","e4",
        "d8","d8","d8","d8","d8","e8","f4","d8","d8","d8","d8","d8","f8","f4",
        "f8","e8","e4","f8","d8","d4","c8","e8","f8","f8","e8","e8","e4"]

np.random.seed(2021)

data = ["e8","d8","e4","d4","g8","g4","c8","c4","g8","c4"]

def get_data(seq, window_size):
    dataset = []
    
    for i in range(len(seq)-window_size): # 
        subset = seq[i:i+window_size+1] # seq 
        dataset.append([code2_idx[item] for item in subset])
    
    dataset = np.array(dataset) # np.arr 변환
    
    x_train = dataset[:,:window_size]
    y_train = dataset[:,window_size] # or windowsize, -1 
    # print(x_train.shape) # 51, 3, int32
    # print(x_train.dtype)
    # print(y_train.shape, y_train.dtype)  # 51,    int32
    
    # 정규화
    
    return x_train,y_train


# print(get_data(seq, 3)) #
# print(get_data(data, 2))

# def do_learn_cnn():
#     x_train, y_train = get_data(seq, 4)
    
#     x_train = x_train/10.0
#     y_train = y_train.reshape(50,1)
    
#     model = models.Sequential()

#     model.add(layers.Conv2D(32),(3,3),
#               activation='relu', input_shape=(28,28,1))
    
#     model.add(layers.MaxPooling2D(2,2))
    
#     model.add(layers.Conv2D(64),(3,3),
#               activation='relu')
#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='relu', input_shape = (4,1)))
#     model.add(layers.Dense(11, activation='softmax'))
    
#     model.compile(loss = 'sparse_categorical_crossentropy',
#                   optimizer = 'adam',
#                   metrics = ['accuracy'])
    
#     log = model.fit(x_train, y_train,
#                     epochs = 1000, batch_size = 10,
#                     validation_split=0.2)
#     tools.show_log(log)
#     model.save('0602_cnn.h5')
# do_learn_cnn()

def do_learn_dnn():
    x_train, y_train = get_data(seq, 4)
    
    x_train = x_train.reshape(50,4,1)
    
    # 정규화 
    x_train = x_train/10.0
    y_train = y_train.reshape(50,1)
    print(x_train.shape)
    print(y_train.shape)
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', input_shape = (4,1)))
    model.add(layers.Dense(11, activation='softmax'))
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    log = model.fit(x_train, y_train,
                    epochs = 10000, batch_size = 10,
                    validation_split=0.2)
    tools.show_log(log)

    model.save('0602_dnn.h5')

# do_learn_dnn()    
    

def do_learn_lstm1():
    x_train, y_train = get_data(seq, 4)    
    
    x_train = x_train.reshape(50,4,1)
    
    # 정규화
    x_train = x_train/10.0
    

    model = models.Sequential()
    
    # 단독형
    model.add(layers.LSTM(128,stateful=True,
                               batch_input_shape=(1,4,1)))
    
    # # 이어져있는 형 - 연속형
    # model.add(layers.LSTM(128, return_sequences=True,
    #                        input_shape=(4,1)))
    # model.add(layers.LSTM(64)) 
    
    # DNN 붙인다.
    model.add(layers.Dense(11, activation='softmax'))
    
    # model.summary()
    
    
    # x = np.array([1,2,3,4])
    # # print(x.shape)
    # x = x.reshape(1,4,1)
    
    # y = model.predict(x)
    # print(y)
    
    print(y_train.shape)
    print(x_train.shape)
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    # 단독 fit stateful일 경우 사용할 수 없다. - 앞에 꺼가 피드백 받으며 나아가는 구조
    # log = model.fit(x_train, y_train,
    #                 epochs = 1000, batch_size = 10,
    #                 validation_split=0.2)
    
    # tools.show_log(log)
    
    # stateful 사용시 
    for epoch_idx in range(200):
        model.fit(x_train, y_train,
                  epochs = 1, batch_size = 1,
                  shuffle=False)        
        model.reset_states()
        
    model.save('0602_lstm1_2.h5')

do_learn_lstm1()

def do_learn_lstm():
    x_train, y_train = get_data(seq, 4)    
    
    x_train = x_train.reshape(50,4,1)
    
    # 정규화
    x_train = x_train/10.0
    

    model = models.Sequential()
    
    # 단독형
    # model.add(layers.LSTM(128, activation='relu', 
                               # input_shape=(4,1)))
    
    # 이어져있는 형 - 연속형
    model.add(layers.LSTM(128, return_sequences=True,
                           input_shape=(4,1)))
    model.add(layers.LSTM(64)) 
    
    # DNN 붙인다.
    model.add(layers.Dense(11, activation='softmax'))
    
    # model.summary()
    
    
    # x = np.array([1,2,3,4])
    # # print(x.shape)
    # x = x.reshape(1,4,1)
    
    # y = model.predict(x)
    # print(y)
    
    print(y_train.shape)
    print(x_train.shape)
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    log = model.fit(x_train, y_train,
                    epochs = 10000, batch_size = 10,
                    validation_split=0.2)
    
    tools.show_log(log)
    
    
    model.save('0602_lstm2.h5')
    
# do_learn_lstm()



def do_learn_rnn():
    x_train, y_train = get_data(seq, 4)    
    
    x_train = x_train.reshape(50,4,1)
    
    # 정규화
    x_train = x_train/10.0
    

    model = models.Sequential()
    
    model.add(layers.SimpleRNN(1, return_sequences=True, 
                               input_shape=(4,1)))
    model.add(layers.SimpleRNN(1, return_sequences=True))
    model.add(layers.SimpleRNN(1,)) # 기본 return_sequences = False
    
    # DNN 붙인다.
    model.add(layers.Dense(11, activation='softmax'))
    
    # model.summary()
    
    
    # x = np.array([1,2,3,4])
    # # print(x.shape)
    # x = x.reshape(1,4,1)
    
    # y = model.predict(x)
    # print(y)
    
    print(y_train.shape)
    print(x_train.shape)
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    log = model.fit(x_train, y_train,
                    epochs = 1000, batch_size = 10,
                    validation_split=0.2)
    
    tools.show_log(log)
    
    
    model.save('0602_rnn.h5')
    
   
# do_learn_rnn()

def do_predict():
    
    model = models.load_model('0602_lstm1_2.h5')
    # model = models.load_model('0602_rnn.h5')
    # model = models.load_model('0602_cnn.h5')
    seq_in = ["g8","e8","e4","f8"] # 처음 4
    seq_out = [a for a in seq_in] # seq_in
    
    # print(seq_out) # 50 번 예측
    
    pred_cnt = 50
    
    seq_in = [code2_idx[item]/10.0 for item in seq_in]# 10, 8, 2, 9]
    
    
    for i in range(pred_cnt):
        sample_in = np.array(seq_in) # np.array형태로
        
        sample_in = sample_in.reshape(1,4,1) # 1,4,1로 
        pred_out = model.predict(sample_in)
        
        index = np.argmax(pred_out) # 숫자로 변환
        
        seq_out.append(idx2_code[index]) # 숫자로 되있던거 "g8"~변환
        seq_in.append(index/10.0)
        
        seq_in.pop(0)
    
        
    
    print(seq_out)
    
    total = len(seq)
    d_cnt = (np.array(seq) != np.array(seq_out)).sum()    
    print(total)
    print(d_cnt)
    
    # rnn
    # 1000번정도 돌리면 2개 더 맞춤.. 
    # 100번 = d_cnt = 38
    # 1000번 = d_cnt = 36
    
    # dnn
    # 1000번 = d_cnt = 35
    # 10000번 = d_cnt = 27
do_predict()

# rnn 1000번 35개 틀림
# lstm 1000번 단층 33개 틀림
# lstm 1000번 연속층 31개 틀림







    
