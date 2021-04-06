# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:51:50 2021

@author: as_th
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# 이진분류와 ㄱ

import clcclib_v2 as tools
import numpy as np
from tensorflow.keras import utils, datasets
from tensorflow.keras import models, layers, callbacks

def get_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    y_train = y_train % 2 
    y_test = y_test % 2 # 0이면 짝수, 1이면 홀수
    
    return (x_train, y_train), (x_test, y_test)


def do_learning():
    (x_train,y_train), (x_test,y_test) = get_data()
    
    
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28,28)))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    log = model.fit(x_train,y_train, 
              epochs=3, batch_size=100,
              validation_split=0.2
              )
    
    tools.show_log(log)
    score = model.evaluate(x_test, y_test) # 15번 결과 0.98정도
    
    print(score)
# do_learning()


def do_learning1(): # binary 모델링
    (x_train, y_train), (x_test, y_test) = get_data()
    
    # 순차 모델 생성
    model = models.Sequential()
    
    model.add(layers.Flatten(input_shape = (28,28)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid')) # binary 문제라서 unit =1, acti = sigmoid 
    # tanh - callback은 초기값 문제로 바로 멈춤, sigmoid는 계속 진행
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', # binary optimizer = rmsprop
                  metrics=['accuracy'])
    
    check_point = callbacks.ModelCheckpoint('my3.h5',# 학습 중 최고 정확도 저장
                                            monitor='val_loss' # 
                                            )   
    
    
    log = model.fit(x_train, y_train,
                    epochs = 3, batch_size = 100,
                    validation_split=0.2,
                    callbacks=[check_point],
                    )
    
    tools.show_log(log)
    score = model.evaluate(x_test, y_test)
    
    print(score)
    
# do_learning()
do_learning1()
    
    
    # 데이터 사망시 다시 로드해서 진행
def do_learing1_load():
    (x_train, y_train), (x_test, y_test) = get_data()
    
    model = models.load_model('my3.h5')
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', # binary optimizer = rmsprop
                  metrics=['accuracy'])
    
    check_point = callbacks.ModelCheckpoint('my3.h5',# 학습 중 최고 정확도 저장
                                            monitor='val_loss' # 
                                            )   
    
    
    log = model.fit(x_train, y_train,
                    epochs = 3, batch_size = 100,
                    validation_split=0.2,
                    callbacks=[check_point],
                    )
    
    tools.show_log(log)
    score = model.evaluate(x_test, y_test)
    
    print(score)
    
    # 확실히 유의미한 변화가 있음 - 상승
    # [0.12889089176452545, 0.9705],[0.09191288088420406, 0.9763] , - 이전 data 
    # [0.07655168330114684, 0.9823], [0.09332128459332889, 0.9839] - 로드한 data
    # 하지만 activation 함수 tanh 사용시 너무 작은 epochs로 인해 중복된 값이 나온다.
    # 초기값이 너무 작을 경우에는 고정된 값이 나오는 문제 발생. epochs = 3경우
    
    
do_learing1_load()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
