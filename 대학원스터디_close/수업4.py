# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:29:35 2021

@author: as_th
"""

import numpy as np
from tensorflow.keras import layers, models, optimizers,utils


def test1():
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    
    # 모델
    
    model = models.Sequential()
    model.add(layers.Dense(units=8, activation='sigmoid',input_dim=2)) 
    model.add(layers.Dense(units=1, activation='sigmoid'))
    

    # 하이퍼 파라미터 설정
    mysgd = optimizers.SGD(learning_rate = 0.1)
    model.compile(loss = 'binary_crossentropy', optimizer=mysgd, metrics=['accuracy'])    
    # accuracy 정확도 1epoch당 정확도 나오게 설정
    # optimizer = weight 업데이트 , sgd는 확률적 미분? 방법
    # loss - 에러율 1을 넘어갈 때도 있다. 
    # 일반적으로 acc(정확도)가 1이면 학습이 끝난다.
    
    # 학습 
    model.fit(x, y, epochs = 10000, batch_size=1) # epochs 반복 횟수 - 학습 횟수
    # 학습하는 방법 데이터 4건인데 하나에 하나씩 batch_size = 1 , 1epoch = 4weight 업데이트
    # 배치 사이즈는 1로 하면 제일 좋지만 시간이 오래걸리기 때문에 배치 사이즈 조절이 필수다.
    # ex) batch_size = 2, 1epoch = 2weight 업데이트
    
    
    # 예측
    y_pre = model.predict(x)
    print('y : ', y_pre)
    
#test1()


# iris 데이터
from sklearn import datasets, preprocessing, model_selection
import pandas as pd


def get_data():
    data = datasets.load_iris()
    x = data['data']
    y = data['target']

    # print(' x : ', x.shape) # 150, 4
    # print(' y : ', y.shape) # 150
    # print(' y[:10] : ', y[:10])
    
    print(y[:10])
    y = utils.to_categorical(y)
    # print(y.shape, y[:10])
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2)
    
    return (x_train, y_train),(x_test, y_test)

def get_data1():
    # data = datasets.load_iris()
    data = pd.read_csv('iris.csv')
    
    x = data.iloc[:,:4]
    y = data.iloc[:,4]
    
    print(x.shape, y.shape)
    
    enc = preprocessing.LabelEncoder().fit(y)
    y = enc.transform(y) # 문자를 숫자로 변경
    
    print(y[:10])
    y = utils.to_categorical(y) # 원핫 인코딩으로 변경
    print(y[:10])
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2)
    # 8 : 2로 데이터 나눔
    
    return (x_train, y_train),(x_test, y_test)


def test2():
    
    # 모델
    model = models.Sequential()  #모델 생성
    model.add(layers.Dense(units=512, activation='sigmoid',input_dim=4))
    model.add(layers.Dense(units=3, activation='sigmoid'))
    
    # 파라미터 설정
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    # 학습
    model.fit(x_train,y_train, epochs = 1000, batch_size = 16)
    
    # 평가
    score = model.evaluate(x_test, y_test)
    
    print('score : ', score) # loss, 정확
    
test2()
