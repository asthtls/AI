# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:27:04 2021

@author: as_th
"""

# 회귀문제 값 예측
# 주식 데이터는 시계열 데이터

# 회귀분석

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers


def get_data():
    x_train = np.random.random((1000,1)) # 데이터 가져오기 랜덤으로 0~1사이
    # 1차원 1000개
    
    # print(x_train.shape, x_train[:10])
    
    # x_train1 = x_train.reshape(1000, )
    # print(x_train1) [0 ~ 1]
    
    y_train = 2*x_train + np.random.random((1000, 1))/3
    
    # print(y_train[:10])
    
    x_test = np.random.random((100, 1)) # 테스트 데이터는 100개만 보통 75 . 25
    
    y_test = 2*x_test + np.random.random((100, 1)) /3 
    
    x_train = x_train.reshape(1000,)
    x_test = x_test.reshape(100,)
    
    y_train = y_train.reshape(1000, )
    y_test = y_test.reshape(100, )
    
    # print(x_test)
    
    return (x_train, y_train), (x_test, y_test) # 데이터 reshape 후 리턴  2차원 -> 1차원

# get_data()
    

def do_learning():
    (x_train, y_train), (x_test, y_test) = get_data()
    
    # print(x_train.shape) # (1000,)
    
    model = models.Sequential()
    
    model.add(layers.Dense(64, activation='relu', input_dim=1)) # relu 미사용시 
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) # 숫자가 나오므로 1
    # unit = 1, 회귀분석이므로 마지막 activation = 없다.
    
    model.compile(loss='mse',
                  optimizer ='rmsprop' # adam도 사용해보기
                  # 정확도는 회귀분석이므로 의미가 없다.
                  ) # 중요한건 loss
    
    log = model.fit(x_train, y_train,
                    epochs=100, batch_size=64
                    )
    
    # 평가 의미없음. 회귀분석이므로 - predict()으로 예상
    
    model.save('kk.h5')
    
    plt.plot(log.history['loss']) # 
    plt.show()
    
    # 명확한 패턴의 공식이므로 잘 맞춤 
    # 하지만 현실의 데이터는 비선형, 매우 정리하기 힘든 데이터이므로 어렵다.
    # 라벨링 매우 중요
    
    
# do_learning()
    
    
def do_predict():
    (x_train, y_train), (x_test, y_test) = get_data()
    
    model = models.load_model('kk.h5')
    
    
    # x = x_test[:]
    y_pre = model.predict(x_test)
    
    print(y_pre)
    
    print(y_test[:5])
    
    plt.plot(x_test, y_test, 'rx') 
    
    plt.plot(x_test, y_pre, 'bo') # 예측값이 오차범위 안에 존재
    
    plt.show()
    
    
    
do_predict()












    
    
    
    
    
    
    
    
    
    
    











