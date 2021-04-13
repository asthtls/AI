# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:20:30 2021

@author: as_th
"""

# 2021 04 07

# 영상 데이터 처리

import numpy as np
import matplotlib.pyplot as plt

import cv2 # opencv

from tensorflow.keras import datasets, models 
from tensorflow.keras import callbacks, layers

import clcclib_v2 as tools


def get_data_learn(): # 훈련용 데이터
    (x_train, y_train),(x_test, y_test) = datasets.fashion_mnist.load_data()    
    
    # print(x_train.shape, y_train.shape) # 60000, 28, 28 . 60000,
    
    # 정규화
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # y는 loss를 무엇을 쓰느냐에 따라서 정규화를 미리 하고, 안하고가 갈린다.
    # sparse 사용시 미리 정규화 x
    
    x_train = x_train.reshape((60000, 28,28, 1)) # 혹은 np.reshape(x_train((60000, 28, 28, 1)))
    x_test = x_test.reshape((10000, 28, 28, 1))
    
    
    return (x_train, y_train), (x_test, y_test)
    
def do_learn_v1():
    (x_train, y_train), (x_test, y_test) = get_data_learn()
    
    
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3,3),  # 필터 32개? 3x3 크기
                            activation='relu', input_shape=(28,28,1)))
    # Conv2D는 채널까지 고려 - 채널은 색상
    
    model.add(layers.MaxPooling2D((2,2))) # 잘라냄 - 크기 명확하게 바꿔줌 - 노이즈 제거    
    model.add(layers.Conv2D(64, (3,3),
                            activation='relu' ))

    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3),
                            activation='relu'))  # 필터링 과정 
# =============================================================================
#     # model.add(layers.Conv2D(128, (4,4),
#     #                         activation='relu'))
#     # 128, 4, 4 -  acc: 0.8910
#   [0.3074947472929955, 0.891]
#     epochs = 3, batch_size = 64, validation_split=0.2
# =============================================================================
    model.add(layers.Flatten()) # 처음만 input_shape 나중은 필요없음 - 펼치기
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax')) # 출력 10개, 확률 softmax
    
    # model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy', # y one-hot encoding을 안했으니 sparse 사용 
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    
    # earlystop
    mystop = callbacks.EarlyStopping(monitor="val_loss",
                                     patience=10) # 변화- monitor 기록, 기다림 pat 
    
    
    log = model.fit(x_train, y_train,
                    epochs=10, batch_size=64,
                    validation_split=0.2, # 6만개중 0.2
                    callbacks=[mystop] # validation_loss 관찰 - 이상 발생 후 10epochs까지 관찰
                    )
    # 5회까지 earlystop 사용해도 epochs가 너무 작아서 실제 체감은 별로 x
    # 10회까지 earlystop 사용해도 epochs가 너무 작아서 loss가 계속 줄기 때문에 체감 x
    # 예상으로는 50회 부터는 유의미?, 10회시 약간 과대적합
    
    tools.show_log(log) # log , logs
    
    score = model.evaluate(x_test, y_test) # 모델 평가
    
    print(score)
    
    model.save('0407_1_earlystop.h5')
    



def do_learn():
    (x_train, y_train), (x_test, y_test) = get_data_learn()
    
    
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3,3),  # 필터 32개? 3x3 크기
                            activation='relu', input_shape=(28,28,1)))
    # Conv2D는 채널까지 고려 - 채널은 색상
    
    model.add(layers.MaxPooling2D((2,2))) # 잘라냄 - 크기 명확하게 바꿔줌 - 노이즈 제거    
    model.add(layers.Conv2D(64, (3,3),
                            activation='relu' ))

    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3),
                            activation='relu'))  # 필터링 과정 
# =============================================================================
#     # model.add(layers.Conv2D(128, (4,4),
#     #                         activation='relu'))
#     # 128, 4, 4 -  acc: 0.8910
#   [0.3074947472929955, 0.891]
#     epochs = 3, batch_size = 64, validation_split=0.2
# =============================================================================
    model.add(layers.Flatten()) # 처음만 input_shape 나중은 필요없음 - 펼치기
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax')) # 출력 10개, 확률 softmax
    
    # model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy', # y one-hot encoding을 안했으니 sparse 사용 
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    
    # earlystop
    mystop = callbacks.EarlyStopping(monitor="val_loss",
                                     patience=10) # 변화- monitor 기록, 기다림 pat 
    
    my_checkpoint = callbacks.ModelCheckpoint("check_0407.h5", # 저장 파일 이름 
                                              save_weights_only=False,
                                              monitor = "val_accuracy",
                                              mode = "max", # 저장 타이밍?
                                              save_best_only=True, # 최고점만 저장할거냐
                                              )
    
    
    log = model.fit(x_train, y_train,
                    epochs=3, batch_size=64,
                    validation_split=0.2, # 6만개중 0.2
                    callbacks=[mystop,my_checkpoint] # validation_loss 관찰 - 이상 발생 후 10epochs까지 관찰
                    )
    # 5회까지 earlystop 사용해도 epochs가 너무 작아서 실제 체감은 별로 x
    # 10회까지 earlystop 사용해도 epochs가 너무 작아서 loss가 계속 줄기 때문에 체감 x
    # 예상으로는 50회 부터는 유의미?, 10회시 약간 과대적합
    
    tools.show_log(log) # log , logs
    
    score = model.evaluate(x_test, y_test) # 모델 평가
    
    print(score)
    
    # model.save('0407_1_earlystop.h5')
    
# =============================================================================
#     
#      dropout 미사용시
#     epochs = 3, batch_size = 64, validation_split=0.2
#     acc = 0.8899, [0.30823662559986115, 0.8899]
#     
#     
# =============================================================================
# =============================================================================
# 
#  dropout 사용 - 1번   
# epochs = 3, batch_size = 64, validation_split=0.2  
#  acc: 0.8841
# [0.3212956861972809, 0.8841]
# =============================================================================


do_learn()    
 
    

def do_predict():
    x = cv2.imread('test_data/test0.bmp',0)
    # print(x.shape) # 28,28,3 - 목적은 1채널 데이터는 현재 3채널
    # imread 파라미터 0 넣으면 28, 28  - 0 넣으면 1채널로 강제 변경
    
    model = models.load_model('0407_1.h5')
    
    
    x = cv2.imread('test_data/test0.bmp',0)
    x = x.reshape(1,28,28,1) 
    y_pre = model.predict(x) # 28, 28, 1 이라서 에러
    
    # print(y_pre) # [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]] 이미지는 신발임.
    
    y_pre_class = np.argmax(y_pre)
    
    # print(y_pre_class) # 9번째 
    
    y_pre_class1 = model.predict_classes(x)
    # print(y_pre_class1) # argmax 상태로 나온다. [9]
    
    # 일반적으로 사용시 model.predict() 사용 권장 
    # model.predict_classes 사용시 놓치는 부분이 생길 수 있다.
    
    
    
    

# do_predict()











    
    
 
    
 
