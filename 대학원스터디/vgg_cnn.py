# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 18:20:43 2021

@author: as_th
"""
# 04 21  CNN 기반 

# 전이학습   vgg 모델 keras 포함

from tensorflow.keras import datasets, utils
from tensorflow.keras import models, layers, callbacks, applications
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import clcclib_v2 as tools
# applications 전이학습 포함 많은 모델이 있다.

# make_model_v1, v2 두 가지 방식 

def make_model_v2():
    model = applications.VGG16(include_top=False,
                               input_shape=(32,32,3))
    
    flat = layers.Flatten()(model.layers[-1].output) 
    # 파이토치 방식?
    
    layer1 = layers.Dense(256,activation='relu')(flat)
    layer2 = layers.Dense(10, activation='softmax')(layer1)
    
    new_model = models.Model(inputs=model.inputs, outputs=layer2)
    
    # params 끄끼
    for layer in new_model.layers[:-3]: # 뒤에 3층은 Flatten, relu,softmax
        layer.trainable = False

    
    new_model.summary() #
    
    return new_model
        
    
    
def make_model_v1():
    vgg_model = applications.VGG16(include_top=False, 
                                   input_shape=(32,32,3)) # False 32,32,3 params = 1400만
    # True 224,224,3 약 1억 4천만 params
    
    for layer in vgg_model.layers:
        layer.trainable = False
        
        
    model = models.Sequential() # 순차 모델
    
    model.add(vgg_model)
    model.add(layers.Flatten()) # 펼치기    
    
    # DNN 부분 분류
    model.add(layers.Dense(256, activation='relu'))
    # 출력층
    model.add(layers.Dense(10, activation='softmax'))
        
    model.summary() # 학습 가능은 133,898뿐이다. 총 params은 14,848,586
    
    return model


def make_data(): # _ - 변수는 받지만 사용하지 않는다. annonimus? 어노니머스
    (x_train, y_train),(_, _) = datasets.fashion_mnist.load_data() 
    
    # 폴더 생성
    # tools.create_folder('fashion_mnist')
    
    x_train_my = np.empty((0,32,32,3), int) # integer형식
    
    
    for i in range(1000):
        x = x_train[i]
        cv2.imwrite('temp.png',x)
        img = cv2.imread('temp.png', 0)
        img1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # color지만 채널을 전부 grey
        # fashion_mnist 데이터가 color가 아니라서 강제로 3채널로 변경
        
        img1_size = cv2.resize(img1, (32,32)) # 28,28 -> 32, 32
        
        x_train_my = np.append(x_train_my, np.array([img1_size]),
                               axis=0)
        # 나중에 y도 만들기
            
    np.save('my.npy', x_train_my)
    

def get_data_train():
    (_, y_train_org), (_,_) = datasets.fashion_mnist.load_data()

    x_train_my = np.load('my.npy')
    
    x_train = x_train_my[0:900]
    x_test = x_train_my[900:]
    
    # print(x_train.shape) # 900, 32, 32, 3
    # print(x_test.shape) # 100, 32, 32, 3
    
    y_train = y_train_org[0:900]
    y_test = y_train_org[900:1000]
    
    # print(y_train.shape) # 900
    # print(y_test.shape) # 100
    
    return (x_train, y_train), (x_test, y_test)
    
def do_learn():
    (x_train, y_train), (x_test, y_test) = get_data_train()
    
    x_train = x_train /255.0
    x_test = x_test /255.0
    
    # y는 sparse
    
    model = make_model()
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    my_checkpoint = callbacks.ModelCheckpoint("check_0421_1.h5",
                                              save_weights_only=False,
                                              monitor='val_acc',
                                              mode = 'max',
                                              save_best_only = True) # 최고점만 저장
    
    log = model.fit(x_train, y_train,
                    epochs = 20, batch_size = 100,
                    validation_split=0.2,
                    callbacks = [my_checkpoint])
    
    tools.show_log(log)
    
    score = model.evaluate(x_test,y_test)
    
    print(score) # 2번 [1.1884526062011718, 0.67]
    # 10번 [0.7082094192504883, 0.79]
    # 20번 [0.6891275405883789, 0.79] 10~20번은 차이가 별로 없다. 
    
    pass

    
    
def main():
    make_model_v2()
    # make_model_v1() # vgg16 모델 생성 return model
    # make_data() # 학습할 데이터 만들기
    # get_data_train() # 훈련 데이터 가져오기 x_train
    # do_learn()
    pass


main()











