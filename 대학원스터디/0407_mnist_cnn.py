# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:44:52 2021

@author: as_th
"""

# mnist cnn 학습

import numpy as np
import matplotlib.pyplot as plt

import cv2 # opencv

from tensorflow.keras import datasets, models 
from tensorflow.keras import callbacks, layers

import clcclib_v2 as tools





def get_data_learn():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    
    # print(x_train.shape, y_train.shape) # 60000, 28, 28. /y - 60000
    
    # 정규화
    x_train = x_train/255.0
    x_test = x_test/255.0

    # y는 sparse    
    
    x_train = x_train.reshape((60000, 28,28, 1)) # 혹은 np.reshape(x_train((60000, 28, 28, 1)))
    x_test = x_test.reshape((10000, 28, 28, 1))
    
    return (x_train, y_train), (x_test, y_test)
    
def get_data_predict():
    
    (x_train, y_train), (x_test, y_test) = get_data_learn()
    
    
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3,3),  # 필터 32개? 3x3 크기
                            activation='relu', input_shape=(28,28,1)))
    
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(64, (3,3),
                            activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(64, (3,3),
                            activation = 'relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax'))    
    
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = 'adam',
                  metrics=['accuracy']
                  )
    
    
    mystop = callbacks.EarlyStopping(monitor="val_loss",
                                     patience=10) # 변화- monitor 기록, 기다림 pat 
    
    my_checkpoint = callbacks.ModelCheckpoint("check_0407_2.h5", # 저장 파일 이름 
                                              save_weights_only=False,
                                              monitor = "val_accuracy",
                                              mode = "max", # 저장 타이밍?
                                              save_best_only=True, # 최고점만 저장할거냐
                                              )
    
    
    log = model.fit(x_train, y_train,
                    epochs = 50, batch_size = 64,
                    validation_split=0.2,
                    callbacks=[mystop,my_checkpoint]
                    )
    
    tools.show_log(log)
    
    score = model.evaluate(x_test, y_test)
    
    print(score)
    

    model.save('0407_2.h5')




    
# get_data_predict()

def make_data():
    # tools.create_folder('test2_data') # 폴더 생성
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    
    for i in range(10):
        file_name = 'test{}.bmp'.format(i)
        # 데이터 확보
        x = x_train[i]
        cv2.imwrite('test2_data/'+file_name, x) 
        
        
        
    y = y_train[0:10] # 10개    
    return y


def do_predict():
    
    model = models.load_model('0407_2.h5')
    x = cv2.imread('test2_data/test0.bmp', 0)
    x = x.reshape(1,28, 28, 1)
    print(x.shape)
    
    y_pre = model.predict(x)
    
    print(y_pre)
    
    y_pre_class = np.argmax(y_pre)
    print(y_pre_class)

def main():
    get_data_learn()
    get_data_predict()
    make_data()
    do_predict()
    pass

main()
    
    
