# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:31:59 2021

@author: as_th
"""

# 에폭 중간에 멈추는 방법 == **얼리스탑 세


from tensorflow.keras import utils,datasets
from tensorflow.keras import models,layers,callbacks
import numpy as np

import clcclib_v2 as tools # clcclib_v2.py 

np.random.seed(2021) # 랜덤 고정 
callbacks.TensorBoard(log_dir='...')
# 데이터 받고 전처리 하는 함수
def get_data():
    (x_train,y_train), (x_test,y_test) = datasets.mnist.load_data() # 데이터 받기
    x_train = x_train/255.0 
    x_test = x_test/255.0 
    
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    
    return (x_train,y_train),(x_test, y_test) # 6만개 3차원


# 모델 생성
# 128 -> 64 -> 10 
# 다중분류모델 categorical_entropy

def learning():
    (x_train,y_train),(x_test,y_test) = get_data()
    
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28,28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) # one-hot encodingn 선택이라서 softmax
    model.summary() # 모델이 잘 만들어 졌는지 확인 용도 
    # Non-trainable 전이학습시 학습 안하는 Params
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    
    # 모델 학습 전에 얼리스탑 세팅
    my_stop = callbacks.EarlyStopping(monitor='validation_loss', patience=1)
    my_log = callbacks.TensorBoard(log_dir='mylog')
     # monitor = 뭘 보고 멈출꺼냐 - loss, accuracy
     # patience = 인내, 멈출 위치에서 얼마만큼 더 참고 기다리고 멈출것인가
    
    
    
    
    # 모델 학습
    log = model.fit(x_train,y_train,
              epochs=10, batch_size=100,
              validation_split=0.2,
              callbacks=[my_stop,my_log]
              )
    
    score = model.evaluate(x_test,y_test) # 정답 넣고 평가 
    
    print(score)
    
    model.save('my0317.h5')
    tools.show_log(log)
# learning()



def my_predict():
    model = models.load_model('my0317.h5')
    model.summary()
    (x_train,y_train),(x_test,y_test) = get_data()
        
    x = x_train[0]
    x = x.reshape(1,28,28)
    print(x.shape)
    # 확률까지 받기 model.predict()
    # 값만 받기 model.predict_classes()
    y_pro = model.predict(x)
    print("y_pro : ",y_pro)
    max = np.argmax(y_pro)
    print("y_pro의 최대값 위치 : ", max)    
    y_class = model.predict_classes(x)
    print("x의 값중 최대값 위치 : ",y_class)
        
my_predict()
