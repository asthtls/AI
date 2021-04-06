# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:02:29 2021

@author: as_th
"""

# opencv 활용한 학습 

import numpy as np
import cv2 
import clcclib_v2 as tools

from tensorflow.keras import datasets, utils
from tensorflow.keras import models, layers, callbacks

# test 데이터 몇개만 file 저장 csv~?
def make_data():
    # tools.create_folder('test_data') # 폴더 생성
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    
    
    for i in range(10):
        file_name = 'test{}.bmp'.format(i)
        # 데이터 확보
        x = x_train[i]
        cv2.imwrite('test_data/'+file_name, x) 
        
        
        
    y = y_train[0:10] # 10개    
    return y
    
    
    
    


def get_data():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    
    # 신경망으로 주기 위해 x만 전처리 안해도 상관없지만 학습 느림
    # 0~255데이터라서 이번은 쉽다.
    
    x_train = x_train/255.0 
    x_test = x_test/255.0
    
    return (x_train, y_train), (x_test, y_test)



def do_learn1():
    (x_train, y_train), (x_test, y_test) = get_data()

    model = models.Sequential() # 직렬 모델
    
    model.add(layers.Flatten(input_shape=(28,28))) # 신경망 마지막 출력은 무조건 1차원
    # 현재 x데이터는 2차원이다. Flatten 함수로 1차원 평평하게 변경
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    # 일반적으로 앞에 128 일 경우 뒤에는 점점 작아짐 - 그게 효율이 더 좋게 나옴
    # 출력층 10종류
    model.add(layers.Dense(10, activation='softmax')) # 종류가 10개
    
    model.compile(loss='sparse_categorical_crossentropy', # 그냥 categorical은 one-hot encoding 추가 해야한다.
                  optimizer='adam',
                  metrics = ['accuracy']
                  
                  ) 

    log = model.fit(x_train, y_train,
              epochs = 15, batch_size = 64, # 일반적 2의 배수
              validation_split=0.2
              )
    
    
    tools.show_log(log) # 저한테는 log
    
    score = model.evaluate(x_test, y_test)
    print(score)    
    
    model.save('0331.h5')
# =============================================================================
#    
#      # dropout x
#      # epochs 5회 결과 train acc = 0.8735
#      # val acc = 0.88~?
#      # train loss = 0.81~
#      #
#      
#      # dropout 사용
#      # epochs 5회 결과 
#      train acc = 0.877~
#      val acc
# 
# =============================================================================

# 예측
# 일반적으로 책에는 predict은 함수 하나에 붙어있지만
# 정석적으로 predict() 함수를 만들어서 설정해놓기.
def do_predict():
    y = make_data()
    # print(y) # [9 0 0 3 0 2 7 2 5 5]
    # 하나씩 읽어와서 테스트
    
    file_list = tools.get_file_name('D:/AI/대학원스터디/test_data')
    # spyder 경로 확인하기 
    # print(file_list)
    
    
    model = models.load_model('D:/AI/대학원스터디/0331.h5')
    model.summary()
    
    
    # 경로 확인 꼭 하기, 설정
    for f in file_list:
        x = cv2.imread('D:/AI/대학원스터디/test_data/'+f,0)
        x = x/255.0 # normalize
        x = x.reshape((1,28,28))
        y_pre = model.predict(x)
        
        print(np.argmax(y_pre)) # index로 나온다.
        # TypeError int, float 형
    






def main():
    # get_data()
    # do_learn1()
    # make_data()
    do_predict()
    pass


main()