# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:26:07 2021

@author: as_th
"""

# 04 14 
# checkpoint = monitor = "val_acc"

# train 15개 이미지, test 5개 이미지
# 이미지 한 개 확대생산
# 이미지 다루기
# 폴더 전체 이미지 다루기



import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import clcclib_v2 as my_tools
from tensorflow.keras import preprocessing # preprocessing = ?
from tensorflow.keras import models, layers, callbacks


def get_data_learn():
    # ig - image generateor - preprocessing.~
    ig = preprocessing.image.ImageDataGenerator(rescale=1/255.0) # 기본값, 데이터 1~255 사이 전처리                                     )
    x_train_ig = ig.flow_from_directory('handwriting_shape/train',# 경로
                                        target_size =(24,24), # 사이즈
                                        batch_size = 5, # 한 번에 얼마씩 읽을건지
                                        class_mode = 'categorical'# y
                                        ) 
    # print(type(x_train_ig))
    # print(x_train_ig.labels)# 라벨링 어떻게 했는지 보여줌 # numpy가 아니라 shape x
    # print(x_train_ig.class_indices) # 0은 circle 1은 rectan 2는 triangle
        
    ig_test = preprocessing.image.ImageDataGenerator(rescale=1/255.0)
    x_test_ig = ig_test.flow_from_directory('handwriting_shape/test',
                                            target_size = (24,24),
                                            batch_size = 5,
                                            class_mode = 'categorical' # 분류 문제라서 categorical
                                            )
    # print(x_test_ig.labels)
    # print(x_test_ig.class_indices)
    
    return x_train_ig, x_test_ig
    
    # validation, test, train 데이터 3개가 필요 
    # 실제 현업 프로젝트에서는 validation data가 제일 중요 - 목표가 validation
    
    
def do_learn():
    
    x_train_ig, x_test_ig = get_data_learn()
    
    model = models.Sequential()
    
    # cnn
    model.add(layers.Conv2D(32, (3,3),
                            activation='relu', input_shape=(24,24,3))) # 원본이 3채널 color라서 3, 흑백 1
    model.add(layers.Conv2D(64, (3,3),
                            activation='relu'))
    
    model.add(layers.MaxPooling2D((2,2)))
    
    # MLP  편의상 DNN
    model.add(layers.Flatten()) # 2차원 데이터 펴기
    model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(3, activation='softmax'))
    
    # model.summary()
    
    model.compile(loss = 'categorical_crossentropy',# sparse도 테스트 이미 y one-hot encodingdlfktj
                  optimizer = 'adam',
                  metrics = ['accuracy']) 
    
    
    my_checkpoint = callbacks.ModelCheckpoint("check_0414_1.h5",
                                              save_weights_only=False,
                                              monitor = "val_acc",
                                              mode = "max", # 저장 타이밍,
                                              save_best_only = True # 최고점만 저장
                                              )
    
    
    # y가 없어서 fit_generator()
    log = model.fit_generator(x_train_ig,
                              epochs = 10, # 이미 batch_size = 5로 데이터 받아올때 정해둠
                              steps_per_epoch=45/5,
                               validation_data=x_test_ig,
                               validation_steps=15/5, # 15개 훈련, test 5개
                               callbacks=[my_checkpoint]
                              )  
    
    my_tools.show_log(log)
    
    score = model.evaluate_generator(x_test_ig)
    
    print(score)
    
    model.save('0414_1.h5') # checkpoint 사용시 save 코드 불필요
    
    # 현재는 이미지 종류가 3개뿐이라서 저확도 계속 1.0~
    
    # 보안쪽 적대적 신경망
     
    

def do_predict():
    model = models.load_model('0414_1.h5') # check_0414_1.h5
    # model = models.load_model('check_0414_1.h5')    
    
    x = cv2.imread('handwriting_shape/test/circle/circle016.png') 
    # print(x.shape) # 24.24.3 - 3채널 color
    x = x.reshape(1,24,24,3)
     
    y_pre = model.predict(x)
    
    print(y_pre) # reshape 해야 한다. 24, 24, 3 # [1,0,0] circle 예측
# =============================================================================
#     x1 = cv2.imread(('handwriting_shape/test/rectangle/rectangle016.png'))
#     x1 = x1.reshape(1,24,24,3)
#     
#     y1_pre = model.predict(x1)
#     print(y1_pre)
# =============================================================================
# =============================================================================
#     y_pre_cl = model.predict_classes(x)
#     print(y_pre_cl) # index 0 이 정답 예측 
#     predict 더 사용 추천 - 어떤 확률에 따라 예측했다고 나오기 때문 
# =============================================================================

# 폴더 전부
def do_predict_dir():
    model = models.load_model('0414_1.h5') 
    
    path = 'handwriting_shape/test/triangle' # 폴더 경로
    
    file_list = my_tools.get_file_name(path)
    
    print(file_list)
    
    for f in file_list:
        file_name = path + '/' + f
        x = cv2.imread(file_name)
        x = x.reshape(1,24,24,3)
        y_pre = model.predict_classes(x)
        
        print(f, y_pre)
    
    

    
def make_file_data():
    # 이미지 한 개를 이용해 확대 생산
    ig = preprocessing.image.ImageDataGenerator(rescale=1/255.0,  # 1~255
                                                rotation_range = 10,
                                                width_shift_range = 0.2
                                                
                                                )
    # 원본 이미지
    org_image = cv2.imread('handwriting_shape/test/circle/circle016.png') 
    
    org_image = org_image.reshape((1,24,24,3))
    
    my_tools.create_folder('mydata')
    
    data_count = 10
    index = 0
    
    for batch in ig.flow(org_image, save_to_dir='mydata', 
                         save_format='png', batch_size = 1,
                         save_prefix='K'): # format 저장 형식 png 
        index += 1
        if index >= data_count:
            break
           
    
    
    
def main():
      
    # get_data_learn() # 데이터 처리
    # do_learn() # 학습   
    
    # do_predict_dir() # 폴더 전체 예측
    
    # do_predict() # 이미지 한 개 예측
        
    
    # make_file_data() #이미지 확대 생산
    

main()
















    

    
    
    
    
    
    

    
    