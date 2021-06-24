# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:50:36 2021

@author: Administrator
"""

import numpy as np
from tensorflow.keras import utils,datasets, wrappers
from tensorflow.keras import models, layers, optimizers

# 사이킷런
from sklearn import ensemble,metrics # 앙상블 

def make_model():
    model = models.Sequential()
    model.add(layers.Dense(200, activation='relu',
                           input_shape=(28*28,)))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) # softmax 10개 분류 문제
    
    model.compile(optimizer='adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return model
    
def do_learn():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    x_train = x_train.reshape(60000,-1) # 60000, - 1은 알아서 자동으로  
    x_test = x_test.reshape(10000,-1) # 이렇게 혹은 Flatten
    
    # print(x_train.shape) # 60000,784
    # print(x_test.shape) # 10000, 784
    
    model1 = wrappers.scikit_learn.KerasClassifier(build_fn=make_model,
                                                   epochs=3, batch_size=100,
                                                   validation_split=0.2) # make_model() x  
    model2 = wrappers.scikit_learn.KerasClassifier(build_fn=make_model,
                                                   epochs=3, batch_size=100,
                                                   validation_split=0.2)
    model3 = wrappers.scikit_learn.KerasClassifier(build_fn=make_model,
                                                   epochs=3, batch_size=100,
                                                   validation_split=0.2)
    
    model1._estimator_type = 'classifier' # 모델 평가 타입
    model2._estimator_type = 'classifier' # 분류 용도
    model3._estimator_type = 'classifier'
    
    # 판단
    my_vote = ensemble.VotingClassifier(estimators=[('model1',model1),
                                                     ('model2',model2),
                                                     ('model3',model3)],
                                        voting='soft') 
    # softvoting, hardvoting 
    # soft = 최종 결과물이 나올 확률 값을 다 더해서 최종 결과물에 대한 각각의 확률을 구한 뒤 최종 값 도출
    # hard =결과물에 대한 최종 값을 투표해서 결정
    # soft, hard https://devkor.tistory.com/entry/Soft-Voting-%EA%B3%BC-Hard-Voting
    
    my_vote.fit(x_train,y_train)
    
    y_pre = my_vote.predict(x_test) 
    print(y_pre) # epoch = 3 .. [7 2 1 ... 4 5 6]
    y_pre1 = my_vote.predict_proba(x_test) # 
    print(y_pre1) # [4.5125215e~~~]
    
    print(metrics.accuracy_score(y_pre, y_test)) # 정확도 계산  0.9622, 0.964
    # y_pre1은 확률
    # y_pre는 정확도라서 y_pre1을 넣으면 에러
    
    # voting 같은 함수 커스텀 - 아마 pytorch? 
    # 좋은 하드웨어 필요 - 좋은 gpu나, 클라우드 컴퓨팅 이용
    # 다음주 시계열 RNN - 순환 신경망  Recurrent Neural Network
    
def main():
    do_learn()
    # make_model()

main()




