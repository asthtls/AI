# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:38:51 2021

@author: as_th
"""

# opencv 3.4.1 version
# opencv 기본 함수 활용

import numpy as np
import cv2 # opencv 

import clcclib_v2 as tools
from tensorflow.keras import datasets

import matplotlib.pyplot as plt


def get_data():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    # print(x_train.shape, y_train.shape) # 60000, 28, 28 / 60000,
    # print(x_test.shape, y_test.shape) # 10000, 28, 28 / 10000,
    
    x = x_train[0] # 첫 번째 데이터
    # print(x) # 0~255 28,28 2차원 데이터
    x2 = x_train[1] 
    # cv2.imwrite("test1.bmp", x)# 신발 # 이미지를 써라, 문자열 이름, 저장 데이터
    # cv2.imwrite("test2.bmp", x2) # 옷 # 28,28 크기 
    
    # print(y_train[0:10]) # 9가 신발 [9 0 0 3 0 2 7 2 5 5]
    # print(y_train[10:20]) # [0 9 5 5 7 9 1 0 6 4] # 아마 옷?
    
    # 반복문으로 이미지 확인
# =============================================================================
#     
#     for i in range(3,13):
#         file_name = "test{}.bmp".format(i) 
#         
#         x = x_train[i]
#         cv2.imwrite(file_name, x)
#         
# =============================================================================
        
    
def get_data1():    
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # 0~9 손글씨 데이터
# =============================================================================
#     
#     print(x_train.shape, y_train.shape) # 60000, 28, 28 / 60000,
#     print(x_test.shape, y_test.shape) # 10000, 28, 28 / 10000,
# =============================================================================
    # 기본적으로 신경망은 28x28, 25x25이상이여야 학습 가능(작아도 가능하지만 효율이 낮다.)
    # 컬러 데이터는 흑백 데이터랑 다름 - 컬러는 rgb 데이터 하나당 3개 값
    # 일반적으로 컬러 데이터는 grey으로 바꾸고 학습
    # 이미지 데이터는 opencv 덕분에 편함 resize - 이미지 크기 조절
    # cv2.resize
    x = x_train[0]
    
    # print(x)
# =============================================================================
#     
#     cv2.imshow("xxx", x) # 여기까지만 하면 바로 사라짐
#     
#     cv2.waitKey(0) # 아무키나 누를때까지 대기
#     
#     cv2.destroyAllWindows() # 삭제 안할시 렉, 메모리 낭비 
#     
# =============================================================================
    # imwrite - 파일 저장 
    
    # plt.imshow(x) #이미지 보기 
    
    # cv2.resize() - 이미지 크기 조절
    
    

    
    
def main():
    # get_data()
    get_data1()
    pass
    
















    
main()