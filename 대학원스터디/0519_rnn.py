# -*- coding: utf-8 -*-
"""
Created on Wed May 19 19:02:41 2021

@author: as_th
"""

# RNN 순환 신경망
# Recurrent Neural Network - 자연어처리, 주식 가격 변동, 시계열 데이터 예측에 주로 사용한다.
# 과거가 현재에 영향을 미치는 데이터를 다룸

# 전처리

import clcclib as tools
import numpy as np

code2_idx = {"c4":0, "d4":1, "e4":2, "f4":3,"g4":4,"a4":5,
             "c8":6, "d8":7, "e8":8, "f8":9, "g8":10, "a8":11}
idx2_code = {0:"c4", 1:"d4", 2:"e4", 3:"f4", 4:"g4", 5:"a4",
             6:"c8", 7:"d8", 8:"e8", 9:"f8", 10:"g8", 11:"a8"}

data = ["g8","e8","e4","f8","d8","d4","c8","d8","e8","f8","g8","g8","g4",
        "g8","e8","e8","e8","f8","d8","d4","c8","e8","g8","g8","e8","e8","e4",
        "d8","d8","d8","d8","d8","e8","f4","d8","d8","d8","d8","d8","f8","f4",
        "f8","e8","e4","f8","d8","d4","c8","e8","f8","f8","e8","e8","e4"]


# 데이터 전부 숫자로 변환
def test1():
    
    ans = []
    for a in data:
        idx = code2_idx[a]
        ans.append(idx) 
    
    # 압축 -> ans = [code2_idx[a] for a in data]
    # append, 한 줄로 줄일 수 있는 코드일 경우 압축해서 사용 ->  리스트 컴팩션
    # print(ans)

    # ans = [code2_idx[a] for a in data]
    print(ans)
    
def sequence2_data():
    
    dataset = [] # 데이터 
    window_size = 4 # x 4개 
    
    for i in range(len(data)-window_size): #
        subset = data[i:i+window_size+1] # i부터 i+5+1
        ans = [code2_idx[a] for a in subset] # code2_idx
        dataset.append(ans) 
        
    # print(dataset)
    dataset = np.array(dataset)
    # print(dataset.shape) # 50, 5 = 50개의 데이터 5개씩

    x_train = dataset[:,:4]
    
    # print(x_train)
    # print(x_train.shape) # 50,4
    
    y_train = dataset[:,[4]]
    print(y_train)
    print(y_train.shape) # 50, 1
    
sequence2_data()
    
# def main():
#     # tools.play_beep2(data) # 2/4
#     # tools.play_beep(data) # 4/4
#     # test1()
#     sequence2_data()
#     pass    
# main()


def test2():
    data = [[1,2,3],
            [4,5,6],
            [7,8,9]]
    data = np.array(data)
    
    print(data.shape) # 3,2
    a = data[1][0]
    print(a)
    
    a = data[1,0] # data[1][0] == data[1,0] numpy는 가능하게 해준다.
    print(a)
    
    
    data1 = data[:,[0,1]]
    print(data1)
    
    data2 = data[:,:2]
    print(data2)
    
    pass
    
# test2()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
