# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:03:16 2021

@author: as_th
"""

import numpy as np

# # conda install numpy
# # pip install numpy


# def test1():
#     A = np.array([[1,2],
#                   [3,4],
#                   [5,6]])
#     B = np.array([[2,3],[4,5],[5,6]])
    
#     print("A : ",A.shape)
#     print("B : ",B.shape)
    
#     C = A+B
    
#     print("A + B : ",C)
    
#     D = A-B
    
#     print("A - B  : ", D)

#     E = 2*A
    
#     print("2 * A : ", E)
    
# test1()

# def test2():
#     A = np.array([[1,2],
#                   [3,4],
#                   [5,6]]) # 3x2
#     B = np.array([[1,2,3,4],
#                   [4,5,6,7]]) # 2 x 4
    
#     print("A.shape : ", A.shape) # 3, 2
#     print("B.shape : ", B.shape) # 2, 4
    
#     C = A @ B
#     print("A @ B : ",C)
    
#     D = np.dot(A,B)
#     print("np.dot(A,B) : ", D)
     
    
# test2()



def MLP():
    x = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]]) # 4 x 3
    y = np.array([[0],
                  [1],
                  [1],
                  [0]]) # 4 x 1 
    
    # print("x.shape, y.shape :",x.shape, y.shape)
    
    
    weight0 = 2*np.random.random((3,6)) - 1 # 3 x 6
    weight1 = 2*np.random.random((6,1)) - 1 # 6 x 1
    # print("weight0 : ", weight0)
    # print("weight1 : ", weight1)

    act_f = lambda x : 1/(1+np.exp(-x))
    actf_drive = lambda x: x*(1-x)
    
    for i in range(1000000):
        # 전방향
        layer0 = x
        net1 = layer0 @ weight0 # 4 x 3 @ 3 x 6 = 4 x 6
        # print(net1)
        
        # layer1
        layer1 = act_f(net1)
        # print(layer1)
        
        #layer2
        layer1[:,-1] = 1.0
        net2 = layer1 @ weight1
        layer2 = act_f(net2)
        # print(layer2)


        # 역방향
        layer2_error = layer2 - y
        layer2_delta = layer2_error * actf_drive(layer2)
        
        # print(layer2_delta)
        
        weight1 = weight1 - 0.03 * np.dot(layer1.T, layer2_delta)
        # print('@', weight1)
        
        layer1_error = np.dot(layer2_delta,weight1.T)
        layer1_delta = layer1_error*actf_drive(layer1)
        weight0 = weight0 - 0.03*np.dot(layer0.T,layer1_delta)
        
        print('Layer2 : ',i+1, layer2)

            
MLP()