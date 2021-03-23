# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:25:28 2021

@author: as_th
"""

# y = (x-3)**2 + 10 최소값의 x, y
# dy/dx = 2*x - 6

def test1():
    loss_fun = lambda x: (x-3) **2 + 10
    gr_fun = lambda x : 2*x -6
    
    
    x = 10
     
    for i in range(100):
        x = x - 0.2*gr_fun(x)
        loss = loss_fun(x)
        
        print(i, x, loss)
test1()