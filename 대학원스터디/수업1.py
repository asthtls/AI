# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:00:34 2021

@author: as_th
"""

def test1():
    mysum = lambda a,b: a+b
    a = mysum(2, 3)
    print(a)
    
# test1() # 5
    
    
def test2():
    data = [1,2,3,4]
    
    my_data = list(filter(lambda x: x > 2, data))
    
    print(my_data)
    
    
# test2() # 3, 4



