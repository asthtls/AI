# -*- coding: utf-8 -*-
"""

@author: clccclcc

last update: 2021.2.17.a

"""
from matplotlib import pyplot as plt

import cv2
import os


# =============================================================================
# 계이름 연주 
# =============================================================================
import winsound


def play_beep(sol,sol_tempo='4/4'):
    sol_org = {'c':261,'d':293,'e':329,'f':349,'g':391,'a':440,'b':493,'x':37} 
    tempo_org={'2/4':2000, '4/4':4000} 
    
    mel = [ m[0] for m in sol]
    dur = [ int(d[1]) for d in sol ]
    tempo = tempo_org[sol_tempo]
    
    for melody,duration in zip(mel,dur):  
        winsound.Beep(sol_org[melody],tempo//duration)
        
        

def play_beep2(sol,sol_tempo='2/4'):
    sol_org = {'c':261,'d':293,'e':329,'f':349,'g':391,'a':440,'b':493,'x':37} 
    tempo_org={'2/4':2000, '4/4':4000} 
    
    mel = [ m[0] for m in sol]
    dur = [ int(d[1]) for d in sol ]
    tempo = tempo_org[sol_tempo]
    
    for melody,duration in zip(mel,dur):  
        winsound.Beep(sol_org[melody],tempo//duration)        
# =============================================================================
# History 보기    
# =============================================================================
# import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
# 손실 이력 클래스 정의
class LossHistory(callbacks.Callback):
    def init(self):
        self.losses = []
        self.accuracy=[]
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
def show_history(history):
    plt.plot(history.losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
def show_historyEx(hist,val=False):    
    fig, loss_ax = plt.subplots()
    
    acc_ax = loss_ax.twinx()
    
    loss_ax.plot(hist.losses , 'y', label='train loss')
    # if val:
    #     loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    
    acc_ax.plot(hist.accuracy , 'b', label='train acc')
    # if val:
    #     acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')
    
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    
    plt.show() 
# =============================================================================
# 파일을 보여 주기 
# =============================================================================
def show_img(file_name,type=0):
    img = cv2.imread(file_name,type) # 1:color 0:gray -1: 알파  
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')    
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
# =============================================================================
# fit한 로그를 보여 주기 
# =============================================================================
def show_log(hist,val=True):
    
    fig, loss_ax = plt.subplots()
    
    acc_ax = loss_ax.twinx()
    
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    if val:
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    
    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    
    if val:
        acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
    
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')
    
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    
    plt.show()
    
def show_log1(hist,val=True):
    
    fig, loss_ax = plt.subplots()
    
    acc_ax = loss_ax.twinx()
    
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    if val:
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    
    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    if val:
        acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')
    
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    
    plt.show()    
# =============================================================================
#  fit 한 로그를 저장하고 가져 옴    
# =============================================================================
# import pickle

# def save_log(hist,file_name):   
#     with open(file_name,'wb') as file:
#         pickle.dump(hist,file)
  
    
        
# def load_log(file_name):
#     with open(file_name,'rb') as file:
#         log = pickle.load(file) 
    
#     return log

# =============================================================================
# 파일 다루기 
# =============================================================================
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            return True
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        return False

# =============================================================================
# 해당 폴더의 파일 리스트를 리턴함 
# =============================================================================
def get_file_name(path):
    file_list = os.listdir(path)
    return file_list

