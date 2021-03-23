# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from tensorflow.keras import datasets, utils
from tensorflow.keras import models, layers

def test1():
    
    (x_train, y_train), (x_test,y_test) = datasets.mnist.load_data()
    
    x_train = x_train/255.0 # .0을 붙이는 이유는 정수는 소수점이 날아가기 때문
    x_test = x_test/255.0
    
    y_train = utils.to_categorical(y_train) 
    y_test = utils.to_categorical(y_test)
    
    model = models.Sequential()
    
    model.add(layers.Flatten(input_shape=(28,28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    
    log = model.fit(x_train,y_train, 
              epochs=15, batch_size=100,
              validation_split=0.2
              )
    
    score = model.evaluate(x_test,y_test)
    
    print(score)

    model.save("my1.h5")    
    
    pass


test1()


def test2():
    model = models.load_model('my1.h5')
    model.summary()
    
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    x_test = x_test/255.0
    
    for i in range(100):
       x = x_test[i]
       x = x.reshape(1,28,28)
       
# =============================================================================
#        y_pre = model.predict(x)
#        
#        
#        y_pre1 = np.argmax(y_pre)
#        
# =============================================================================

       y_pre1 = model.predict_classes(x)

       print(i, " : ", y_test[i],y_pre1)
        
        
        
    
    
    
    
# test2()    
    