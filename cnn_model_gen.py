#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:59:30 2018

@author: salil
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
#, Concatenate
#from tensorflow.keras.optimizers import RMSprop

class ModelGenerator:
    
    def build_cnn(self):
        
        model1 = Sequential()
        
        model1.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu', input_shape = (28,28,1)))
        model1.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model1.add(MaxPool2D(pool_size=(2,2)))
        model1.add(Dropout(0.2))
        
        
        model1.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model1.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model1.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model1.add(Dropout(0.2))
        
        #model = Sequential()
        #conv3_1 = Sequential()
        #conv3_1.add(Dense(512,input_shape = (14,14,1), activation = 'relu'))
         
        #conv3_2 = Sequential()
        #conv3_2.add(Dense(512, input_shape =  (14,14,1), activation = 'relu'))
        #model.add(Concatenate([conv3_1, conv3_2]))
        #model.add(Dense(512, input_shape =  (7,7,1), activation = 'relu'))
        
        model1.add(Flatten())
        model1.add(Dense(1000, activation = "relu"))
        model1.add(Dropout(0.2))
        model1.add(Dense(500, activation = "relu"))
        model1.add(Dropout(0.5))
        model1.add(Dense(10, activation = "softmax"))
        
        model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model1