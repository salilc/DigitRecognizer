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
        # Building the CNN model by adding Maxpooling and Dropout layers to avoid overfitting
        model = Sequential()
        
        model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu', input_shape = (28,28,1)))
        model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))   
        
        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', 
                         activation ='relu'))
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
     
        # Flattening  for the fully connected layers
        model.add(Flatten())
        model.add(Dense(1000, activation = "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(500, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation = "softmax"))
        #Using the adam optimizer and choosing accuracy as the metric
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model
