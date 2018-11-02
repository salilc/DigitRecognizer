#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:01:01 2018

@author: salil
"""
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

class ModelTuner:
    def cnn_tune_hyper(self,model,train_x,train_y):
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state=7)
        lr_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                         patience=3, 
                                         verbose=1, 
                                         factor=0.5, 
                                         min_lr=0.00001)
        
        datagen = ImageDataGenerator(
                  featurewise_center=False,            # set input mean to 0 over the dataset
                  samplewise_center=False,             # set each sample mean to 0
                  featurewise_std_normalization=False, # divide inputs by std of the dataset
                  samplewise_std_normalization=False,  # divide each input by its std
                  zca_whitening=False,                 # apply ZCA whitening
                  rotation_range=30,                   # randomly rotate images in the range (degrees, 0 to 180)
                  zoom_range = 0.1,                    # Randomly zoom image 
                  width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)
                  height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)
                  horizontal_flip=False,               # randomly flip images
                  vertical_flip=False)                 # randomly flip images
        
        epochs = 1
        batch_size = 64
        
        datagen.fit(train_x)
        model.fit_generator(datagen.flow(train_x,train_y, batch_size=batch_size),
                                      epochs=epochs, 
                                      validation_data=(val_x,val_y),
                                      verbose=1, 
                                      steps_per_epoch=train_x.shape[0] // batch_size,
                                      callbacks=[lr_reduction])