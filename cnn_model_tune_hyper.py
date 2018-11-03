#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:01:01 2018

@author: salil
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

class ModelTuner:
    # Tune the CNN model
    def cnn_tune_hyper(self,model,train_x,train_y,test):
        epochs = 10
        batch_size = 64
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state=7)
  
        lr_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                         patience=3, 
                                         verbose=1, 
                                         factor=0.5, 
                                         min_lr=0.00001)
        # Data Augmentation  
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
        
     
        
        datagen.fit(train_x)
        # fit the model
        model.fit_generator(datagen.flow(train_x,train_y, batch_size=batch_size),
                                      epochs=epochs, 
                                      validation_data=(val_x,val_y),
                                      verbose=1, 
                                      steps_per_epoch=train_x.shape[0] // batch_size,
                                      callbacks=[lr_reduction])
        self.print_confusion_matrix(model,val_x,val_y)
        self.predict_results(test,model)
        return model
        
    # confusion matrix for true and predicted labels 
    def print_confusion_matrix(self,model,val_x,val_y):
        # Predict the values from the validation dataset
        y_pred = model.predict(val_x)
        # Convert predictions classes to one hot vectors 
        y_pred_classes = np.argmax(y_pred,axis = 1) 
        # Convert validation observations to one hot vectors
        y_true = np.argmax(val_y,axis = 1) 
        # compute the confusion matrix
        confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
        # plot the confusion matrix
        f,ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show() 
        
    # predict results on the test set
    def predict_results(self,test,model):
        results = model.predict(test)
        # select the index with the maximum probability
        results = np.argmax(results,axis = 1)
        results = pd.Series(results,name="Label")
        print results