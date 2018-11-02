#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:56:50 2018

@author: salil
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class DataProcessor:
    #converts raw images from the mnist dataset to csv format
    def convert(self,imgf, labelf, outf, n):
        f = open(imgf, "rb")
        o = open(outf, "w")
        l = open(labelf, "rb")
    
        f.read(16)
        l.read(8)
        images = []
    
        for i in range(n):
            image = [ord(l.read(1))]
            for j in range(28*28):
                image.append(ord(f.read(1)))
            images.append(image)
    
        for image in images:
            o.write(",".join(str(pix) for pix in image)+"\n")
        f.close()
        o.close()
        l.close()
        
    # process the csv files and make the data reayd for modelling.    
    def read_process_file(self,train_file,test_file):
        train = pd.read_csv(train_file,header=None)
        test = pd.read_csv(test_file,header=None)
        #print train.head(10)      
        train_y = train[0]
        print train_y.head(10)
        
        # Drop 'label' column
        train_x = train.drop(labels = [0],axis = 1) 
        g = sns.countplot(train_y)      
        train_y.value_counts()
        plt.show()        
        print (train_x.isnull().any().describe())
        print (train_x.shape)
        
        # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
        train_x = train_x.values.reshape(-1,28,28,1)
        print (train_x.shape)
        #test = test.values.reshape(-1,28,28,1)
        train_x = train_x / 255.0
        test = test / 255.0
        # Encode labels to one hot vectors 
        train_y = to_categorical(train_y, num_classes = 10)
        plt.imshow(train_x[9][:,:,0])
        return train_x,train_y,test