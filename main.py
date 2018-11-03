#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:02:59 2018

@author: salil

"""
from read_file import DataProcessor
from cnn_model_gen import ModelGenerator
from cnn_model_tune_hyper import ModelTuner
           
if __name__ == '__main__':
    
    # Setting file paths for input images and labels of the mnist dataset
    train_img_path = "/Users/salil/Downloads/train-images-idx3-ubyte"
    train_label_path = "/Users/salil/Downloads/train-labels-idx1-ubyte"
    test_img_path = "/Users/salil/Downloads/t10k-images-idx3-ubyte"
    test_label_path = "/Users/salil/Downloads/t10k-labels-idx1-ubyte"
    dest_train_csv = "/Users/salil/Downloads/mnist_train.csv"
    dest_test_csv = "/Users/salil/Downloads/mnist_test.csv"
    #set training size
    train_size = 60000
    #set test size
    test_size = 10000

    dp = DataProcessor()
    #converting raw images to csv for training and test sets
    dp.convert(train_img_path, train_label_path, dest_train_csv, train_size)
    dp.convert(test_img_path, test_label_path, dest_test_csv, test_size)
    
    #data preprocessing 
    train_x,train_y,test = dp.read_process_file(dest_train_csv,dest_test_csv)

    #Building and tuning cnn models
    modgen = ModelGenerator()
    model = modgen.build_cnn()
    modtune = ModelTuner()
    modtune.cnn_tune_hyper(model,train_x,train_y,test)
   