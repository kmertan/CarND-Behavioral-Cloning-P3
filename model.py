#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:19:31 2017

@author: kmertan
"""

import os
import csv
import cv2
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import sklearn


lines = []

with open('./driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)
        
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

def generator(samples, batch_size = 32):
    
    num_samples = len(samples)
    while 1: # Loop forever 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            
            images = []
            measurements = []
            correction = .15
            
            for line in batch_samples:
                for i in range(3):
                    
                    source_file = line[i]
                    filename = './IMG/' + source_file.split('/')[-1]
                    image = cv2.imread(filename)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    images.append(image)
                
                    # Augment the data by adding the images flipped over their verticle axes
                    images.append(cv2.flip(image, 1))
                    
                    measurement = float(line[3])
                    
                    if i == 1:
                        measurement += correction
                    
                    if i == 2:
                        measurement -= correction
                        
                    measurements.append(measurement)
                    
                    #Repeat for the flipped image
                    measurements.append(-measurement)
    
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.core import Dropout

train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

model = Sequential()
model.add(Lambda(lambda x: x / 255. - .5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))

# Implementing NVIDIA's pipeline
#model.add(Convolution2D(3, 5, 5, border_mode = 'same', subsample = (2, 2)))
#model.add(Activation('relu'))

model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(48, 3, 3, subsample = (2, 2), activation = 'relu'))

model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))

model.add(Flatten())

model.add(Dropout(.5))
model.add(Dense(100))
model.add(Dropout(.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mae', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch = 6*len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 20)

model.save('model.h5')
