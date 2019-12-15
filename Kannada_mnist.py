# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 19:06:16 2019

@author: Arka_Thesis

This script used for detecting MNIST-Kannada dataset
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf


config = tf.ConfigProto( device_count = {'GPU': 1 } ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
def main():
    df=pd.read_csv("train.csv") #Reading the training dataframe
    label=df['label'].values #Rading the labels
    
    print(type(label)) #Checking the label type
    
    #Converting labels to the one hot encoding
    label_onehot=to_categorical(label)
    
    label_onehot_inverted=np.argmax(label_onehot[0])
    
    
    #Loading the training data and then visulaizing it
    training_data=df.iloc[:,1:].values
    print(training_data[0].reshape((28,28)))
    
    #Visualizing the data
    #img=Image.fromarray(training_data[0].reshape((28,28)))
    
    #Splitting the data
    X_train,X_valid,Y_train,Y_valid=train_test_split(training_data,label_onehot,test_size=0.33,random_state=42)
    
    """
    #Defining the keras ImageDataGenerator
    train_datagen=ImageDataGenerator(featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
    
    valid_datagen=ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
    
    
    #train_datagen.fit(X_train)
    #valid_datagen.fit(X_valid)
    
    train_generator=train_datagen.flow(X_train,Y_train,batch_size=16)
    valid_generator=valid_datagen.flow(X_valid,Y_valid,batch_size=16)
    """
    
    
    #Defining the model
    kannada_model=Sequential()
    kannada_model.add(Dense(1024, input_dim=784,activation='relu'))
    kannada_model.add(Dense(784,activation='relu'))
    kannada_model.add(Dense(256,activation='relu'))
    kannada_model.add(Dense(128,activation='relu'))
    kannada_model.add(Dropout(0.3))
    kannada_model.add(Dense(10,activation='softmax'))
    kannada_model.summary()
    
    #compiling the model
    kannada_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    validation=(X_valid,Y_valid)
    history=kannada_model.fit(X_train,Y_train,batch_size=50,epochs=10,validation_data=validation)
    
    kannada_model.save("Kannada_model.h5")
    #Printing all the history in the list
    print(history.history.keys())
    
    #Summarizing the history
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('epoch')
    plt.legend(['train','valid'])
    plt.savefig("Accuracy.jpg")
    plt.show()
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Loss.jpg")
    plt.show()
    
if __name__=="__main__":
    main()
    
    
