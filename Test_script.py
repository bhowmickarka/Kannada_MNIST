# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:11:24 2019

@author: Arka_Thesis
"""

import keras
import pandas as pd
import numpy as np
from keras.models import load_model

#Enabling the GPU
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 } ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
#Enabling the GPU 

def main():
    #importing the saved model
    model=load_model("Kannada_model.h5")
    model.summary()
    #Importing the test dataframe
    df=pd.read_csv("test.csv")
    
    X_test=df.iloc[:,1:].values
    #X_test=X_test.reshape((5000,784))
    
    #Performing the prediction
    pred=model.predict_classes(X_test)
    
        
    #Creating a submission dataframe
    _id=list(np.arange(len(X_test)))
    
    data={"id":_id,"label":pred}
    df=pd.DataFrame(data)
    df.to_csv("Submission.csv")
    
if __name__=="__main__":
    main()    
    