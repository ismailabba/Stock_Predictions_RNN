#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:42:20 2021

@author: Ismailam
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow. keras.layers import Dense, Dropout, LSTM




##Load the data 
company = 'FB'
start = dt.datetime(2014,1,1)
end = dt.datetime(2021,1,1)


data = web.DataReader(company, 'yahoo', start, end)


#Prepare Data
#scales the data between 0 amd 1

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))



##How many days we want to look into the past to predict the next day
days_Past = 100
 
x_train = []
y_train = []

for x in range(days_Past, len(scaled_data)):
    x_train.append(scaled_data[x-days_Past:x, 0])
    y_train.append(scaled_data[x, 0])
    
    
    
x_train, y_train = np.array(x_train), np.array(y_train)
xtrain = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))