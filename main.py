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
end = dt.datetime(2020,1,1)


data = web.DataReader(company, 'yahoo', start, end)


#Prepare Data
#scales the data between 0 amd 1

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))



##How many days we want to look into the past to predict the next day
days_Past = 60
 
x_train = []
y_train = []

for x in range(days_Past, len(scaled_data)):
    x_train.append(scaled_data[x-days_Past:x, 0])
    y_train.append(scaled_data[x, 0])
    
    
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))


# Build the model 
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) 

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)



#Test the model on previpous data
startTest = dt.datetime(2020,1,1)
endTest = dt.datetime.now()


test_data = web.DataReader(company, 'yahoo', startTest, endTest )

#actual prices from the stock markedt
actual_prices  = test_data['Close'].values


tdataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = tdataset[len(tdataset) - len(test_data) - days_Past:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)



###predict on the test data 
x_test = []

for x in range(days_Past, len(model_inputs)):
    x_test.append(model_inputs[x-days_Past:x, 0])
    

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predicted_prices  = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color="black", label = f"Actual {company} Price")
plt.plot(predicted_prices, color = "green", label=f"predicted {company} Price")
plt.title(f'{company} Share Price')
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()



#####preducting nect stock market day
real_data = [model_inputs[len(model_inputs)+1 - days_Past:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")