# -*- coding: utf-8 -*-
"""Univariate Time Series Forecasting.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zAnBOPwfofM88hGaP3SKKpw38SOqv3YF

# Packages
"""

pip install yfinance

import random
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable

"""# Data

"""

random.seed(100)
random5000_data = [random.randrange(1, 5000, 1) for i in range(1000)]
random50000_data = [random.randrange(1, 50000, 1) for i in range(1000)]
random500000_data = [random.randrange(1, 50000, 1) for i in range(1000)]

airline_data = pd.read_csv("https://raw.githubusercontent.com/blue-yonder/pydse/master/pydse/data/international-airline-passengers.csv", sep=";")
m1_data = pd.read_csv("https://raw.githubusercontent.com/blue-yonder/pydse/master/pydse/data/m1-us-1959119922.csv", sep=";")
canada_cpi_data = pd.read_csv("https://raw.githubusercontent.com/blue-yonder/pydse/master/pydse/data/monthly-cpi-canada-19501973.csv", sep=";")
us_oil_sales_data = pd.read_csv("https://raw.githubusercontent.com/blue-yonder/pydse/master/pydse/data/us-monthly-sales-of-petroleum-an.csv", sep=";")

meli_data = yf.download('meli', progress = False)['Adj Close']
snp_data = yf.download('^GSPC', progress = False)['Adj Close']
rfr_data = yf.download('^TNX', progress = False)['Adj Close']

series = [pd.Series(random5000_data), pd.Series(random50000_data), pd.Series(random500000_data), pd.Series(airline_data['Passengers']), pd.Series(m1_data['M1']), pd.Series(canada_cpi_data['CPI']), 
          pd.Series(us_oil_sales_data['Petrol']), pd.Series(meli_data), pd.Series(snp_data), pd.Series(rfr_data)]
series_names = ['Random_5000', 'Random_50000', 'Random_500000', 'Airline', 'M1', 'Canada CPI', 'US Oil Sales', 'MELI', 'S&P 500', 'Risk Free Rate']

for i in range(len(series)):
  plt.figure(figsize=(20,20))
  plt.subplot(len(series), 1, 1 + i)
  plt.plot(series[i])
  plt.title(series_names[i], y=0.8, loc='left')
  plt.show()

"""## Preprocessing"""

min_max_scaler = preprocessing.MinMaxScaler()

scaled_series = []

for i in range(len(series)):
  a = min_max_scaler.fit_transform(np.array(series[i]).reshape((-1, 1)))
  scaled_series.append(a)

dict_train = {}
dict_test = {}

coef = 0.8

for i in range(len(series)):
  dict_train['serie_{0}'.format(i)] = scaled_series[i][:int(len(series[i])*coef)]

for i in range(len(series)):
  dict_test['serie_{0}'.format(i)] = scaled_series[i][int(len(series[i])*coef):]

dict_x_train = {}
dict_x_test = {}
window = 12

for i in range(len(series)):
  dict_x_train['serie_{0}'.format(i)] = pd.concat([pd.DataFrame(list(dict_train.values())[i]).shift(a, fill_value = 0) for a in range(1, window + 1)], axis = 1, keys = [f"serie_{a}" for a in range(1, window + 1)])

for i in range(len(series)):
  dict_x_test['serie_{0}'.format(i)] = pd.concat([pd.DataFrame(list(dict_test.values())[i]).shift(a, fill_value = 0) for a in range(1, window + 1)], axis = 1, keys = [f"serie_{a}" for a in range(1, window + 1)])

dict_y_train = dict_train
dict_y_test = dict_test

"""Here is important to take into account that the amount of neurons used in the LSTM layer must respect the cycle of the serie. In this case I made the assumption that all series have an annual cycle.

## Model
"""

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape = [1, window]),
  tf.keras.layers.LSTM(window),
  tf.keras.layers.Dense(1)
])
    
model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = "mean_squared_error")

history = {}

for i in range(len(series)):
  history['model_{0}'.format(i)] = model.fit(np.array(list(dict_x_train.values())[i]).reshape((list(dict_x_train.values())[i].shape[0], 1, window)), 
                                             list(dict_y_train.values())[i], epochs = 100, batch_size = 1, verbose = True)

"""## Evaluation"""

preds = {}

for i in range(len(series)):
  preds['preds_{0}'.format(i)] = model.predict(np.array(list(dict_x_test.values())[i]).reshape((list(dict_x_test.values())[i].shape[0], 1, window)))

errors = {}

for i in range(len(series)):
  errors['error_{0}'.format(i)] = np.sqrt(np.mean((list(preds.values())[i].flatten() - list(dict_y_test.values())[i])**2))

correlations = {}

for i in range(len(series)):
  correlations['correlation_{0}'.format(i)] = np.corrcoef(pd.Series(list(preds.values())[i].flatten()), 
                                                          pd.Series(list(dict_y_test.values())[i].reshape((list(dict_y_test.values())[i].shape[0]))))[0][1]

x = PrettyTable()
x.field_names = ['Serie', 'RMSE', 'Correlation', 'Length']

for k in range(len(series)):
  x.add_row([series_names[k], round(list(errors.values())[k], 2), round(list(correlations.values())[k], 2), len(series[k])])

x.sortby = 'Correlation'
x.reversesort = True
print(x)

for k in range(len(series)):
  plt.figure(figsize=(20, 20))
  plt.subplot(len(series), 1, 1 + k)
  plt.plot(list(dict_y_test.values())[k], label = 'Real')
  plt.plot(list(preds.values())[k], label = 'Predicted')
  plt.xlabel("Time")
  plt.ylabel("Scaled value")
  plt.legend(loc = 'best')
  plt.title(f'{series_names[k]}, Correlation: {round(list(correlations.values())[k], 2)}, RMSE: {round(list(errors.values())[k], 2)}')
  plt.show()