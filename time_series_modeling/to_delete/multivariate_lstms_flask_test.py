#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from scipy.stats import pearsonr 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def json_to_prices(json):
    df = pd.read_json(json)
    prices = df['prices']
    coin_prices = pd.DataFrame(prices.tolist(), columns=['time', 'price'])
    prices = coin_prices['price']
    return prices

def normalize_prices(prices):
    normalized_prices = (prices - np.mean(prices)) / np.std(prices)
    return normalized_prices

def split_data(normalized_prices, split=0.8):
    train_size = int(len(normalized_prices) * split)
    test_size = len(normalized_prices) - train_size
    
    training_data = normalized_prices[0:train_size]
    test_data = normalized_prices[train_size:len(normalized_prices)]
    return training_data, test_data

def create_dataset_helper(data, time_step=1):
    X, y = [], []
    # data = data.reset_index(drop=True)

    for i in range(len(data) - time_step):      
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

def create_dataset(data, time_steps=1):
    X, y = create_dataset_helper(data, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, 0])  # Assuming the first column is DAI price
    return np.array(X), np.array(y)

def create_model(features, time_step):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, features)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# def create_multivariate_model():
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 3)))
#     model.add(LSTM(units=50))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

def predictions(model, X_train, y_train, X_test, y_test, prices):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_predict = (train_predictions * np.std(prices)) + np.mean(prices)
    y_train = (y_train * np.std(prices)) + np.mean(prices)
    test_predict = (test_predictions * np.std(prices)) + np.mean(prices)
    y_test = (y_test * np.std(prices)) + np.mean(prices)
    
    train_score = sqrt(mean_squared_error(y_train, train_predict))
    print('Train Score: %.20f RMSE' % (train_score))
    test_score = sqrt(mean_squared_error(y_test, test_predict))
    print('Test Score: %.20f RMSE' % (test_score))
    return train_predict, test_predict

def output_to_list(json, array):
    df = pd.read_json(json)
    prices = df['prices']
    coin_prices = pd.DataFrame(prices.tolist(), columns=['time', 'price'])
    timeprev = coin_prices['time']
    
    latest_timestamp = timeprev.max()
    #print(latest_timestamp)
    
    num_predictions = len(array)
    predictions_timestamps = [latest_timestamp + i for i in range(1, num_predictions+1)]
    
    print(type(predictions_timestamps))
    print(type(array))
    
    predictions_df = pd.DataFrame({'timestamp': predictions_timestamps, 'predicted price': array.tolist()})
    print(predictions_df)
    return prices.tolist()

def dai_multivariate():
    dai = json_to_prices('time_series_modeling/data/dai90days.json')
    eth = json_to_prices('time_series_modeling/data/ethereum90days.json')
    usdc = json_to_prices('time_series_modeling/data/usdc90days.json')

    #normalize data

    normalized_dai = normalize_prices(dai)
    normalized_eth = normalize_prices(eth)
    normalized_usdc = normalize_prices(usdc)

    # combine the data

    combined_data = pd.concat([normalized_dai, normalized_eth, normalized_usdc], axis=1)

    # remove null values

    combined_data = combined_data.dropna()

    # remove column names
    combined_data = combined_data.values
    print(combined_data)   

    time_step = 10

    combined_training_data, combined_test_data = split_data(combined_data)

    X_train, y_train = create_sequences(combined_training_data, time_step)
    X_test, y_test = create_sequences(combined_test_data, time_step)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = create_model(3, time_step)
    model.fit(X_train, y_train, epochs=50, batch_size=64)


    train_predict, test_predict = predictions(model, X_train, y_train, X_test, y_test, dai)

    test_output_test = { 
        'message' : 'imagine this is the model json outputoutput',
        'UNIX_stamp': "1 million bajillion dollars"
    }
    
    predictedoutput = output_to_list('time_series_modeling/data/dai90days.json', test_predict)

    return predictedoutput

def print_test():
    test_output_test = { 
    'message' : 'imagine this is the test output, for testing whether this python file works'
    }
    return test_output_test




dai_multivariate()