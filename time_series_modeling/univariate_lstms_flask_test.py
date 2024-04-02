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
    data = data.reset_index(drop=True)

    for i in range(len(data) - time_step):      
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

def create_dataset(data, time_steps=1):
    X, y = create_dataset_helper(data, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y

def create_model():
    time_step = 10
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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
    print('Test Predictions: ' + str(test_predict))
    return train_predict, test_predict

def plot_time_series(train_predict, test_predict, prices, time_step=10):
    train_predict_flattened = train_predict.flatten()
    test_predict_flattened = test_predict.flatten()
    train_size = len(train_predict)
    test_size = len(test_predict)
    
    plt.figure(figsize=(10, 6))
    plt.plot(prices, color='blue', label='Actual Price')
    plt.plot(range(time_step, train_size + time_step), train_predict_flattened, color='green', label='Train Predicted Price')
    plt.plot(range(train_size + time_step, train_size + time_step + test_size), test_predict_flattened, color='red', label='Test Predicted Price')

    plt.title('USDC Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

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

def create_simple_model():
    time_step = 10
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(time_step, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def usdc_univariate():
    prices = json_to_prices('time_series_modeling/data/usdc90days.json')

    # Normalize the prices data
    normalized_prices = (prices - np.mean(prices)) / np.std(prices)
    print(normalized_prices)

    time_step = 10 # the number of time steps you're looking at to predict the next step.

    # Split the data into training and testing data
    training_data, test_data = split_data(normalized_prices)
    X_train, y_train = create_dataset(training_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = create_model()
    model.summary()
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    train_predict, test_predict = predictions(model, X_train, y_train, X_test, y_test, prices)

    # flatten the predictions
    # train_predict_flattened = train_predict.flatten()
    # test_predict_flattened = test_predict.flatten()
    # train_size = len(train_predict)
    # test_size = len(test_predict)

    predictedoutput = output_to_list('time_series_modeling/data/usdc90days.json', test_predict)

    return predictedoutput

    # graph the results

    # reshape train_predict and test_predict to the shape of prices

    # plt.figure(figsize=(10, 6))
    # plt.plot(prices, color='blue', label='Actual Price')
    # plt.plot(range(time_step, train_size + time_step), train_predict_flattened, color='green', label='Train Predicted Price')
    # plt.plot(range(train_size + time_step, train_size + time_step + test_size), test_predict_flattened, color='red', label='Test Predicted Price')

    # plt.title('USDC Price Prediction')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()

def eth_univariate():

    time_step = 10

    prices = json_to_prices('time_series_modeling/data/ethereum90days.json')
    normalized_prices = normalize_prices(prices)
    training_data, test_data = split_data(normalized_prices)
    X_train, y_train = create_dataset(training_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    lstm = create_model()

    lstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    train_predict, test_predict = predictions(lstm, X_train, y_train, X_test, y_test, prices)

    # plot 

    # plot_time_series(train_predict, test_predict, prices, time_step)

    predictedoutput = output_to_list('time_series_modeling/data/ethereum90days.json', test_predict)

    return predictedoutput

def eth_simple_univariate():
    time_step = 10

    prices = json_to_prices('time_series_modeling/data/ethereum90days.json')
    normalized_prices = normalize_prices(prices)
    training_data, test_data = split_data(normalized_prices)
    X_train, y_train = create_dataset(training_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    lstm = create_simple_model()

    lstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    train_predict, test_predict = predictions(lstm, X_train, y_train, X_test, y_test, prices)

    predictedoutput = output_to_list('time_series_modeling/data/ethereum90days.json', test_predict)

    return predictedoutput

def eth_simple_2022_univariate():
    time_step = 10
    prices = json_to_prices('time_series_modeling/data/ethereum90days.json')
    training_data = normalize_prices(prices)
    X_train, y_train = create_dataset(training_data, time_step)

    test_prices = json_to_prices('time_series_modeling/data/ethereum2022test.json')
    test_data = normalize_prices(test_prices)

    X_test, y_test = create_dataset(test_data, time_step)
    lstm = create_model()

    lstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    
    train_predict, test_predict = predictions(lstm, X_train, y_train, X_test, y_test, prices)

    predictedoutput = output_to_list('time_series_modeling/data/ethereum90days.json', test_predict)

    return predictedoutput

def dai_univariate():

    time_step = 10

    prices = json_to_prices('time_series_modeling/data/dai90days.json')
    normalized_prices = normalize_prices(prices)
    training_data, test_data = split_data(normalized_prices)
    X_train, y_train = create_dataset(training_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    lstm = create_model()

    lstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    train_predict, test_predict = predictions(lstm, X_train, y_train, X_test, y_test, prices)

    predictedoutput = output_to_list('time_series_modeling/data/dai90days.json', test_predict)

    return predictedoutput

def usdt_univariate():

# # USDT Univariate LSTM Time Series

    time_step = 10

    prices = json_to_prices('time_series_modeling/data/usdt90days.json')
    normalized_prices = normalize_prices(prices)
    training_data, test_data = split_data(normalized_prices)
    X_train, y_train = create_dataset(training_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    lstm = create_model()

    lstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    train_predict, test_predict = predictions(lstm, X_train, y_train, X_test, y_test, prices)

    predictedoutput = output_to_list('time_series_modeling/data/usdt90days.json', test_predict)

    return predictedoutput

def generalized_eth_model():
    # ## Generalized Ethereum Model
    # 
    # April 27, 2023 - March 27, 2024
    # 
    # 1682571600(april) -  1687842000(june)
    # 
    # 1687842000(june) - 1693112400(august)
    # 
    # 1693112400(august) - october(1698382800)
    # 
    # 1698382800(october) - 	1703656800 (december)
    # 
    # 1703656800 (december) - 1709013600(feb 2024)
    # 
    # 1709013600(feb 2024) - 1711515600(march 2024)

    prices_1 = json_to_prices('time_series_modeling/general_model_data/eth_april_june.json')
    prices_2 = json_to_prices('time_series_modeling/general_model_data/eth_june_august.json')
    prices_3 = json_to_prices('time_series_modeling/general_model_data/eth_august_october.json')
    prices_4 = json_to_prices('time_series_modeling/general_model_data/eth_october_december.json')
    prices_5 = json_to_prices('time_series_modeling/general_model_data/eth_december_feb.json')
    prices_6 = json_to_prices('time_series_modeling/general_model_data/eth_feb_march.json')

    print(prices_1)

    # want to combine all the data into one dataframe
    prices = pd.concat([prices_1, prices_2, prices_3, prices_4, prices_5, prices_6])

    #reset the index
    prices = prices.reset_index(drop=True)
    print(prices)

    time_step = 10
    training_data = normalize_prices(prices)
    X_train, y_train = create_dataset(training_data, time_step)

    test_prices = json_to_prices('time_series_modeling/data/ethereum2022test.json')
    test_data = normalize_prices(test_prices)

    X_test, y_test = create_dataset(test_data, time_step)
    lstm = create_model()

    lstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

    train_predict, test_predict = predictions(lstm, X_train, y_train, X_test, y_test, prices)

    predictedoutput = output_to_list('time_series_modeling/data/ethereum2022test.json', test_predict)

    return predictedoutput
