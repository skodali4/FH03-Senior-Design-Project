import pickle as pickle
import pandas as pd
import numpy as np
import pandas as pd  
import json

def output_to_list(preds, timestamp):
    latest_timestamp = timestamp
    print(latest_timestamp)
    
    num_predictions = len(preds)
    predictions_timestamps = [latest_timestamp + (3648365 * i) for i in range(1, num_predictions+1)]
    
    print(predictions_timestamps)
    print(type(preds))
    
    predictions_df = pd.DataFrame({'timestamp': predictions_timestamps,'predicted price': preds.tolist()})
    #print(predictions_df.to_dict(orient="records"))
    return predictions_df.to_dict(orient="records")

def dai_multivariate():
    with open("time_series_modeling/test_data/dai_multi_test.json") as f:
        X_test = json.load(f)

    standard_dev = X_test[0][0][0]
    mean = X_test[1][1][1]
    time_stamp = X_test[2][2][2]
    
    X_test = X_test[3:]

    X_test = np.array(X_test)
    pickled_model = pickle.load(open('time_series_modeling/models/dai_multivariate_model.pkl', 'rb'))
    predictions = pickled_model.predict(X_test)

    predicted_price_norm = (predictions * standard_dev) + mean
    return output_to_list(predicted_price_norm, time_stamp)

def dai_univariate():
    with open("time_series_modeling/test_data/dai_test.json") as f:
        X_test = json.load(f)

    standard_dev = X_test[0][0][0]
    mean = X_test[1][1][1]
    time_stamp = X_test[2][2][2]
    
    X_test = X_test[3:]

    X_test = np.array(X_test)
    pickled_model = pickle.load(open('time_series_modeling/models/dai_price_model.pkl', 'rb'))
    predictions = pickled_model.predict(X_test)

    predicted_price_norm = (predictions * standard_dev) + mean
    return output_to_list(predicted_price_norm, time_stamp)

def eth_univariate():
    with open("time_series_modeling/test_data/eth_test_data.json") as f:
        X_test = json.load(f)

    standard_dev = X_test[0][0][0]
    mean = X_test[1][1][1]
    time_stamp = X_test[2][2][2]
    
    X_test = X_test[3:]

    X_test = np.array(X_test)
    pickled_model = pickle.load(open('time_series_modeling/models/ethereum_model.pkl', 'rb'))
    predictions = pickled_model.predict(X_test)

    predicted_price_norm = (predictions * standard_dev) + mean
    return output_to_list(predicted_price_norm, time_stamp)

def usdc_univariate():
    with open("time_series_modeling/test_data/usdc_test.json") as f:
        X_test = json.load(f)

    standard_dev = X_test[0][0][0]
    mean = X_test[1][1][1]
    time_stamp = X_test[2][2][2]
    
    X_test = X_test[3:]

    X_test = np.array(X_test)
    pickled_model = pickle.load(open('time_series_modeling/models/usdc_model.pkl', 'rb'))
    predictions = pickled_model.predict(X_test)

    predicted_price_norm = (predictions * standard_dev) + mean
    return output_to_list(predicted_price_norm, time_stamp)

def usdt_univariate():
    with open("time_series_modeling/test_data/usdt_test.json") as f:
        X_test = json.load(f)

    standard_dev = X_test[0][0][0]
    mean = X_test[1][1][1]
    time_stamp = X_test[2][2][2]
    
    X_test = X_test[3:]

    X_test = np.array(X_test)
    pickled_model = pickle.load(open('time_series_modeling/models/usdt_model.pkl', 'rb'))
    predictions = pickled_model.predict(X_test)

    predicted_price_norm = (predictions * standard_dev) + mean
    return output_to_list(predicted_price_norm, time_stamp)


def print_test():
    test_output_test = { 
    'message' : 'imagine this is the test output, for testing whether this python file works'
    }
    return test_output_test

