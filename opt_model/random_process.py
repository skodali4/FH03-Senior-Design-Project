import numpy as np
import matplotlib.pyplot as plt
import random
from statsmodels.tsa.arima.model import *
from statsmodels.tsa.statespace import sarimax
import itertools
import pandas as pd
import warnings
from statsmodels.graphics.tsaplots import plot_predict

warnings.filterwarnings("ignore")


## {Description} 
'''   
    NOTE:
        - Uses some unrealistic expectation on the Demand process
        (i.e. that it is following supply). We should think about best way to simulate demand
        forward in time ( or maybe it should just be additive noise with our forecast model)

        -  I'm also using AKM (In)Stability Paper's model of a cryptocurrency. This is probebly fine for now

    TODO:  
        This file needs a lot of work. 

        (a) Can we build a simple Arima forecaster? Please feel free to blow away the code I have now
        (b) We need to research a realistic (simple) model for demand forecaster
        (c) Add Monte Carlo Simulation Code ( or could be separate file)
        (d) Think about other ways to simulate cryptocurrencies
        (e) Maybe scale up to Multi-Collateral Version
'''


## Contains (poorly fit) shell ARIMA model
## Contains Deterministic Demand Process which depends on Noise
#   Demand Process with additive noise will serve as the initial "forecast"

## Notes from (In)Stability of ... stablecoin_deleveraging repo : https://github.com/aklamun/stablecoin_deleveraging/blob/master/simulation_code/optimization_algos.py
# Borrow Speculator Model :
# BTC process : use the t-distribution idea

class SimulateDemand():

    def __init__(self, init):
        self.curr_supply = init
        self.curr_demand = init
        self.curr_time = 0

    def set_supply(self, S):
        self.curr_supply = S 
    
    def increment(self, alpha = 0.5):
        delta = (self.curr_supply  - self.curr_demand)
        self.curr_demand = self.curr_demand  + alpha*delta + 0.3*np.random.randn()
                            # + np.sin(.25*self.curr_time)  
        
        self.curr_time += 1
        return self.curr_demand 

    def forecast(self, n = 1, sigma = 0.3):
        # fake firecast for now
        sigma = 1
        d = self.curr_demand
        D = [d]
        for i in range(n):
            d += sigma*np.random.randn()
            D.append(d)
        return D

    def poisson_shock(n, sigma, p =.99):
        uni = np.random.rand(n)
        vec = (uni > p) * sigma * np.random.randn(n)
        return vec

##### Lines 59 to 90 credit to aklamun's repo ##########################

class Asset(object):
    ''' Attributes:
    p_0 = price at t-1
    p_1 = price at t
    '''
    
    def __init__(self, p_0=None, p_1=None, eta=None, collat=0, beta=0):
        self.p_0 = p_0
        self.p_1 = p_1


class Cryptocurrency(Asset):
    '''Attributes:
    df = degrees of freedom for t-distribution
    stdv = standard deviation
    drift
    '''
    def __init__(self, p_0=None, p_1=None, df=3, stdv=1., drift=0.):
        super().__init__(p_0, p_1)
        self.df = df
        self.stdv = stdv
        self.drift = drift
    
    def next_return(self):
        '''simulate multiplicative return on asset price'''
        assert self.df>2
        scale = self.stdv/np.sqrt(self.df/(self.df-2))
        ret = np.exp(scale*np.random.standard_t(self.df) + self.drift)
        self.p_0 = self.p_1
        self.p_1 = ret*self.p_0
        return

######################################################################

class ArimaForecaster:
    ''' For now this is just some made up thing
    moving forward, want to develop a more realistic model'''

    def __init__(self, init_length = 200, plot_init = False):
        self.Demand = []
        self.Supply = []
        self.Price = []
        self.S_t = 0
        self.D_t = 0
        self.curr_time = 0
        self.fit_length = init_length
        ## Noisy Demand Process:
        alpha, mu = .5 , 0.1*(self.S_t)

        ## Some Demand Init Process -- Chase the Supply -- supply oscillates/expands slowly
        self.init_test_supply_demand_curves(init_length , plot = plot_init)
        self.Demand = pd.Series(self.Demand)
        # interchangeable
        # self.model =  ARIMA(self.Demand, order = (4,2,0), 
        #                         enforce_stationarity=False, 
        #                         enforce_invertibility=False)
        # self.forecast = self.model.fit()

    def look_one_step_forward(self):
        # print(self.Demand[-2:])
        self.forecast =  ARIMA(self.Demand[-self.fit_length + 1:], order = (4,2,0), 
                                enforce_stationarity=False, 
                                enforce_invertibility=False).fit()
        prediction = self.forecast.predict( self.curr_time )
        return prediction

    def look_n_steps_forward(self, n =5):
        prediction = self.forecast.predict( self.curr_time,self.curr_time + n -1)
        return prediction
    
    def increment_supply(self):
        self.S_t = np.sin(.15*self.curr_time) + min(3*np.sqrt(self.curr_time), 1000)
        self.Supply.append(self.S_t)
        return self.S_t

    def increment_step(self):
        self.curr_time += 1
    
    def increment_demand(self):
        alpha = 0.5
        self.D_t = self.D_t + alpha*(self.S_t - self.D_t) + np.sin(alpha*self.curr_time) 
        self.D_t = self.D_t + np.random.randn()
        self.Demand.loc[self.curr_time] = self.D_t
        return self.D_t

    # def get_forecast(self, stop, results start = self.curr_time):
    #     forecast = results.predict(start,stop) 
    #     return forecast

    # def fit_model(self, which = "arima420", plot = True):
    #     mod = self.model_choices[which]
    #     self.model = mod.fit()

    def init_test_supply_demand_curves(self, length , supply_curve = "sinusoidal growth", 
                                                        alpha = .5, plot = True):
        valid = {"linear", "sinusoidal", "square root", \
                            "logarithmic", "sinusoidal growth"}
        if supply_curve not in valid:
            raise ValueError("results: status must be one of %r." % valid)
        if supply_curve == "linear":
            f = lambda x,t: x+t
        elif supply_curve == "sinusoidal":
            f = lambda x,t: np.sin(t) + x
        elif (supply_curve == "sinusoidal growth"):
            f = lambda x,t: np.sin(.15*t) + min(3*np.sqrt(t), x) # X represents the target maximum
        else:
            f = lambda t : np.sqrt(t)
        ## Model some initial growth is sqrt
        self.Supply = [f(1000,i) for i in range(length) ]
        mu = 0
        for i in range(length): 
            self.Demand.append( mu + np.random.randn() )
            mu = mu + alpha*(self.Supply[i] - mu) + np.sin(alpha*i) 
            self.Price.append(self.Demand[i]/self.Supply[i])
        t = np.arange(0,length,1)

        if (plot):
            plt.plot(t, self.Supply)
            plt.plot(t, self.Demand)
            plt.show()
        
        self.D_t = self.Demand[-1]
        self.S_t = self.Supply[-1]
        self.curr_time = length
        return


    def plot_price(self):
        
        fig, ax = plt.subplots()
        t = np.arange(0,100)
        # plt.plot(t,D_t)
        # plt.plot(t,Supply) 
        plt.plot(t, self.Price)
        plt.show()
        return
    
    def tune_arima(self):
        steps = [10,50,80]
        p = d = q = range(5)
        pdq = list(itertools.product(p , d, q))
        grid = []
        for step in steps:
            series = self.Demand[-step-1 :-1]
            for comb in pdq:
                model = ARIMA(series, order = comb, 
                              enforce_stationarity=False, enforce_invertibility=False)
                out = model.fit()
                grid.append([step, comb, out.aic])
        df = pd.DataFrame(grid)
        df.to_csv("grid_results.csv")
        return

    
## Now I have it such that it fits and generates synthetic data in one go
# forecaster = ArimaForecaster( init_length = 200) # init_length gives arima a chance to do a little better
# predictions = [0]*200
# forecast = forecaster.look_one_step_forward()
# demand = forecaster.Demand

# for i in range(100):
#     forecaster.increment_step()
#     demand.loc[forecaster.curr_time] = forecaster.increment_demand()
#     forecaster.increment_supply()
#     data = forecaster.look_one_step_forward().loc[forecaster.curr_time]
#     predictions.append(data)

# predicitons = pd.Series(predictions)
# print(type(predictions))
# print(type(demand))
# print(len(predictions))
# print(len(demand))

# # arima.plot_predict(dynamic = False)
# fig, ax = plt.subplots()
# # plt.plot(predictions)
# ax.plot(demand)
# ax.plot(forecaster.Supply)
# ax.plot(predicitons, color = "orange")
# # plt.plot(forecaster.Demand)
# # plot_predict(forecaster.forecast, ax = ax, color = 'red')
# ax.set_ylim( 0, 200)
# plt.show()





### Some grid search bs -------
# forecast.tune_arima()
# tune = pd.read_csv("grid_results.csv")
# tune.drop(tune.columns[[0,1]],axis = 1, inplace = True)
# print(tune.head())
# tune1 = tune.iloc[1:126].reset_index().drop('index', axis = 1)
# # tune2 = tune.iloc[126:241].reset_index().drop('index', axis = 1)
# # tune3 = tune.iloc[241:].reset_index().drop('index', axis = 1)
# # print(tune1["1"].iloc[tune1["2"].idxmin()])
# # print(tune2["1"].iloc[tune2["2"].idxmin()])
# # print(tune3["1"].iloc[tune3["2"].idxmin()])
# params = tune1["1"].iloc[tune1["2"].idxmin()]
# series1 = forecast.Demand[-11 :-1]
