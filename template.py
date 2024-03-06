import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
#import remote DAI simulation
sns.set_theme()


#TODO
class SingleAgentDaiSimulator:
    
    def __init__(
            self,
            beta,
            supply,
    ):
        
        self.Beta = beta # Collateral Ratio
        self.S    = supply
    
    def get_input(self ):
        self.__init__()
        return

    def run():
        #TODO wrapper calls to DAI library
        return

def randomize_shares(N):
    s = np.linspace(-2,2,N+1)
    p = [norm.cdf(sample) for sample in s]
    diff = [p[i+1] - p[i] for i in range(len(p)-1)]
    return diff/np.sum(diff)


def normal_risk(N):
    ## Draw VAR values from normal distribution
    x = np.random.randn(N)
    return x

def normal_betas(N,B = 2):
    return np.random.normal(B, scale=.1,size = N)

class DaiSpeculator:

    #TODO : wrapper for calling DAI Lib agents one by one
    def __init__(
            self,
            supply,
            beta,
            var
    ):
         self.Beta = beta
         self.S = supply
         self.Var = var
