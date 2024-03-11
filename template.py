import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
#import remote DAI simulation
sns.set_theme()

""" {Template Description}
This file is attempting to lay out a simple bare bones design of how we might interface with user input and then
pass data into alkalum's deleveraging code in order to get back a DAI prediction for "T" time steps (requested by user)
Furthermore, this same basic template can be applied for other stablecoins we end up including in the final project
USDC, RAI, and UT Coin with their own set of respective inptus. I've alos begun implementing a few functions partially to 
hopefully give the idea of how it might work (note the functions wont run as is of course)
"""


def get_input(self ):
    """ Some function which communicates/interface with UI team's code or 
    somehow takes direct input from user in meantime for example pygui for a simple test"""

    #import pygui
    #interface with react js etc.
    data = uicall()
    """data = { init_supply, init_eth, init_eth_price, 
                num_agents (later extension? use 1 for now), 
                collateral_ratio(s), requested_time_forecast, 
                num_simulations (parameter controlling how many iterations to run)
            }
    """

    # NOTE: num_simulations -> call the Monte_Carlo.py (or use the repeated sims technique) to generate a distribution of solutions
    ## when plotting we can find some nice functions which takes discrete N time series and blends into continuous dist 
    ## would also be nice to have some confidence intervals -- in other words treat num_sims as "samples" from model ( remember the mode
    ## is stochastic because we add noise to speculator's response and because we use a random variable to simulate ETH price forward movements

    return data

def simulate():

    data = get_user_input()

    ## NOTE: could also be preprocessing to agents i.e. "num_agents" , risk levels
    agent = DaiSpeculator(
            init_supply = data.init_supply,
            init_eth = data.init_eth,   
            init_eth_price =  data.init_eth_price,   
            beta = data.beta,           
            var = data.var
            # and anything i'm forgetting^
    )
    
    solution_dist = []
    for n in range(data.num_simualtions):
        series = agent.solve()
        # simulate mixture of low-information with highly informed
        uninformed = [.2*(np.random.normal(series[i] , 1) for i in range(len(series))]
        informed = [.8*(np.random.normal(series[i] , .1) for i in range(len(series))]
        solution_dist.append(uninformed +informed)
        # or something like this, 20% and 80% are arbitrary ^^
    return solution_dist

def another_simulate()
    " This is just another example of how we might construct the above ^" 

    ## Some Dummy Inputs 
    T = 10  # Time steps in advance when the user 
    DATE = "01/01/2024" # Date migt be used to pull some real starting condtions
    ETH_BANK = 5   # This is the Agent's starting Ethereum amount
    Beta = 2.2     # This is the Agent's Collateral Ratio 
    Supply = 1000  # "pull_dai_value_from( DATE )" -- might be a hypothetical supply 
    Demand_Forecast = []*T  # This is a prediction T steps ahead 

    vars = normal_risk(N)
    betas = normal_betas(N,B)
    shares = randomize_shares(N)
    
    agents = [DaiSpeculator(supply = shares[i]*Supply, beta = betas[i], 
                                    var = vars[i]) for i in range(N)]
    
    for n,agent in enumerate(agents):
        print("\nSpec Agent ", n)
        print("Supply Share:" , agent.S)
        print("Collateral Ratio:", agent.Beta)
        print("Risk Tol: ", agent.Var)

    return



class DaiSpeculator:
""" {Description}  This is a suggested structure to use for wrapping
    calls to the DAI lib (https://github.com/aklamun/stablecoin_deleveraging). TODO: 
    please write / fill in the required variables that are receieved from the user and then
    passed to aklamun code in the necessary places. It might be best to take just the essentials
    from aklamun and place them here at the bottom of the file, organized as modularly as possible. 
    Ideally, we can make simple i/o calls with this single interface

    Note: one difference to be aware of is that the paper solves a single-step problem. So for us to propagate out
    the simulation in time, it should run in loop t = 0 : T and it should update the state at every step before solving 
    the problem again at the next step.
    
    Also, I've labeled the variables below with what I believe they are referred to as in the (In)Stability paper
""" 
    
    def __init__(
            self,
            init_supply,    # "L"
            init_eth,       # "m"
            init_eth_price, # pE --> in deleveraing library this is randomly drawn from distribution at each step
            beta,           # Beta = 1.5
            var             # See "Value at Risk" section (let's use Normal or Heavy-Tailed -> formula is something like exp( alpha  ) etc.
    ):
         
        ## TODO
        self.Beta = beta
        self.S = supply
        self.Var = var

        return
    
    def call_deleveraging_library(self):
        # TODO
        """ basically, connect the dots from our input variables
        to functions in deleveraaging code. The output is the time series
        prediction"""
        deleveraging_prediction = None
        return deleveraging_prediction 

    def solve(self):
        ## Just putting this wrapper "solve" here to make it clear
        # that in this context solving is equivalent to calling the delveraging lib
        return call_deleveraging_library(self)


""" NOTE : here are some functions to give us some ideas how to build "multi-agent" model
          -> it can be a distribution of agents with differing risk tolerances and which act on a scale
               of "informedness". These correspond to (1) solving the front end problem multiple times at different 
               risk-level (var) inputs and adding noise to the solution output according to how "informed" agents are.
               However, I don't know this may just resemble as a single agent acting in the average of these parameters.
"""

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


### Suggestion --> Extend this file with the "Dai Library" code ? --> alternatively, this could be organized into a separate file or group of files
