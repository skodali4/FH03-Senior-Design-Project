
from template import *

## Some Example Inputs
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


##TODO Cynthia, Aditya -- you guys work on acceptint user defined and passing to alkalum "library"
