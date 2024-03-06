
from template import *

DATE = "01/01/2024"
N = 5
B = 2.2
Supply = 1000  # "pull_dai_value_from( DATE )"

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