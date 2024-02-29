from optimization import NStepLQPeg
from random_process import SimulateDemand , Cryptocurrency
import matplotlib.pyplot as plt
import numpy as np
from optimization import IpSpeculatorResponse as Speculator
import time
from plots import make_all_plots
import argparse 
import pandas as pd
import os

# [ "run" , "plot", "all"]

OPTION = "all"


## pass by cli
parser = argparse.ArgumentParser(description='pass options')
parser.add_argument('-o', '--option', nargs='?', type=str, help='options: run, plot, all', required=False)
args = parser.parse_args()

if (args.option is not None):
    OPTION = args.option

if __name__ == "__main__":

    ## NOTE: these are user-defined starting conditions, 
    #           and are just dummy values for now
    S0 = 250    # market owns 250 DAI
    N0 = 40     # market owns 40 ETH at $15 per.
    pE = 15
    alpha = 1.1
    ETH = Cryptocurrency(p_0 = pE, p_1 = pE, stdv = .05)  # Actual ETH process
    demand_process = SimulateDemand(S0)
    sim_steps = 15
    horizon = 3

    if OPTION =="all":
        GO, PLOT, SAVE = True, True, True
    elif OPTION == "run":
        GO, PLOT, SAVE = True, False, True
    else:
        GO, PLOT, SAVE = False, True, False

    collect = ['collateral_prices', 'supply_series', 'demand_series', 'rate_trajectory']
    colprices, supseries, demseries, rates = [pE],[S0],[S0],[alpha]
    pack = []
    if(GO):
        for t in range(sim_steps):
            # get updated supply information
            t0 = time.time()
            demand_process.set_supply(S0) 
            D_t = demand_process.increment()
            D_pred = np.array([demand_process.forecast(n=horizon)])
            eth_forecast = []
            SYNTH = Cryptocurrency(p_0 = pE, p_1 = pE, stdv = .05) # Simualte Forecast
            for i in range(10):
                SYNTH.next_return() 
                eth_forecast.append(SYNTH.p_1)

            ## NOTE Protocol, Stupid LQ
            protocol = NStepLQPeg(curr_supply = S0, horizon = horizon, forecast = D_pred)
            (x,u) = protocol.solve() 
            delta_ref, d_pred = list(u)[0], list(D_pred)[0]
            
            ## NOTE : Need to add terminal rate calculation ... should be better
            del_net = np.sum(delta_ref)
            pT = eth_forecast[-1]
            alph_T = min(max(.85,  (N0*pT + del_net) / (2*(S0 + del_net)) ), 2.5)

            ## NOTE printout to user-----------------------------
            print("Eth Price Flux : ", pT)
            # print("\n----- Sim Step ",t,"------------")
            # print("Predicted final Ct price : ", pT)
            # print("del_net : " , del_net)
            # print("Target Rate: ", alph_T)

            ## NOTE Speculator, Attempt Intelligence
            market = Speculator(
                    initial_supply = S0,
                    initial_bank = N0,
                    initial_rate = alpha,
                    terminal_rate = alph_T ,
                    horizon = horizon,
                    demand_forecast = d_pred,
                    collateral_forecast = eth_forecast,
                    target_traj = delta_ref,
                    penalty_weight = 2
            )
            ## MPC , take alpha[1] --------------------------------------------------
            data = market.solve()
            alphas , deltas = data[2] , data[3]
            # print(alphas)
            # print(deltas)
            t1 = time.time()
            # print("Horizon {} One Loop Completion Time: {}".format(horizon, round(t1-t0, 3)))

            ### Simulate the Speculator Actual Actions --------------------------------
            ## NOTE : Or Method 2 ( basic market sim)
            # (1) use the delta trajectory already provided 
            # (2) 20% large additive noise , 80% less additive --> 
            delta_informed = [np.random.normal(.8*d, .02*S0) for d in deltas]
            delta_uninform = [np.random.normal(.2*d, .05*S0) for d in deltas]
            ## NOTE : Update and repeat
            delta_sim = delta_informed + delta_uninform
            ETH.next_return()
            S0 , alpha , pE =  S0 + delta_sim[0] , alphas[1], ETH.p_1
            colprices.append(pE) , demseries.append(D_t), supseries.append(S0) 
            rates.append(alpha)

        '''## TODO : Method 1
        # (1) Add noise to eth_forecast
        # (2) Solve speculator problem with .80 informed , .20 uninformed 
        ##     Faster Solving (?) -- rollout 5 steps
        ##     Variable actions (?)'''
        pack = np.array([colprices, supseries, demseries,rates]).T
    
    if (SAVE):

        pack = np.array([colprices, supseries, demseries,rates]).T
        # print(pd.DataFrame(pack, columns = collect).head())
        if not (os.path.exists("simdata/")):
            os.mkdir("simdata/")
        pd.DataFrame(pack, columns = collect).to_csv(
            "simdata/Run_{}_Steps_{}_Horizon.csv".format(sim_steps, horizon))

    if (PLOT):
        df = pd.read_csv("simdata/Run_{}_Steps_{}_Horizon.csv".format(sim_steps, horizon))
        make_all_plots(df)
