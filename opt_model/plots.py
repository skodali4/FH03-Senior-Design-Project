import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set_theme()

DATASET = "simdata\Run_15_Steps_3_Horizon.csv"


def make_all_plots(df):
    sns.set_theme()
    plt.figure()
    plt.plot(df['collateral_prices'], color = "darkorange", linewidth = 2)
    plt.title("Collateral Price Process (TODO: Overaly Forecast)")

    fig, axs = plt.subplots(2,1)
    prices = [d/s for d,s in zip(df['supply_series'],df['demand_series'])]
    # print(prices)
    axs[0].plot(df['supply_series'], color = "tab:blue", label = "supply", linewidth = 2)
    axs[0].plot(df['demand_series'], color = "firebrick", label = "demand", linewidth = 2)
    axs[0].legend()
    axs[1].plot(prices)
    axs[1].set_title("Stablecoin Price")
    # df['demand_forecast']

    plt.figure()
    plt.plot(df["rate_trajectory"], color = "magenta", linewidth = 2)
    plt.title("Rate Changes")

    plt.show()

# Main builds all figures, Option to also call 1 by 1
if __name__ == "__main__":
    df = pd.read_csv(DATASET)
    print("Check")
    print(df.head())

    