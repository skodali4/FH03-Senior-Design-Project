from flask import Flask, request
from template import call_deleveraging_library, average_prices
import json

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def main():
    if request.method == 'POST':
        num_sims = request.args.get('num_sims')
        a = request.args.get('alpha')
        b = request.args.get('beta')
        num_eth = request.args.get('num_ethereum')
        num_stbl = request.args.get('num_stablecoin')
    prices, alphas, betas = call_deleveraging_library(input_n_sims = num_sims, input_alpha = a, input_beta = b, input_n_eth = num_eth, input_n_stbl = num_stbl)
    #call_deleveraging_library(input_n_sims = 3, input_alpha = 0.1/-1, input_beta = 2/-1, input_n_eth = 400, input_n_stbl = 0)
    final_prices = average_prices(prices)
    #prices_json = json.dumps(final_prices)
    out_file = open("sim_output.json", "w")
    final_prices = final_prices.tolist()
    json.dump(final_prices, out_file, indent = 6)
    out_file.close()

    print(num_sims)
    print(a)
    print(b)
    print(num_eth)
    print(num_stbl)

app.run(port=5000)