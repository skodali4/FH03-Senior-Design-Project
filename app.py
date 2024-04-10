from flask import Flask, jsonify, request, json
from time_series_modeling.run_model_predictions import dai_multivariate, dai_univariate, eth_univariate, usdc_univariate, usdt_univariate, print_test
from simulation_code.template import call_deleveraging_library, average_prices
from flask_cors import CORS


app = Flask(__name__)

CORS(app)

@app.route('/')
def basictest():
    result = print_test()
    return jsonify(result)

@app.route('/dai/multi')
def dai_multi():
    result = dai_multivariate()
    return jsonify(result)

@app.route('/dai/uni')
def dai_uni():
    result = dai_univariate()
    return jsonify(result)

@app.route('/eth')
def eth_uni():
    result = eth_univariate()
    return jsonify(result)

@app.route('/usdc')
def usdc_uni():
    result = usdc_univariate()
    return jsonify(result)

@app.route('/usdt')
def usdt_uni():
    result = usdt_univariate()
    return jsonify(result)

@app.route('/simulation', methods = ['GET'])
def main():
    if request.method == 'GET':
        num_sims = int(request.args.get('num_sims'))
        a = float(request.args.get('alpha'))
        b = float(request.args.get('beta'))
        num_eth = int(request.args.get('num_ethereum'))
        num_stbl = int(request.args.get('num_stablecoin'))
    #prices, alphas, betas = call_deleveraging_library(input_n_sims = num_sims, input_alpha = a, input_beta = b, input_n_eth = num_eth, input_n_stbl = num_stbl)
        if num_sims is not None:
            prices, alphas, betas = call_deleveraging_library(input_n_sims = num_sims, input_alpha = a, input_beta = b, input_n_eth = num_eth, input_n_stbl = num_stbl)
        else:
            prices, alphas, betas = call_deleveraging_library(input_n_sims = 3, input_alpha = .1, input_beta = 2, input_n_eth = 400, input_n_stbl = 0)
    # #call_deleveraging_library(input_n_sims = 3, input_alpha = 0.1/-1, input_beta = 2/-1, input_n_eth = 400, input_n_stbl = 0)
        final_prices = average_prices(prices)
    # #prices_json = json.dumps(final_prices)
        out_file = open("sim_output.json", "w")
        final_prices = final_prices.tolist()
        json.dump(final_prices, out_file, indent = 6)
        out_file.close()
        return jsonify(final_prices)
        return jsonify({
         'num_sims': num_sims,
         'alpha': a,
         'beta': b,
         'num_ethereum': num_eth,
         'num_stablecoin': num_stbl
    })
    else:
        return jsonify({
         'num_sims': num_sims,
         'alpha': a,
         'beta': b,
         'num_ethereum': num_eth,
         'num_stablecoin': num_stbl
     })
    

    # return jsonify({
    #     'num_sims': num_sims,
    #     'alpha': a,
    #     'beta': b,
    #     'num_ethereum': num_eth,
    #     'num_stablecoin': num_stbl
    # })

    # print(num_sims)
    # print(a)
    # print(b)
    # print(num_eth)
    # print(num_stbl)

if __name__ == '__main__':
    app.run(debug=True)
