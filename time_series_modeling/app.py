from flask import Flask, jsonify
from run_model_predictions import dai_multivariate, dai_univariate, eth_univariate, usdc_univariate, usdt_univariate, print_test

# TODO denormalize output

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
