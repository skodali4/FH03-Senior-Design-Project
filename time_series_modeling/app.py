from flask import Flask, jsonify
from multivariate_lstms_flask_test import dai_multivariate, print_test
from univariate_lstms_flask_test import usdc_univariate, eth_univariate, dai_univariate, usdt_univariate
from univariate_lstms_flask_test import eth_simple_2022_univariate, eth_simple_univariate, generalized_eth_model

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

@app.route('/eth/uni')
def eth_uni():
    result = eth_univariate()
    return jsonify(result)

@app.route('/eth/simple')
def eth_simple():
    result = eth_simple_univariate()
    return jsonify(result)

@app.route('/eth/simple/2022')
def eth_simple22():
    result = eth_simple_2022_univariate()
    return jsonify(result)

@app.route('/eth/generalized')
def eth_gen():
    result = generalized_eth_model()
    return jsonify(result)

@app.route('/usdc/uni')
def usdc_uni():
    result = usdc_univariate()
    return jsonify(result)

@app.route('/usdt/uni')
def usdt_uni():
    result = usdt_univariate()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
