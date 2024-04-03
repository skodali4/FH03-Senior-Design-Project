from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def main():
    if request.method == 'POST':
        num_sims = request.args.get('num_sims')
        a = request.args.get('alpha')
        b = request.args.get('beta')
        num_eth = request.args.get('num_ethereum')
        num_stbl = request.args.get('num_stablecoin')

    print(num_sims)
    print(a)
    print(b)
    print(num_eth)
    print(num_stbl)

app.run(port=5000)