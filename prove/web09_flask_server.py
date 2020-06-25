
from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# hello world, metodo GET
@app.route('/', methods=['GET'])
def index():

    response = jsonify({
        'message': 'Ciao Polletto!'
    })

    return response

# hello world, POSTare dati
@app.route('/parrot', methods=['POST'])
def parrot():
    payload = request.json
    print(payload)
    return jsonify({
        'received': payload
    })

@app.route('/sentiment', methods=['POST'])
def sentiment():
    payload = request.json
    print(payload)

    X = [payload['sentence']]
    print(X)

    pred = sentiment_pipeline.predict(X)
    print(pred)

    prob = sentiment_pipeline.predict_proba(X)
    classes   = sentiment_pipeline['model'].classes_
    nice_prob = dict( zip(classes, prob[0].tolist()) )
    print(classes, prob, nice_prob)

    return jsonify({
        'sentence'    : payload['sentence'],
        'predictions' : pred[0].tolist(),
        'probability' : nice_prob
    })

@app.route('/bikes', methods=['POST'])
def bikes():
    payload = request.json
    print(payload)
    X = pd.DataFrame(payload) # se è uno solo, [payload]
    print(X)
    pred = bike_pipeline.predict(X)
    print(pred)
    return jsonify({
        'predictions'   : pred.tolist()
    })

if __name__ == '__main__':

    sentiment_pipeline = load('../sentiment_pipeline.joblib')
    bike_pipeline      = load('../bikes_pipeline.joblib')

    # inizialmente solo run dell'app
    app.run(debug=True, port=2228) # live reloading!