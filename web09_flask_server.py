
from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return '<h1>Ciao Polletti</h1>'

@app.route('/json', methods=['GET'])
def json():
    return jsonify({
        'messaggio': 'Ciao polletti'
    })

@app.route('/parrot', methods=['POST'])
def parrot():

    body = request.json
    print(body)

    response = jsonify({
        'parrot': body
    })

    return response

@app.route('/sentiment', methods=['POST'])
def sentiment():

    # leggi payload
    payload = request.json
    X = [payload['sentence']]

    # utilizza modello per la predizione
    pred = sentiment_pipeline.predict(X)
    pred_proba = sentiment_pipeline.predict_proba(X)
    classes    = sentiment_pipeline['model'].classes_
    nice_prob  = dict( zip(classes, pred_proba[0].tolist()) )

    # invia al client la riposta
    return jsonify({
        'sentence': payload['sentence'],
        'prediction': pred[0],
        'probability': nice_prob
    })

@app.route('/bikes', methods=['POST'])
def bikes():
    payload = request.json
    print(payload)
    X = pd.DataFrame(payload)

    pred = bikes_pipeline.predict(X)

    return jsonify({
        'prediction': pred.tolist()
    })

if __name__ == '__main__':

    sentiment_pipeline = load('sentiment_pipeline.joblib')
    bikes_pipeline = load('bikes_pipeline.joblib')

    app.run(debug=True, port=2228)
















