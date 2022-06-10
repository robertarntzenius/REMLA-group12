"""
Flask API of the SMS Spam detection model model.
"""
#import traceback
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd

import json
import main as predictor

# from text_preprocessing import prepare, _extract_message_len, _text_process

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    predictor.predict()
    with open("reports/bag-of-words-metrics.json", "r") as file:
        prediction = json.load(file)

    return prediction

if __name__ == '__main__':
    # clf = joblib.load('output/model.joblib')
    app.run(host="0.0.0.0", port=8080, debug=True)
