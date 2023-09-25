import numpy as np
import pickle
import gunicorn
from flask import Flask, request,jsonify

from typing import Optional, List, Any, Dict
from fastapi import FastAPI
from pydantic import Field, BaseModel
from os.path import realpath,join, dirname

FILE_DIR = realpath(dirname(__file__))
MODEL_DIR = join(FILE_DIR, "models")

with open(join(MODEL_DIR, "winequality.pkl"), 'rb') as ins:
    model,encoder = pickle.load(ins)

def prepare_features(attributes):

    data_points =[list(attributes.values())]

    features = np.array(data_points)
    #Add other feature engineering here.
    #'%s_%s' % (ride['PULocationID'], ride['DOLocationID])
    return features 

def predict(features):
    preds = model.predict(features)
    return float(preds[0])

app = Flask('winequality-prediction')


@app.route('/predict', methods=["POST"])
def predict_endpoint():
    wine = request.get_json()
    print(wine)
    features = prepare_features(wine)
    preds = predict(features)
    result = {
        "quality": preds
    }
    return jsonify(result)


if __name__ == "__main__":
    #Flask.run(app= app, port = 8080, log_level= 'info')
    app.run(debug=True, host= '0.0.0.0', port=9696)