import pandas as pd
from flask import Flask,jsonify,request
import mlflow
import os
import pickle
from mlflow import MlflowClient
from dotenv import load_dotenv
load_dotenv()

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("ACCESS_KEY")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("SECRET_KEY")

# Use below setting for local run
# MLFLOW_TRACKING_URI="http://0.0.0.0:5000"

# Use below setting for Docker image
MLFLOW_TRACKING_URI="http://mlflow_server:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(MLFLOW_TRACKING_URI)

MODEL_NAME = "Duration-Prediction-Model"
MODEL_VERSION_ALIAS = "Champion"

# ENDPOINT URL
ENDPOINT_HOST = "0.0.0.0"
ENDPOINT_PORT = "9090"
ENDPOINT_URL=f"http://{ENDPOINT_HOST}:{ENDPOINT_PORT}/predict"

def load_model_and_dv():
    champion_model_uri = f"models:/{MODEL_NAME}@{MODEL_VERSION_ALIAS}"
    champion_model = mlflow.pyfunc.load_model(champion_model_uri)
    RUN_ID=champion_model._model_meta.run_id
    dv_path = client.download_artifacts(RUN_ID,"dv.pkl",dst_path=".")
    print(f"Downloading the dict vectorizer to {dv_path}")
    with open(dv_path,'rb') as file:
        dv = pickle.load(file)
    return champion_model,dv

def prepare_features(data):
    features = {}
    features['PU_DO'] = '%s_%s' % (data['PULocationID'],data['DOLocationID'])
    features['trip_distance'] = data['trip_distance']
    return features

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    features = prepare_features(data)
    model,dv = load_model_and_dv()
    X = dv.transform(features)
    predictions = model.predict(X)
    result = {
        "Duration": predictions.tolist()
    }
    return jsonify(result)
    
if __name__== '__main__':
    app.run(debug="true",host=f"{ENDPOINT_HOST}",port=f"{ENDPOINT_PORT}")