import mlflow

TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "yellow-taxi-duration"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)