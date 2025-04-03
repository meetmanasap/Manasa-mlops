import os
import pickle
import click
import mlflow
from mlflow import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


#TRACKING_URI = "sqlite:///mlruns.db"
TRACKING_URI = "http://localhost:5050"
EXPERIMENT_NAME = "duration-prediction"
RUN_NAME = "Random Forest Regressor"
client = MlflowClient(TRACKING_URI)

mlflow.tracking.set_tracking_uri(TRACKING_URI)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):
    experiment_details = mlflow.get_experiment_by_name(f"{EXPERIMENT_NAME}")
    if experiment_details:
        experiment_id = experiment_details.experiment_id
        print(f"Experiment ID: {experiment_id}")
    else:
        print(f"The Experiment {EXPERIMENT_NAME} does not exist currently ")
        print(f"Creating new experiment {EXPERIMENT_NAME}...")
        experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME,
                                artifact_location=f"s3://mlflow-artifacts/{EXPERIMENT_NAME}",
                                tags={"version": "v1"})
        print(f"Experiment ID: {experiment_id}")
        
    mlflow.autolog()
    with mlflow.start_run(run_name=RUN_NAME,experiment_id=experiment_id):
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        


if __name__ == '__main__':
    run_train()