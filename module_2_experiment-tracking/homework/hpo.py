import os
import pickle
import click
import mlflow
from mlflow import MlflowClient
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("ACCESS_KEY")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("SECRET_KEY")

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "duration-prediction-hyperopt"
RUN_NAME = "Random Forest Regressor"
client = MlflowClient(TRACKING_URI)

mlflow.set_tracking_uri(f"{TRACKING_URI}")
mlflow.set_experiment(f"{EXPERIMENT_NAME}")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))    
    def objective(params):                
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            return {'loss': rmse, 'status': STATUS_OK}       

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    trials = Trials()
        
    with mlflow.start_run(run_name=f"{RUN_NAME}") as parent_run:
        best_result = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=trials,
            rstate=rstate
        )
        for trial in trials.trials:
            with mlflow.start_run(run_name=f"Combination{trial['tid']}",nested=True) as child_run:
                mlflow.log_metric("rmse",f"{trial['result']['loss']}")
                mlflow.log_param("max_depth",f"{trial['misc']['vals']['max_depth'][0]}")
                mlflow.log_param("n_estimators",f"{trial['misc']['vals']['n_estimators'][0]}")
                mlflow.log_param("min_samples_split",f"{trial['misc']['vals']['min_samples_split'][0]}")
                mlflow.log_param("min_samples_leaf",f"{trial['misc']['vals']['min_samples_leaf'][0]}")
                mlflow.log_param("random_state",search_space['random_state'])
        
        print(space_eval(search_space, best_result))
        mlflow.log_params(best_result)
        mlflow.log_param("random_state",search_space['random_state'])     
        mlflow.log_metric("rmse",trials.best_trial['result']['loss'])
        

if __name__ == '__main__':
    run_optimization()