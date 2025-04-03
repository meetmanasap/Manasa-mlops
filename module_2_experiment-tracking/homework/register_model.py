import os
import pickle
import click
import mlflow
from datetime import datetime as dt

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from dotenv import load_dotenv
load_dotenv()

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("ACCESS_KEY")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("SECRET_KEY")

HPO_EXPERIMENT_NAME = "duration-prediction-hyperopt"
EXPERIMENT_NAME = "duration-prediction-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

TRACKING_URI = "http://localhost:5050"
MODEL_NAME="duration-prediction-model"


mlflow.set_tracking_uri(f"{TRACKING_URI}")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(float(params[param]))
        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        mlflow.log_metric("test_rmse", test_rmse)

        # Log the model
        # mlflow.sklearn.log_model(rf, f"{MODEL_NAME}")

        # Log the dict vectorizer
        mlflow.log_artifact('output/dv.pkl')


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )[0]
    print("best models runs.....", best_run.info.run_id)
    # Register the best model
    MODEL_URI = f"runs:/{best_run.info.run_id}/model"
    print(MODEL_URI)
    mv= mlflow.register_model(MODEL_URI,name="Duration-Prediction-Model")    
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    client.set_registered_model_alias(name=mv.name,version=mv.version,alias="Champion")
    client.update_model_version(name=mv.name,version=mv.version,description=f"The model was promoted to Champion on {dt.today().date()}")


if __name__ == '__main__':
    run_register_model()