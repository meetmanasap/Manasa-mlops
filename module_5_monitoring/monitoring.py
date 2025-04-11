import datetime
import time
import random
import logging
import pandas as pd
import joblib
import psycopg2
from sqlalchemy import create_engine
import os
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric,ColumnSummaryMetric
from dotenv import load_dotenv
load_dotenv()

GRAFANA_PG_USER=os.getenv("GRAFANA_PG_USER")
print(GRAFANA_PG_USER)
GRAFANA_PG_PASSWORD=os.getenv("GRAFANA_PG_PASSWORD")
GRAFANA_PG_DATABASE=os.getenv("GRAFANA_PG_DATABASE")

POSTGRES_USER=os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD")
POSTGRES_DATABASE=os.getenv("POSTGRES_DATABASE")
PG_PORT=os.getenv("PG_PORT")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p') 

SEND_TIMEOUT = 10
rand = random.Random()


create_table_statement = """
drop table if exists evidently_metrics;
create table evidently_metrics(
    timestamp timestamptz,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float,
    fare_amount_quantile float,
    trip_distance_quantile float
);
"""

reference_data = pd.read_parquet("data/reference.parquet")
reference_data = reference_data.astype({'ehail_fee':object})
raw_data = pd.read_parquet("data/green_tripdata_2024-03.parquet")
raw_data = raw_data.astype({'ehail_fee':object})

with open("models/lin_reg.bin", "rb") as f_in:
    model = joblib.load(f_in)

begin = datetime.datetime(2024, 3, 1, 0, 0)
total_days = 31

target = "duration_min"
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None,
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
    ColumnQuantileMetric(column_name="trip_distance", quantile=0.5),
])

def prep_db():
    conn = psycopg2.connect(host="localhost",port=f"{PG_PORT}",user=f"{POSTGRES_USER}",dbname=f"{POSTGRES_DATABASE}",password=f"{POSTGRES_PASSWORD}")
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{}';".format(GRAFANA_PG_DATABASE))
    exists = cursor.fetchone()
    if not exists:
        print("Creating database '{}'".format(GRAFANA_PG_DATABASE))
        cursor.execute("CREATE database {};".format(GRAFANA_PG_DATABASE))
        print("Database '{}' created successfully".format(GRAFANA_PG_DATABASE))
        cursor.execute("CREATE USER {} WITH PASSWORD '{}';".format(GRAFANA_PG_USER,GRAFANA_PG_PASSWORD))
        print("User '{}' created successfully".format(GRAFANA_PG_USER))
        cursor.execute("GRANT ALL PRIVILEGES ON DATABASE {} TO {};".format(GRAFANA_PG_DATABASE,GRAFANA_PG_USER))
        print("Granted Privileges on Database '{}' to '{}' successfully".format(GRAFANA_PG_DATABASE,GRAFANA_PG_USER))
        cursor.execute("ALTER DATABASE {} OWNER TO {};".format(GRAFANA_PG_DATABASE,GRAFANA_PG_USER))
        cursor.execute("GRANT ALL ON SCHEMA public TO {};".format(GRAFANA_PG_USER))
    conn.close()
    conn = psycopg2.connect(host="localhost",port=f"{PG_PORT}",user=f"{GRAFANA_PG_USER}",dbname=f"{GRAFANA_PG_DATABASE}",password=f"{GRAFANA_PG_PASSWORD}")
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(create_table_statement)
    conn.close()

def calculate_metrics_postgresql(curr, i):    
    current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
        (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i+1)))]
    
    # current_data.fillna(0, inplace=True)
    current_data["prediction"] = model.predict(current_data[num_features + cat_features].fillna(0))

    # calculate dict
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    result = report.as_dict()

    # prediction drift
    prediction_drift = result['metrics'][0]['result']['drift_score']
    # number of drifted columns
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    # share of missing values
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    # 0.5 quantile fare_amount
    fare_amount_quantile = float(result['metrics'][3]['result']['current']['value'])
    # 0.5 quantile trip_distance
    trip_distance_quantile = float(result['metrics'][4]['result']['current']['value'])

    print((begin + datetime.timedelta(i)), prediction_drift, num_drifted_columns, share_missing_values, fare_amount_quantile, trip_distance_quantile)
   
    # data = {
    # "timestamp": [begin + datetime.timedelta(i)] ,
    # "prediction_drift": [prediction_drift],
    # "num_drifted_columns": [num_drifted_columns], 
    # "share_missing_values": [share_missing_values],
    # "fare_amount_quantile": [fare_amount_quantile], 
    # "trip_distance_quantile": [trip_distance_quantile]
    # }

    # df = pd.DataFrame(data=data)

    # engine = create_engine('postgresql+psycopg2://grafana:grafana@localhost:5432/grafana')
    # df.to_sql('evidently_metrics', engine,if_exists="append")

    curr.execute(
        "insert into evidently_metrics values(%s, %s, %s, %s, %s, %s)",
         ((begin + datetime.timedelta(i)),float(prediction_drift),float(num_drifted_columns), float(share_missing_values), float(fare_amount_quantile), float(trip_distance_quantile),)
    )
def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg2.connect(host="localhost",port=f"{PG_PORT}",user=f"{GRAFANA_PG_USER}",dbname=f"{GRAFANA_PG_DATABASE}",password=f"{GRAFANA_PG_PASSWORD}") as conn:
        for i in range(0, total_days):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i)
            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")

if __name__=="__main__":
    main()