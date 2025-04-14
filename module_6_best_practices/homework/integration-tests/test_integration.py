import os
import pandas as pd
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("ACCESS_KEY")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("SECRET_KEY")

# Use below setting for local run
S3_ENDPOINT_URL = "http://127.0.0.1:9000"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Use below setting for Docker image
# os.environ["S3_ENDPOINT_URL"] = os.path.basename(os.getenv("MINIO_ENDPOINT")).split("//")[0]

os.environ["INPUT_FILE_PATTERN"] = "s3://best-practices/predictions_{month:02d}_{year:04d}.parquet"
os.environ["OUTPUT_FILE_PATTERN"] = "s3://best-practices/test_prediction_{month:02d}_{year:04d}.parquet"

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ["PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime"]
df_input = pd.DataFrame(data, columns=columns)
year = 2023
month = 1 
INPUT_FILE = os.environ["INPUT_FILE_PATTERN"].format(year=year, month=month)
OUTPUT_FILE = os.environ["OUTPUT_FILE_PATTERN"].format(year=year, month=month)

options = {
    "client_kwargs": {
        "endpoint_url": S3_ENDPOINT_URL,
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
        "verify": False,
    },
}

df_input.to_parquet(
    INPUT_FILE,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

os.system(f"cd .. && python batch.py --month {month} --year {year}")
prediction_df = pd.read_parquet(OUTPUT_FILE,engine='pyarrow',storage_options=options)

prediction_sum = prediction_df.predicted_duration.sum()
print("Sum of predicted Durations is:", prediction_sum)

expected_sum = 36.28

assert abs(prediction_sum-expected_sum)<1e-2