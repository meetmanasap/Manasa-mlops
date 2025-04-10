import pickle
import pandas as pd
import calendar
import os
import sys
from dotenv import load_dotenv
load_dotenv()
from minio import Minio
from minio.error import S3Error
import warnings
warnings.filterwarnings('ignore')

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MINIO_ENDPOINT")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("ACCESS_KEY")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("SECRET_KEY")

# Use below setting for local run
endpoint = "127.0.0.1:9000"

# Use below setting for Docker image
# endpoint = os.path.basename(os.getenv("MINIO_ENDPOINT")).split("//")[0]


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')    
    return df

def upload_to_s3(output_file):

    minio_client = Minio(endpoint,
        access_key= os.getenv("ACCESS_KEY"),
        secret_key= os.getenv("SECRET_KEY"),
        secure= False
    )
    source_file = os.path.basename(output_file)
    month = calendar.month_name[int(os.path.splitext(os.path.basename(output_file))[0].split("-")[-1])]
    year = int(os.path.splitext(os.path.basename(output_file))[0].split("-")[-2])
    print(month)
    print(year)

    bucket_name = "batch-predictions"
    destination_file = f"{year}/{month}/{source_file}"

    found = minio_client.bucket_exists(bucket_name)
    if not found:
        minio_client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    minio_client.fput_object(
        bucket_name, destination_file,output_file,
    )
    print(
        source_file, "successfully uploaded as object",
        destination_file, "to bucket", bucket_name,
    )

def main():
    # year = int(input(f"\n Please enter the YEAR for the data in (yyyy) format: "))
    # month = int(input(f"\n Please enter the MONTH for the data in (MM) format: "))
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    print(f"\nCalculating predictions for {calendar.month_name[month]} {year}.\nPlease wait.......")
    try:
        input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    except Exception as e:
        print("Entered year or month is wrong. Please check the data")
        print(e)
    output_file = f'output/batch-{year:04d}-{month:02d}.parquet'
    
    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("Standard Deviation of Predictions:",y_pred.std())
    print("Mean of Predictions:",y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    try:
        upload_to_s3(output_file)
    except S3Error as exc:
        print("error occurred.", exc)

if __name__=="__main__":
    main()