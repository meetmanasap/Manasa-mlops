## Q1. Notebook

We'll start with the same notebook we ended up with in homework 1.

Run this notebook for the March 2023 data.

What's the standard deviation of the predicted duration for this dataset?

* 1.24
* **6.24**
* 12.28
* 18.28


## Q2. Preparing the output

Like in the course videos, we want to prepare the dataframe with the output. 

First, let's create an artificial `ride_id` column:

```python
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
```

Next, write the ride id and the predictions to a dataframe with results. 

Save it as parquet:

```python
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
```

What's the size of the output file?

* 36M
* 46M
* 56M
* **66M**


## Q3. Creating the scoring script

Now let's turn the notebook into a script. 

Which command you need to execute for that?
 **jupyter nbconvert --to script homework/homework.ipynb**

## Q4. Virtual environment

Now let's put everything into a virtual environment. We'll use pipenv for that.

Install all the required libraries. Pay attention to the Scikit-Learn version: it should be the same as in the starter
notebook.

After installing the libraries, pipenv creates two files: `Pipfile`
and `Pipfile.lock`. The `Pipfile.lock` file keeps the hashes of the
dependencies we use for the virtual env.

What's the first hash for the Scikit-Learn dependency?
 **"sha256:0650e730afb87402baa88afbf31c07b84c98272622aaba002559b614600ca691"**

## Q5. Parametrize the script

Let's now make the script configurable via CLI. We'll create two 
parameters: year and month.

Run the script for April 2023. 

What's the mean predicted duration? 

* 7.29
* **14.29**
* 21.29
* 28.29

## Q6. Docker container 
Now run the script with docker. What's the mean predicted duration
for May 2023? 

* **0.19**
* 7.24
* 14.24
* 21.19


## Bonus1: upload the result to the cloud (Not graded)

Just printing the mean duration inside the docker image 
doesn't seem very practical. Typically, after creating the output 
file, we upload it to the cloud storage.

Modify your code to upload the parquet file to S3/GCS/etc.

**Ans - Added the screenshot of bucket (S3/Minio) where the parquet file is uploaded**