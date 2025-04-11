FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

LABEL maintainer="Manasa Thirumala"
LABEL description=" Ride Duration Model Batch Prediction Service Docker file"

ARG MINIO_ENDPOINT
ARG ACCESS_KEY
ARG SECRET_KEY

ENV MLFLOW_S3_ENDPOINT_URL=$MINIO_ENDPOINT
ENV AWS_ACCESS_KEY_ID=$ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=$SECRET_KEY

WORKDIR /app

COPY /homework/homework.py /app/homework.py
RUN mkdir -p /app/output
COPY poetry.lock /app
COPY pyproject.toml /app

RUN apt update -y
RUN apt install curl vim -y

RUN pip install --upgrade pip 
RUN python -m pip install --user pipx
RUN python -m pipx ensurepath --force

RUN /root/.local/bin/pipx install poetry==1.8.3

RUN /root/.local/bin/poetry config virtualenvs.create false
RUN /root/.local/bin/poetry install --no-interaction --no-root --sync \
    && rm -rf /root/.cache/pypoetry

ENTRYPOINT ["python", "/app/homework.py"]