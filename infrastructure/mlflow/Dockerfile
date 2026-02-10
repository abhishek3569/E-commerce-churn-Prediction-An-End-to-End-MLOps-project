FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow psycopg2-binary boto3

ENV MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
ENV MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "/mlflow/artifacts"]
