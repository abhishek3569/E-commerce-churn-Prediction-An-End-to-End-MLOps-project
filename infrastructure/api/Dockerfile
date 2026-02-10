FROM python:3.11-slim

WORKDIR /app

# Build from project root: docker build -f infrastructure/api/Dockerfile .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
