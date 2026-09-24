FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY dashboard/ dashboard/
COPY configs/ configs/
COPY data/ data/
COPY scripts/ scripts/

RUN mkdir -p data/raw data/processed data/mlflow_artifacts

# Expose FastAPI (8000), Streamlit (8501), and MLflow (5000)
EXPOSE 8000 8501 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health && curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["python", "scripts/entrypoint.py"]
