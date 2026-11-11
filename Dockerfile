# Multi-stage Dockerfile for Carbon-Aware AutoML Framework
# Stage 1: Dependency builder
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.10-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

COPY config.yaml main.py pyproject.toml requirements.txt ./
COPY carbon_automl/ ./carbon_automl/
COPY datasets/ ./datasets/
COPY experiments/ ./experiments/
COPY models/ ./models/

EXPOSE 8000 8501

CMD ["uvicorn", "carbon_automl.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
