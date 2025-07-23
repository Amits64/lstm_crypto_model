# -------- STAGE 1: Build Stage --------
FROM python:3.9-slim-bullseye AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
 && rm -rf /var/lib/apt/lists/*

# Copy requirement files
COPY requirements.txt .

# Install Python packages into a temp location
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# -------- STAGE 2: Runtime Stage --------
FROM python:3.9-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Ensure model directory exists
RUN mkdir -p model

# Expose app port
EXPOSE 5000

# Entry point
CMD ["python", "app.py"]

