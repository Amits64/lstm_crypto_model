# -------- STAGE 1: Build Stage --------
FROM python:3.9-slim-buster AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Set working directory
WORKDIR $APP_HOME

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files
COPY requirements.txt .

# Install Python packages into /install
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# -------- STAGE 2: Runtime Stage --------
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Set working directory
WORKDIR $APP_HOME

# Copy only installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p model

# Expose the application port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

