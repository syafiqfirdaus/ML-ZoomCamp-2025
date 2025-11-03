# Multi-stage Dockerfile for Mutual Funds & ETFs Prediction Service
# ML ZoomCamp 2025 - Midterm Project

# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py .
COPY predict.py .
COPY MutualFunds.csv .
COPY ETFs.csv .

# Train models (optional - comment out if you want to train separately)
# RUN python train.py

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=predict.py
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/')" || exit 1

# Run the Flask application
CMD ["python", "predict.py"]
