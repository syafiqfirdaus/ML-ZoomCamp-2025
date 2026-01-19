FROM python:3.11-slim

WORKDIR /app

# Use JSON array syntax for COPY to handle spaces in paths
COPY ["Capstone 2/requirements-k8s.txt", "requirements.txt"]

# Install dependencies (only what's needed for serving to keep it slim)
# We need tensorflow, flask, gunicorn, joblib, numpy
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY ["Capstone 2/best_model.onnx", "."]
COPY ["Capstone 2/scaler.bin", "."]
COPY ["Capstone 2/app.py", "."]

# Standard Render port is 10000, but we can configure it
ENV PORT=10000
EXPOSE 10000

CMD gunicorn --bind 0.0.0.0:10000 app:app
