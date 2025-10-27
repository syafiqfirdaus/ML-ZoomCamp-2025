FROM agrigorev/zoomcamp-model:2025

# Install pip
RUN pip install --upgrade pip

# Copy project files
COPY pyproject.toml .
COPY predict_docker.py .

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "predict_docker:app", "--host", "0.0.0.0", "--port", "8000"]
