
FROM agrigorev/zoomcamp-model:2025

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY predict_docker.py .

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "predict_docker:app", "--host", "0.0.0.0", "--port", "8000"]
