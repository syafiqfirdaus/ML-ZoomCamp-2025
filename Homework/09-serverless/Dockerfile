FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py inference.py local_app.py ./
RUN python download_model.py

EXPOSE 9696
CMD ["gunicorn", "--bind=0.0.0.0:9696", "local_app:app"]
