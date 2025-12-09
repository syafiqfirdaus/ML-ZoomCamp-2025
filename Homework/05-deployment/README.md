# Homework 5 Project - ML ZoomCamp 2025

This project contains the implementation for Homework 5 on Model Deployment.

## Project Structure

```
homework5_project/
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Lock file with package hashes
├── pipeline_v1.bin         # Trained model pipeline
├── predict.py              # FastAPI service for pipeline_v1
├── predict_docker.py       # FastAPI service for Docker (uses pipeline_v2)
├── Dockerfile              # Docker configuration
├── test_q3.py             # Test script for Question 3
├── test_q4.py             # Test script for Question 4
└── test_q6.py             # Test script for Question 6
```

## Setup

This project uses `uv` for dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
uv sync
```

## Running the FastAPI Service (Questions 3 & 4)

### Start the service

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 8000
```

### Test Question 3

```bash
uv run python test_q3.py
```

### Test Question 4

```bash
uv run python test_q4.py
```

## Docker Deployment (Questions 5 & 6)

### Pull the base image

```bash
docker pull agrigorev/zoomcamp-model:2025
```

### Check image size

```bash
docker images agrigorev/zoomcamp-model:2025
```

### Build the Docker image

```bash
docker build -t homework5-model .
```

### Run the container

```bash
docker run -d -p 8000:8000 --name homework5-container homework5-model
```

### Test Question 6

```bash
uv run python test_q6.py
```

### Stop and remove the container

```bash
docker stop homework5-container
docker rm homework5-container
```

## Answers

| Question | Answer |
|----------|--------|
| Q1: uv version | uv 0.9.5 |
| Q2: First scikit-learn hash | sha256:b4fc2525eca2c69a59260f583c56a7557c6ccdf8deafdba6e060f94c1c59738e |
| Q3: Probability (pipeline_v1) | 0.534 → Select **0.533** |
| Q4: Probability (FastAPI) | 0.534 → Select **0.534** |
| Q5: Docker image size | 121 MB → Select **121 MB** |
| Q6: Probability (Docker/pipeline_v2) | 0.99 → Select **0.99** |

## Notes

- Questions 3 and 4 use `pipeline_v1.bin` and produce the same result (0.534)
- Question 6 uses `pipeline_v2.bin` from the Docker base image, which is a different model and produces a different result (0.99)
