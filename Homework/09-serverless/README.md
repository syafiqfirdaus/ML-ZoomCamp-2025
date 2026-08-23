# Homework 09 - Local ONNX Deep Learning Inference

This folder contains the solution for Homework 09 of the ML Zoomcamp 2025.

## Files

- `q1_q4_solver.py`: Python script to solve Questions 1-4 (Model inspection, Preprocessing, Local Inference).
- `inference.py`: Shared ONNX image preprocessing and inference code.
- `local_app.py`: Local Flask prediction API.
- `download_model.py`: Downloads the course model artifacts.
- `Dockerfile`: Runs the local API in a container.
- `homework.md`: The homework assignment text.

## Prerequisites

- Python 3.10+
- Docker

## Usage

### Questions 1-4

To run the script that answers Questions 1-4, you need to set up a virtual environment and install the dependencies:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy onnxruntime pillow

# Run the solver script
python3 q1_q4_solver.py
```

This script will:

1. Download the model files.
2. Print the input/output names (Q1).
3. Print the target size (Q2).
4. Print the first pixel R channel value (Q3).
5. Print the model output prediction (Q4).

### Local API

```bash
pip install -r requirements.txt
python download_model.py
python local_app.py
```

Then run `python test_local.py` from another terminal, or POST JSON containing an image URL to `http://localhost:9696/predict`.

### Docker

To build and run the local API container:

1. **Build the image**:

   ```bash
   docker build -t homework09-hair-classifier .
   ```

2. **Run the container**:

   ```bash
   docker run --rm -p 9696:9696 homework09-hair-classifier
   ```

3. **Test the function**:

   Open a new terminal and send a POST request with an image URL:

   ```bash
   curl -X POST "http://localhost:9696/predict" \
   -H "Content-Type: application/json" \
   -d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'
   ```

   The output should be approximately `-0.10` (Q6).

## Notes

- This folder originated as serverless coursework, but its maintained interface is now fully local.
- Model artifacts are downloaded locally and excluded from version control.
