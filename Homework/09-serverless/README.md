# Homework 09 - Serverless Deep Learning

This folder contains the solution for Homework 09 of the ML Zoomcamp 2025.

## Files

- `q1_q4_solver.py`: Python script to solve Questions 1-4 (Model inspection, Preprocessing, Local Inference).
- `lambda_function.py`: AWS Lambda function handler code.
- `Dockerfile`: Configuration to build the Docker image for the Lambda function.
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

### Question 5 & 6 (Docker)

To build and run the Docker container for the Lambda function:

1. **Build the image**:

   ```bash
   docker build -t homework09-hair-classifier .
   ```

   (Note: The base image `agrigorev/model-2025-hairstyle:v1` is approx 608MB, which is the answer to Q5).

2. **Run the container**:

   ```bash
   docker run -it --rm -p 8080:8080 homework09-hair-classifier
   ```

3. **Test the function**:

   Open a new terminal and send a POST request with an image URL:

   ```bash
   curl -XPOST "http://localhost:8080/2015-03-31/functions/function/invocations" \
   -d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'
   ```

   The output should be approximately `-0.10` (Q6).

## Notes

- The Docker image uses `agrigorev/model-2025-hairstyle:v1` as the base, which already includes the model file `hair_classifier_empty.onnx`.
- The `lambda_function.py` includes the logic required to download the image, preprocess it, and run the ONNX model.
