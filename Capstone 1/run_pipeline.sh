#!/bin/bash
set -e

# Activate venv implicitly by using full path or assume source
# Using full path to python in venv
PYTHON=".venv/bin/python"

echo "=========================================="
echo "Step 1: Fetching Data"
echo "=========================================="
$PYTHON fetch_data.py

echo "=========================================="
echo "Step 2: Preprocessing Data"
echo "=========================================="
$PYTHON preprocess.py

echo "=========================================="
echo "Step 3: Training Model"
echo "=========================================="
$PYTHON train.py

echo "=========================================="
echo "Pipeline Completed Successfully"
echo "=========================================="
