#!/bin/bash
set -e

# Setup bin dir
mkdir -p $HOME/.gemini/bin
export PATH=$HOME/.gemini/bin:$PATH

echo "Installing tools..."

# Install kubectl
if ! command -v kubectl &> /dev/null; then
    echo "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    mv kubectl $HOME/.gemini/bin/
else
    echo "kubectl already installed"
fi

# Install kind
if ! command -v kind &> /dev/null; then
    echo "Installing kind..."
    curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64
    chmod +x kind
    mv kind $HOME/.gemini/bin/
else
    echo "kind already installed"
fi

# Download files
BASE_URL="https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2025/05-deployment/homework"
FILES="pipeline_v2.bin q6_predict.py pyproject.toml uv.lock .python-version"

echo "Downloading files..."
for file in $FILES; do
    if [ ! -f "$file" ]; then
        echo "Downloading $file..."
        wget -q "$BASE_URL/$file" || echo "Failed to download $file"
    else
        echo "$file already exists"
    fi
done

echo "Environment check:"
echo "Docker:"
docker --version || echo "Docker not found"
echo "Kubectl:"
kubectl version --client
echo "Kind:"
kind --version

