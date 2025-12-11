# Homework 10: Kubernetes Deployment

This repository contains the solution for Module 10 of the ML Zoomcamp 2025. The goal is to deploy a lead scoring model using Kubernetes (Kind).

## Prerequisites

- Docker
- Kubernetes CLI (`kubectl`)
- Kind (`kind`)
- Python 3.13+

## Instructions

1. **Setup Environment**:
    Download the model and test files:

    ```bash
    bash setup_hw10.sh
    ```

2. **Build Docker Image**:

    ```bash
    docker build -f Dockerfile_full -t zoomcamp-model:3.13.10-hw10 .
    ```

3. **Local Testing**:
    Run the container and test with `q6_test.py`:

    ```bash
    docker run -it --rm -p 9696:9696 zoomcamp-model:3.13.10-hw10
    python q6_test.py
    ```

4. **Create Cluster**:

    ```bash
    kind create cluster
    ```

5. **Deploy to Kubernetes**:
    Load the image into Kind, then apply the deployment and service:

    ```bash
    kind load docker-image zoomcamp-model:3.13.10-hw10
    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    ```

6. **Test Service**:
    Forward the port and run the test script again:

    ```bash
    kubectl port-forward service/subscription-service 9696:80
    python q6_test.py
    ```
