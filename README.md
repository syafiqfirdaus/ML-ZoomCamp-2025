# ML Zoomcamp 2025

This repository contains my solutions and coursework for the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club.

## 📚 Homework Modules

| Module | Topic | Description | Link |
|--------|-------|-------------|------|
| **01** | **Introduction** | Basics of Machine Learning, environment setup, Pandas, and NumPy. | [View](./Homework/01-intro) |
| **02** | **Regression** | Linear Regression, predicting car prices using the Car Fuel Efficiency dataset. | [View](./Homework/02-regression) |
| **03** | **Classification** | Logistic Regression, lead scoring prediction. | [View](./Homework/03-classification) |
| **04** | **Evaluation** | Evaluation metrics (ROC AUC, F1 Score), Cross-Validation, and hyperparameter tuning. | [View](./Homework/04-evaluation) |
| **05** | **Deployment** | Deploying models with Flask and Docker. | [View](./Homework/05-deployment) |
| **06** | **Decision Trees** | Decision Trees, Random Forest, and XGBoost (Tree-based thinking). | [View](./Homework/06-trees) |
| **08** | **Deep Learning** | Convolutional Neural Networks (CNNs) for hair type classification using TensorFlow/Keras. | [View](./Homework/08-Deep%20Learning) |
| **09** | **Serverless** | Deploying Deep Learning models to AWS Lambda using TFLite and Docker. | [View](./Homework/09-serverless) |
| **10** | **Kubernetes** | Deploying models with Kubernetes, Docker, and Kind. | [View](./Homework/10-kubernetes) |

## 🚀 Midterm Project: Mutual Funds & ETFs Performance Prediction

A comprehensive end-to-end Machine Learning project that predicts the **1-year return** and classifies the **investment quality** of mutual funds and ETFs.

### Key Highlights

- **Problem**: Predicting financial performance and risk ratings.
- **Data**: Kaggle Mutual Funds and ETFs dataset.
- **Tech Stack**:
  - **Models**: XGBoost (Regression & Classification), Random Forest.
  - **Deployment**: Flask web service containerized with Docker.
  - **Analysis**: Feature engineering, extensive EDA, and risk metric evaluation.

[**Explore the Midterm Project**](./Midterm%20Project/README.md)

## 🏆 Capstone Project 2: ASEAN Investment Vehicle Analysis

A deep learning project exploring investment suitability in the ASEAN market (Singapore, Malaysia, Indonesia, etc.).

### Key Highlights

- **Goal**: Predict investment vehicle suitability (Attractive/Unattractive) using historical data.
- **Tech Stack**:
  - **Models**: Neural Networks (TensorFlow/Keras), converted to **ONNX** for optimized inference.
  - **Data**: Real-time/Historical data fetching via `yfinance`.
  - **Deployment**:
    - **Serverless**: AWS Lambda (Containerized, ONNX Runtime).
    - **Kubernetes**: Scalable deployment with Kind.
    - **Cloud**: Live deployment on Render.

[**Explore Capstone Project 2**](./Capstone%202/README.md)
