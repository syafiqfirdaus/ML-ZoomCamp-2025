# ML Zoomcamp 2025

This repository contains my solutions and coursework for the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club.

## 🎓 Certificate

I am proud to have successfully completed the ML Zoomcamp 2025!

- **Certificate ID**: `b77e193b1f0a0b3b86dd9c9de5695bf1fd573e35`
- **Official Certificate**: [Download PDF](./Certificate/Syafiq%20Firdaus%20-%20Certs.pdf)
- **Verification URL**: [DataTalks.Club Certificate](https://certificate.datatalks.club/ml-zoomcamp/2025/b77e193b1f0a0b3b86dd9c9de5695bf1fd573e35.pdf)

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
| **09** | **Local ONNX Inference** | Running a deep-learning image classifier through a local Flask API and Docker. | [View](./Homework/09-serverless) |
| **10** | **Kubernetes** | Deploying models with Kubernetes, Docker, and Kind. | [View](./Homework/10-kubernetes) |

## 🚀 Midterm Project: Mutual Funds & ETFs Performance Prediction

A comprehensive end-to-end machine-learning project that estimates the **reported 1-year return** and classifies the **investment quality** of mutual funds and ETFs from a cross-sectional dataset.

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
    - **Local API**: Flask and ONNX Runtime, runnable directly or with Docker.
    - **Kubernetes**: Scalable local deployment with Kind.
    - **Cloud**: Optional deployment on Render.

**🌐 Live API**: [https://capstone2-app.onrender.com](https://capstone2-app.onrender.com)

[**Explore Capstone Project 2**](./Capstone%202/README.md)

## 🔄 August 2026 Maintenance Update

This repository was modernized to keep every maintained inference workflow runnable without AWS Lambda:

- Replaced the Capstone 2 Lambda handler and Lambda container with the local Flask/ONNX API.
- Added direct local and Docker instructions, smoke tests, model-path handling, and stricter API input validation.
- Converted Homework 09 into a locally runnable ONNX inference service while retaining its historical coursework context.
- Corrected preprocessing leakage by fitting imputers and scalers only on training splits in the midterm script and notebook.
- Reframed the midterm regression as cross-sectional return estimation because the dataset does not support genuine forward-return forecasting.
- Removed unsupported performance claims and documented the project limitations more clearly.

## 👥 Contributors

- **Syafiq Firdaus** — project owner, model development, analysis, deployment, and documentation.
- **OpenAI Codex** — AI development collaborator for the August 2026 local-inference migration, preprocessing corrections, tests, and documentation updates.

See the Capstone 2 [contributors file](./Capstone%202/CONTRIBUTORS.md) for additional acknowledgments.
