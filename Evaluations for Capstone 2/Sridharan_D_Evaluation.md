# Evaluation: Sridharan D

**Repository:** [https://github.com/sridharandevcom/ml-zoomcamp-capstone-2-project](https://github.com/sridharandevcom/ml-zoomcamp-capstone-2-project)

## Scores

| Criteria | Points | Notes |
| :--- | :---: | :--- |
| **Problem description** | 2 / 2 | Clear description of the rice image classification problem and dataset. |
| **EDA** | 0 / 2 | `notebooks/01_rice_transfer_learning.ipynb` is referenced in README but missing from the repository. |
| **Model training** | 0 / 3 | Training code is missing (notebook missing, no training script). Model file `rice_resnet18.pth` exists, proving a model was trained, but the code isn't there to evaluate. |
| **Exporting notebook to script** | 0 / 1 | No training script found. |
| **Reproducibility** | 1 / 1 | Deployment instructions are clear and reproducible (Docker, requirements.txt). |
| **Model deployment** | 1 / 1 | Deployed using FastAPI (`app.py`). |
| **Dependency management** | 1 / 2 | `requirements.txt` provided. |
| **Containerization** | 2 / 2 | `Dockerfile` provided. Detailed explanation of Azure ACR usage. |
| **Cloud deployment** | 2 / 2 | Deployed to Azure Container Apps. Screenshots and commands provided in README. |

**Total Score:** 9 / 16

## Comments

The project demonstrates strong engineering skills regarding deployment (Docker, Azure Container Apps) and documentation of the deployment process is excellent. However, the core machine learning part (EDA, training code) is missing from the repository (`notebooks/` folder is missing), which significantly impacted the score. The `README` implies these files should exist.
