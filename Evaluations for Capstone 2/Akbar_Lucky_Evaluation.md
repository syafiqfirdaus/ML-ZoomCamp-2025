# Evaluation: Akbar Lucky

**Repository:** [https://github.com/bhazheng/whoop-data](https://github.com/bhazheng/whoop-data)

## Scores

| Criteria | Points | Notes |
| :--- | :---: | :--- |
| **Problem description** | 2 / 2 | Clear description of the problem (Whoop recovery prediction), context, and business value in README. |
| **EDA** | 2 / 2 | Extensive EDA described and performed in `notebooks/notebook.ipynb`, including target distribution, correlations, and feature importance. |
| **Model training** | 3 / 3 | Trained 3 models (Logistic Regression, LightGBM, MLP) with hyperparameter tuning. |
| **Exporting notebook to script** | 1 / 1 | Training logic exported to `code/train.py` with a clean pipeline structure. |
| **Reproducibility** | 1 / 1 | Detailed README, Makefile, and Pipfile provided. Clean structure. |
| **Model deployment** | 1 / 1 | Model deployed using Flask (`app/main.py`). |
| **Dependency management** | 2 / 2 | Used `Pipenv` (`Pipfile` & `Pipfile.lock`) and `requirements.txt`. |
| **Containerization** | 2 / 2 | `Dockerfile` provided and usage described. |
| **Cloud deployment** | 2 / 2 | Kubernetes deployment files (`kubernetes/`) provided with instructions. |

**Total Score:** 16 / 16

## Comments

Excellent submission. The project structure is very professional, with a clear separation of concerns (code, app, kubernetes). The README is top-notch, providing comprehensive documentation. The inclusion of a Makefile and `Pipfile` greatly enhances reproducibility. Good job on using Scikit-Learn pipelines and multiple models including a Neural Network.
