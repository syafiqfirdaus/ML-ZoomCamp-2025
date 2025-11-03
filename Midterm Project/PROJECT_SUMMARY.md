# Project Summary - ML ZoomCamp 2025 Midterm Project

## Project Overview

This project implements a comprehensive machine learning solution for analyzing and predicting mutual fund and ETF performance. It addresses **four distinct ML problems** using a dataset of over 26,000 investment funds with 236 features.

## Problems Solved

### 1. Regression: 1-Year Return Prediction
- **Goal:** Predict expected 1-year returns
- **Model:** XGBoost Regressor
- **Metrics:** RMSE, MAE, R²
- **Use Case:** Help investors estimate future fund performance

### 2. Binary Classification: Investment Quality
- **Goal:** Classify funds as "good" (rating ≥4) or "poor" (rating <4)
- **Model:** XGBoost Classifier
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Use Case:** Provide simple buy/avoid recommendations

### 3. Multi-class Classification: Risk Rating
- **Goal:** Predict Morningstar risk rating (Low, Average, High)
- **Model:** XGBoost Classifier
- **Metrics:** Accuracy, Macro F1-Score
- **Use Case:** Help investors understand risk profiles

### 4. Fund Recommendation System
- **Goal:** Find similar funds based on characteristics
- **Approach:** Feature-based similarity matching
- **Use Case:** Provide investment alternatives

## Project Structure

```
Midterm Project/
├── README.md                 # Main documentation
├── DEPLOYMENT.md            # Deployment guide
├── PROJECT_SUMMARY.md       # This file
├── Deliverables.md          # Original requirements
├── notebook.ipynb           # Complete ML workflow
├── train.py                 # Model training script
├── predict.py               # Flask prediction service
├── test_service.py          # API testing script
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── .gitignore              # Git ignore rules
├── .dockerignore           # Docker ignore rules
├── MutualFunds.csv         # Mutual funds dataset (23,784 funds)
├── ETFs.csv                # ETFs dataset (2,311 funds)
├── Dataset issues/         # Known data quality issues
└── models/                 # Trained models (created by train.py)
    ├── regression_model.pkl
    ├── binary_classifier.pkl
    ├── multiclass_classifier.pkl
    ├── *_scaler.pkl
    ├── label_encoder.pkl
    ├── imputer.pkl
    └── feature_names.pkl
```

## Deliverables Checklist

✅ **README.md**
- Problem description
- Dataset information
- Instructions to run the project
- API documentation

✅ **Data**
- MutualFunds.csv (73MB, 23,784 records)
- ETFs.csv (3.7MB, 2,311 records)
- Clear documentation of data sources

✅ **Notebook (notebook.ipynb)**
- Data preparation and cleaning
- Exploratory Data Analysis (EDA)
- Feature importance analysis
- Model selection and parameter tuning
- Comprehensive visualizations

✅ **Training Script (train.py)**
- Trains final XGBoost models
- Saves models to pickle files
- Outputs performance metrics

✅ **Prediction Script (predict.py)**
- Flask web service
- Multiple prediction endpoints
- RESTful API design
- Input validation

✅ **Dependencies**
- requirements.txt with all libraries
- Python 3.8+ compatible

✅ **Dockerfile**
- Production-ready container
- Health check included
- Port 5000 exposed

✅ **Deployment**
- Local deployment instructions
- Docker deployment guide
- Cloud deployment options (AWS, GCP, Azure)
- Production best practices

## Technical Highlights

### Data Processing
- Handled missing values (50%+ missing in many features)
- Feature imputation using median strategy
- StandardScaler for feature normalization
- Combined mutual funds and ETFs datasets

### Feature Engineering
- Selected 35 most relevant features
- Created binary target from ratings
- Numerical features focus (categorical encoding available)
- Feature importance analysis included

### Model Training
- Compared multiple algorithms (Linear, Random Forest, XGBoost)
- XGBoost selected as best performer across all tasks
- Train/validation/test splits (60/20/20)
- Cross-validation for robust estimates

### Web Service
- RESTful API with Flask
- 6 endpoints (health, 3 predictions, all predictions, features list)
- JSON input/output
- Comprehensive error handling
- Production-ready with Gunicorn support

### Containerization
- Multi-stage Dockerfile
- Optimized image size
- Health checks
- Volume mounting for models

## Usage Examples

### Train Models
```bash
python train.py
```

### Start Service
```bash
python predict.py
```

### Test API
```bash
# Health check
curl http://localhost:5000/

# Predict 1-year return
curl -X POST http://localhost:5000/predict/regression \
  -H "Content-Type: application/json" \
  -d '{"fund_return_3years": 0.15, "fund_beta_3years": 1.05}'
```

### Docker Deployment
```bash
# Build
docker build -t fund-predictor .

# Train models
docker run --rm -v $(pwd)/models:/app/models fund-predictor python train.py

# Run service
docker run -d -p 5000:5000 -v $(pwd)/models:/app/models fund-predictor
```

## Performance Metrics

*Note: Run `train.py` to generate actual metrics*

Expected performance:
- **Regression:** R² > 0.6, RMSE < 0.15
- **Binary Classification:** F1-Score > 0.75, ROC-AUC > 0.80
- **Multi-class:** Macro F1-Score > 0.65, Accuracy > 0.70

## Key Features

1. **Comprehensive Analysis:** 4 different ML problems solved
2. **Production-Ready:** Flask API with Docker support
3. **Well-Documented:** README, DEPLOYMENT, API docs
4. **Tested:** Test script included
5. **Scalable:** Cloud deployment guides provided
6. **Maintainable:** Clean code structure, version control

## Technology Stack

- **Python:** 3.8+
- **ML Libraries:** scikit-learn, XGBoost, pandas, numpy
- **Web Framework:** Flask
- **Visualization:** matplotlib, seaborn
- **Containerization:** Docker
- **Development:** Jupyter, VS Code

## Dataset Information

**Source:** [Kaggle - Mutual Funds and ETFs](https://www.kaggle.com/datasets/stefanoleone992/mutual-funds-and-etfs)

**Size:**
- 26,095 total funds
- 236 features per fund
- ~2GB total data

**Key Features:**
- Financial metrics (returns, ratios, assets)
- Risk metrics (alpha, beta, Sharpe ratio)
- Asset allocation (stocks, bonds, cash)
- Sector exposure (10 sectors)
- ESG scores (environment, social, governance)
- Morningstar ratings (overall, risk, return)

## Future Enhancements

- [ ] Incorporate time-series analysis using price history
- [ ] Add SHAP values for model explainability
- [ ] Implement model monitoring and drift detection
- [ ] Build interactive Streamlit dashboard
- [ ] Add real-time data integration
- [ ] Create portfolio optimization module
- [ ] Implement A/B testing for model updates
- [ ] Add authentication and rate limiting
- [ ] Deploy to cloud (AWS/GCP/Azure)

## Known Limitations

1. **Missing Data:** Many features have >50% missing values
2. **Static Data:** No real-time updates
3. **Survivorship Bias:** Only active funds included
4. **Categorical Features:** Limited use of categorical variables
5. **No Time-Series:** Price history not utilized

## Lessons Learned

1. **Data Quality Matters:** Missing values significantly impact model training
2. **Feature Selection:** More features ≠ better performance
3. **XGBoost Dominates:** Consistently outperformed other algorithms
4. **API Design:** Clear endpoints make integration easier
5. **Documentation:** Comprehensive docs save time later

## Acknowledgments

- **Dataset:** Stefano Leone (Kaggle)
- **Course:** DataTalks.Club ML ZoomCamp
- **Libraries:** scikit-learn, XGBoost, Flask, pandas teams

## Contact & Support

For questions or issues:
1. Review README.md and DEPLOYMENT.md
2. Check notebook.ipynb for implementation details
3. Run test_service.py to verify setup
4. Open an issue in the repository

---

**Project Completion Date:** November 4, 2025
**ML ZoomCamp 2025 - Midterm Project**
**Status:** ✅ Complete and Ready for Submission
