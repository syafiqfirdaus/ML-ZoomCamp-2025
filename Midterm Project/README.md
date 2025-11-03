# Mutual Funds & ETFs Performance Prediction - ML ZoomCamp 2025 Midterm Project

## Problem Description

This project tackles investment fund analysis using machine learning to solve **four distinct prediction problems**:

### 1. Fund Performance Prediction (Regression)
Predict the **1-year return** of mutual funds and ETFs based on their characteristics, historical performance, asset allocation, and risk metrics. This helps investors identify funds likely to perform well.

### 2. Investment Quality Classification (Binary Classification)
Classify funds as **"Good Investment" or "Poor Investment"** based on their Morningstar overall rating (>=4 stars = good, <4 stars = poor). This provides a simple recommendation system for investors.

### 3. Risk Rating Prediction (Multi-class Classification)
Predict the **Morningstar Risk Rating** (Low, Average, High) for funds based on their characteristics. This helps investors understand the risk profile of investment options.

### 4. Fund Recommendation System
Recommend similar funds based on characteristics like asset allocation, sector exposure, fees, and historical performance metrics.

## Dataset

**Source**: [Kaggle - Mutual Funds and ETFs Dataset](https://www.kaggle.com/datasets/stefanoleone992/mutual-funds-and-etfs)

The dataset contains:
- **MutualFunds.csv**: 23,784 mutual funds with 236 features
- **ETFs.csv**: 2,311 ETFs with similar features
- **Price history files**: Historical pricing data for both fund types

### Key Features:
- **Fund Characteristics**: Symbol, name, category, family, inception date
- **Financial Metrics**: Total net assets, expense ratios, investment minimums
- **Performance Data**: Returns (YTD, 1M, 3M, 1Y, 3Y, 5Y, 10Y), quarterly returns
- **Asset Allocation**: Stocks, bonds, cash, preferred securities percentages
- **Sector Exposure**: Technology, healthcare, financials, energy, etc.
- **Risk Metrics**: Alpha, beta, Sharpe ratio, Treynor ratio, standard deviation
- **Ratings**: Morningstar overall rating, risk rating, return rating
- **ESG Scores**: Environmental, social, governance scores
- **Bond Metrics**: Duration, maturity, quality ratings

### Dataset Issues
See the `Dataset issues/` folder for known data quality concerns documented by the community.

## Project Structure

```
.
├── README.md                      # This file
├── Deliverables.md               # Project requirements
├── notebook.ipynb                # Full ML workflow notebook
├── train.py                      # Model training script
├── predict.py                    # Flask web service for predictions
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── models/                       # Trained models (saved as pickle files)
│   ├── regression_model.pkl
│   ├── binary_classifier.pkl
│   ├── multiclass_classifier.pkl
│   └── scaler.pkl
├── MutualFunds.csv              # Main dataset
├── ETFs.csv                     # ETF dataset
└── Dataset issues/              # Known data quality issues

```

## Machine Learning Approach

### Models Trained

For each problem, we compare multiple algorithms:

#### Regression (1-Year Return Prediction)
- Linear Regression (baseline)
- Ridge Regression
- Random Forest Regressor
- **XGBoost Regressor** (primary model)

#### Binary Classification (Good/Poor Investment)
- Logistic Regression (baseline)
- Random Forest Classifier
- **XGBoost Classifier** (primary model)

#### Multi-class Classification (Risk Rating)
- Logistic Regression (baseline)
- Random Forest Classifier
- **XGBoost Classifier** (primary model)

### Feature Engineering
- Handle missing values (imputation strategies)
- Encode categorical variables (one-hot encoding)
- Scale numerical features (StandardScaler)
- Feature selection based on importance
- Create derived features (e.g., return-to-risk ratios)

### Model Evaluation
- **Regression**: RMSE, MAE, R² score
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Cross-validation for robust performance estimates
- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV

## Installation & Setup

### Prerequisites
- Python 3.8+
- Docker (for containerized deployment)
- 4GB+ RAM recommended

### Local Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Midterm Project"
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset** (if not included)
```bash
# Option 1: Manual download from Kaggle
# Visit: https://www.kaggle.com/datasets/stefanoleone992/mutual-funds-and-etfs

# Option 2: Using Kaggle API (requires kaggle.json in ~/.kaggle/)
kaggle datasets download -d stefanoleone992/mutual-funds-and-etfs
unzip mutual-funds-and-etfs.zip
```

## Usage

### 1. Explore the Data & Train Models (Jupyter Notebook)

```bash
jupyter notebook notebook.ipynb
```

The notebook contains:
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Model training and comparison
- Feature importance analysis
- Model selection and evaluation

### 2. Train the Final Model

```bash
python train.py
```

This script:
- Loads and preprocesses the data
- Trains the best performing models
- Saves models to `models/` directory
- Outputs performance metrics

### 3. Run the Prediction Service

```bash
python predict.py
```

The Flask web service will start on `http://localhost:5000`

#### API Endpoints:

**Health Check**
```bash
curl http://localhost:5000/
```

**Predict Fund Performance (Regression)**
```bash
curl -X POST http://localhost:5000/predict/regression \
  -H "Content-Type: application/json" \
  -d '{
    "total_net_assets": 1000000000,
    "fund_prospectus_net_expense_ratio": 0.05,
    "morningstar_overall_rating": 4,
    "fund_return_3years": 0.15,
    "asset_stocks": 0.85,
    "asset_bonds": 0.10,
    "fund_beta_3years": 1.05
  }'
```

**Classify Investment Quality (Binary)**
```bash
curl -X POST http://localhost:5000/predict/classification \
  -H "Content-Type: application/json" \
  -d '{
    "total_net_assets": 1000000000,
    "fund_prospectus_net_expense_ratio": 0.05,
    "fund_return_3years": 0.15,
    "fund_sharpe_ratio_3years": 1.2
  }'
```

### 4. Docker Deployment

**Build the Docker image**
```bash
docker build -t fund-predictor .
```

**Run the container**
```bash
docker run -p 5000:5000 fund-predictor
```

**Test the service**
```bash
curl http://localhost:5000/
```

## Results & Performance

### Model Performance (Preliminary)

| Problem Type | Best Model | Primary Metric | Score |
|-------------|------------|----------------|-------|
| 1-Year Return Prediction | XGBoost | RMSE | TBD |
| Investment Quality | XGBoost | ROC-AUC | TBD |
| Risk Rating | XGBoost | F1-Score (Macro) | TBD |

*Final metrics will be updated after training*

### Key Findings
- Most important features for performance prediction: historical returns, expense ratios, fund size
- ESG scores show moderate correlation with long-term performance
- Sector allocation significantly impacts risk ratings
- Large-cap funds have more predictable returns than small-cap

## Dependencies

See [requirements.txt](requirements.txt) for full list. Key libraries:
- pandas >= 2.2.0
- numpy >= 1.26.0
- scikit-learn >= 1.4.0
- xgboost >= 2.0.0
- flask >= 3.0.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0

## Future Improvements

- [ ] Incorporate time-series analysis using price history data
- [ ] Build ensemble models combining multiple predictors
- [ ] Add explainability features (SHAP values)
- [ ] Deploy to cloud platform (AWS/GCP/Azure)
- [ ] Create interactive dashboard for fund analysis
- [ ] Implement real-time data updates via financial APIs

## Author

**ML ZoomCamp 2025 - Midterm Project**

## License

Dataset: CC0 Public Domain
Code: MIT License

## Acknowledgments

- Dataset: Stefano Leone ([Kaggle](https://www.kaggle.com/stefanoleone992))
- ML ZoomCamp: DataTalks.Club
- XGBoost, scikit-learn, and Flask communities
