# Deployment Guide

## Quick Start

### 1. Train the Models

First, train all models using the provided script:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train regression, binary classification, and multi-class classification models
- Save all models to the `models/` directory
- Display performance metrics

Expected output:
```
TRAINING COMPLETE - PERFORMANCE SUMMARY
1. Regression (1-Year Return Prediction)
   RMSE: [value]
   MAE:  [value]
   R²:   [value]

2. Binary Classification (Investment Quality)
   Accuracy: [value]
   F1-Score: [value]

3. Multi-class Classification (Risk Rating)
   Accuracy: [value]
   F1-Score (Macro): [value]
```

### 2. Run the Prediction Service

Start the Flask web service:

```bash
python predict.py
```

The service will be available at `http://localhost:5000`

### 3. Test the Service

In a new terminal, run the test script:

```bash
python test_service.py
```

Or test manually with curl:

```bash
# Health check
curl http://localhost:5000/

# Predict 1-year return
curl -X POST http://localhost:5000/predict/regression \
  -H "Content-Type: application/json" \
  -d '{
    "total_net_assets": 1000000000,
    "fund_prospectus_net_expense_ratio": 0.05,
    "fund_return_3years": 0.15,
    "fund_beta_3years": 1.05,
    "asset_stocks": 0.85
  }'
```

## Docker Deployment

### Build the Docker Image

```bash
docker build -t fund-predictor .
```

This will:
- Create a Python 3.11 environment
- Install all dependencies
- Copy application files and data
- Expose port 5000

### Run the Container

**Option 1: Train models inside container**

```bash
# First, train models
docker run --rm -v $(pwd)/models:/app/models fund-predictor python train.py

# Then run the service
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  --name fund-service \
  fund-predictor
```

**Option 2: Use pre-trained models**

If you've already trained models locally:

```bash
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  --name fund-service \
  fund-predictor
```

### Check Container Status

```bash
# View logs
docker logs fund-service

# Check if running
docker ps

# Test the service
curl http://localhost:5000/
```

### Stop the Container

```bash
docker stop fund-service
docker rm fund-service
```

## API Documentation

### Endpoints

#### 1. Health Check
```
GET /
```

Returns service status and available endpoints.

**Example:**
```bash
curl http://localhost:5000/
```

**Response:**
```json
{
  "status": "online",
  "service": "Mutual Funds & ETFs Performance Prediction API",
  "version": "1.0.0",
  "endpoints": { ... }
}
```

---

#### 2. Predict 1-Year Return (Regression)
```
POST /predict/regression
```

Predicts the expected 1-year return for a fund.

**Input:** JSON object with fund features

**Example:**
```bash
curl -X POST http://localhost:5000/predict/regression \
  -H "Content-Type: application/json" \
  -d '{
    "total_net_assets": 1000000000,
    "fund_prospectus_net_expense_ratio": 0.05,
    "fund_return_3years": 0.15,
    "fund_beta_3years": 1.05,
    "asset_stocks": 0.85,
    "asset_bonds": 0.10,
    "fund_sharpe_ratio_3years": 1.2
  }'
```

**Response:**
```json
{
  "prediction_type": "regression",
  "predicted_1year_return": 0.1234,
  "predicted_1year_return_pct": "12.34%",
  "interpretation": "positive",
  "model": "XGBoost Regressor"
}
```

---

#### 3. Classify Investment Quality (Binary Classification)
```
POST /predict/classification
```

Classifies a fund as "good" or "poor" investment.

**Example:**
```bash
curl -X POST http://localhost:5000/predict/classification \
  -H "Content-Type: application/json" \
  -d '{
    "total_net_assets": 500000000,
    "fund_prospectus_net_expense_ratio": 0.08,
    "fund_return_3years": 0.10,
    "fund_sharpe_ratio_3years": 0.8
  }'
```

**Response:**
```json
{
  "prediction_type": "binary_classification",
  "investment_quality": "good",
  "confidence": 0.87,
  "probability_good": 0.87,
  "probability_poor": 0.13,
  "recommendation": "Consider investing",
  "model": "XGBoost Classifier"
}
```

---

#### 4. Predict Risk Rating (Multi-class Classification)
```
POST /predict/risk
```

Predicts the Morningstar risk rating (Low, Average, High).

**Example:**
```bash
curl -X POST http://localhost:5000/predict/risk \
  -H "Content-Type: application/json" \
  -d '{
    "fund_beta_3years": 1.2,
    "fund_stdev_3years": 0.18,
    "fund_sharpe_ratio_3years": 0.9,
    "asset_stocks": 0.90
  }'
```

**Response:**
```json
{
  "prediction_type": "multiclass_classification",
  "risk_rating": "High",
  "confidence": 0.75,
  "probabilities": {
    "Low": 0.05,
    "Average": 0.20,
    "High": 0.75
  },
  "model": "XGBoost Classifier"
}
```

---

#### 5. Get All Predictions
```
POST /predict/all
```

Returns comprehensive analysis with all three predictions.

**Example:**
```bash
curl -X POST http://localhost:5000/predict/all \
  -H "Content-Type: application/json" \
  -d '{
    "total_net_assets": 2000000000,
    "fund_prospectus_net_expense_ratio": 0.03,
    "fund_return_3years": 0.20,
    "fund_beta_3years": 1.1,
    "fund_sharpe_ratio_3years": 1.5,
    "asset_stocks": 0.80
  }'
```

**Response:**
```json
{
  "prediction_type": "comprehensive",
  "regression": {
    "predicted_1year_return": 0.18,
    "predicted_1year_return_pct": "18.00%"
  },
  "classification": {
    "investment_quality": "good",
    "confidence": 0.92
  },
  "risk_rating": {
    "rating": "Average",
    "confidence": 0.68
  },
  "overall_recommendation": {
    "recommendation": "Strong Buy",
    "explanation": "High expected return with good quality rating",
    "risk_profile": "moderate",
    "score": 5
  }
}
```

---

#### 6. List Expected Features
```
GET /features
```

Returns the list of features expected by the models.

**Example:**
```bash
curl http://localhost:5000/features
```

**Response:**
```json
{
  "feature_count": 35,
  "features": [
    "total_net_assets",
    "fund_prospectus_net_expense_ratio",
    "asset_stocks",
    ...
  ],
  "note": "All features are optional. Missing features will be imputed."
}
```

## Production Deployment

### Cloud Deployment Options

#### AWS (Elastic Beanstalk)

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize EB application:
```bash
eb init -p docker fund-predictor
```

3. Create environment and deploy:
```bash
eb create fund-predictor-env
eb deploy
```

4. Open in browser:
```bash
eb open
```

#### Google Cloud Platform (Cloud Run)

1. Build and push to GCR:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/fund-predictor
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy fund-predictor \
  --image gcr.io/YOUR_PROJECT/fund-predictor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure (Container Instances)

1. Create container registry:
```bash
az acr create --resource-group myResourceGroup \
  --name fundpredictor --sku Basic
```

2. Build and push:
```bash
az acr build --registry fundpredictor \
  --image fund-predictor:v1 .
```

3. Deploy:
```bash
az container create --resource-group myResourceGroup \
  --name fund-predictor \
  --image fundpredictor.azurecr.io/fund-predictor:v1 \
  --ports 5000
```

### Production Considerations

1. **Security:**
   - Add authentication (API keys, OAuth)
   - Use HTTPS (SSL/TLS)
   - Implement rate limiting
   - Validate and sanitize input data

2. **Monitoring:**
   - Set up application logging
   - Monitor API response times
   - Track model performance metrics
   - Alert on errors and anomalies

3. **Scalability:**
   - Use gunicorn for multiple workers
   - Implement load balancing
   - Consider serverless options for variable traffic
   - Cache frequent predictions

4. **Model Management:**
   - Version control for models
   - A/B testing for model updates
   - Rollback capability
   - Regular retraining schedule

### Gunicorn Production Server

For production, use Gunicorn instead of Flask's development server:

```bash
# Install gunicorn (already in requirements.txt)
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5000 predict:app
```

Update Dockerfile CMD:
```dockerfile
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "predict:app"]
```

## Troubleshooting

### Issue: Models not found

**Error:** `FileNotFoundError: Models directory 'models' not found`

**Solution:** Train models first:
```bash
python train.py
```

### Issue: Port already in use

**Error:** `Address already in use`

**Solution:** Kill existing process or use different port:
```bash
# Find process
lsof -i :5000

# Kill process
kill -9 <PID>

# Or use different port in predict.py
app.run(host='0.0.0.0', port=5001)
```

### Issue: Docker build fails

**Error:** Memory or disk space issues

**Solution:**
```bash
# Clean up Docker
docker system prune -a

# Increase Docker memory limit (Docker Desktop settings)
```

### Issue: Predictions are inconsistent

**Possible causes:**
- Missing feature values (imputed differently)
- Model needs retraining with fresh data
- Input data outside training distribution

**Solution:**
- Check input data quality
- Provide more complete feature data
- Retrain models with updated dataset

## Support

For issues or questions:
1. Check this documentation
2. Review the [README.md](README.md)
3. Examine the notebook for model details
4. Open an issue in the repository

---

**Last Updated:** 2025-11-04
**Version:** 1.0.0
