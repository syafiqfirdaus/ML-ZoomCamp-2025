# Quick Start Guide

Get the Mutual Funds & ETFs Prediction service up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum
- 3GB free disk space

## Step 1: Install Dependencies (1 minute)

```bash
cd "Midterm Project"
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Train Models (5-10 minutes)

```bash
python train.py
```

Expected output:
```
================================================================================
MUTUAL FUNDS & ETFs PERFORMANCE PREDICTION - TRAINING PIPELINE
================================================================================

Loading data...
  Total funds: 26,095
  Total features: 236

...

TRAINING COMPLETE - PERFORMANCE SUMMARY
================================================================================
✓ All models saved to 'models/' directory!
```

## Step 3: Start the Service (instant)

```bash
python predict.py
```

Expected output:
```
Loading models...
✓ All models loaded successfully!

Starting Flask Prediction Service
================================================================================
 * Running on http://0.0.0.0:5000
```

## Step 4: Test the Service (1 minute)

Open a new terminal and run:

```bash
# Quick health check
curl http://localhost:5000/

# Make a prediction
curl -X POST http://localhost:5000/predict/all \
  -H "Content-Type: application/json" \
  -d '{
    "fund_return_3years": 0.15,
    "fund_beta_3years": 1.05,
    "asset_stocks": 0.80
  }'
```

Or run the test suite:

```bash
python test_service.py
```

## That's It!

You now have a fully functional ML prediction service running locally.

## Next Steps

### Explore the API

Visit http://localhost:5000/ in your browser to see all available endpoints.

### Try Different Predictions

**Predict 1-year return:**
```bash
curl -X POST http://localhost:5000/predict/regression \
  -H "Content-Type: application/json" \
  -d '{"fund_return_3years": 0.20, "fund_sharpe_ratio_3years": 1.5}'
```

**Classify investment quality:**
```bash
curl -X POST http://localhost:5000/predict/classification \
  -H "Content-Type: application/json" \
  -d '{"fund_return_3years": 0.12, "fund_prospectus_net_expense_ratio": 0.05}'
```

**Predict risk rating:**
```bash
curl -X POST http://localhost:5000/predict/risk \
  -H "Content-Type: application/json" \
  -d '{"fund_beta_3years": 1.3, "fund_stdev_3years": 0.20}'
```

### Deploy with Docker

```bash
# Build image
docker build -t fund-predictor .

# Train models (if not already trained)
docker run --rm -v $(pwd)/models:/app/models fund-predictor python train.py

# Run service
docker run -d -p 5000:5000 -v $(pwd)/models:/app/models --name fund-service fund-predictor

# Check logs
docker logs fund-service

# Stop service
docker stop fund-service && docker rm fund-service
```

### Explore the Notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook contains:
- Complete data exploration
- Model training process
- Feature importance analysis
- Visualizations and insights

## Common Issues

**Issue:** `ModuleNotFoundError`
- **Solution:** Make sure you activated the virtual environment and ran `pip install -r requirements.txt`

**Issue:** `FileNotFoundError: models directory not found`
- **Solution:** Run `python train.py` first to create and train the models

**Issue:** `Port 5000 already in use`
- **Solution:** Kill the existing process: `lsof -i :5000` then `kill -9 <PID>`

## Documentation

- [README.md](README.md) - Complete project documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Detailed deployment guide
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Project overview

## Support

Need help? Check the documentation files above or review the code comments in:
- `train.py` - Model training logic
- `predict.py` - API service implementation
- `notebook.ipynb` - Complete ML workflow

---

**Time to get started:** ~7 minutes
**Difficulty:** Beginner-friendly
**Last updated:** November 4, 2025
