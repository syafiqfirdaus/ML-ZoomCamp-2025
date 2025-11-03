#!/usr/bin/env python3
"""
Prediction Service for Mutual Funds & ETFs Performance Prediction
ML ZoomCamp 2025 - Midterm Project

Flask web service providing ML predictions for fund analysis:
1. Regression: 1-year return prediction
2. Binary Classification: Investment quality classification
3. Multi-class Classification: Risk rating prediction
"""

import pickle
import numpy as np
from flask import Flask, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables for models
models = {}


def load_models():
    """Load all trained models from disk"""
    model_dir = 'models'

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Models directory '{model_dir}' not found. Please run train.py first.")

    print("Loading models...")

    try:
        with open(os.path.join(model_dir, 'regression_model.pkl'), 'rb') as f:
            models['regression_model'] = pickle.load(f)

        with open(os.path.join(model_dir, 'regression_scaler.pkl'), 'rb') as f:
            models['regression_scaler'] = pickle.load(f)

        with open(os.path.join(model_dir, 'binary_classifier.pkl'), 'rb') as f:
            models['binary_classifier'] = pickle.load(f)

        with open(os.path.join(model_dir, 'binary_scaler.pkl'), 'rb') as f:
            models['binary_scaler'] = pickle.load(f)

        with open(os.path.join(model_dir, 'multiclass_classifier.pkl'), 'rb') as f:
            models['multiclass_classifier'] = pickle.load(f)

        with open(os.path.join(model_dir, 'multiclass_scaler.pkl'), 'rb') as f:
            models['multiclass_scaler'] = pickle.load(f)

        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            models['label_encoder'] = pickle.load(f)

        with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
            models['feature_names'] = pickle.load(f)

        with open(os.path.join(model_dir, 'imputer.pkl'), 'rb') as f:
            models['imputer'] = pickle.load(f)

        print("✓ All models loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def validate_input(data, required_features=None):
    """Validate input data"""
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"

    if required_features:
        missing = [f for f in required_features if f not in data]
        if missing:
            return False, f"Missing required features: {missing}"

    return True, ""


def prepare_features(data):
    """Prepare input features for prediction"""
    # Create feature vector matching training data
    feature_names = models['feature_names']

    # Initialize with zeros
    features = np.zeros(len(feature_names))

    # Fill in provided features
    for idx, feat in enumerate(feature_names):
        if feat in data:
            features[idx] = data[feat]

    return features.reshape(1, -1)


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'Mutual Funds & ETFs Performance Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'GET - Health check',
            '/predict/regression': 'POST - Predict 1-year return',
            '/predict/classification': 'POST - Classify investment quality',
            '/predict/risk': 'POST - Predict risk rating',
            '/predict/all': 'POST - Get all predictions'
        },
        'example': {
            'url': '/predict/regression',
            'method': 'POST',
            'body': {
                'total_net_assets': 1000000000,
                'fund_prospectus_net_expense_ratio': 0.05,
                'fund_return_3years': 0.15,
                'fund_beta_3years': 1.05,
                'asset_stocks': 0.85,
                'asset_bonds': 0.10
            }
        }
    })


@app.route('/predict/regression', methods=['POST'])
def predict_regression():
    """Predict 1-year fund return"""
    try:
        # Get input data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Prepare features
        X = prepare_features(data)

        # Scale features
        X_scaled = models['regression_scaler'].transform(X)

        # Predict
        prediction = models['regression_model'].predict(X_scaled)[0]

        return jsonify({
            'prediction_type': 'regression',
            'predicted_1year_return': float(prediction),
            'predicted_1year_return_pct': f"{prediction * 100:.2f}%",
            'interpretation': 'positive' if prediction > 0 else 'negative',
            'model': 'XGBoost Regressor'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/classification', methods=['POST'])
def predict_classification():
    """Classify investment quality (good vs poor)"""
    try:
        # Get input data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Prepare features
        X = prepare_features(data)

        # Scale features
        X_scaled = models['binary_scaler'].transform(X)

        # Predict
        prediction = models['binary_classifier'].predict(X_scaled)[0]
        probability = models['binary_classifier'].predict_proba(X_scaled)[0]

        return jsonify({
            'prediction_type': 'binary_classification',
            'investment_quality': 'good' if prediction == 1 else 'poor',
            'confidence': float(probability[prediction]),
            'probability_good': float(probability[1]),
            'probability_poor': float(probability[0]),
            'recommendation': 'Consider investing' if prediction == 1 else 'Avoid or research more',
            'model': 'XGBoost Classifier'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/risk', methods=['POST'])
def predict_risk():
    """Predict Morningstar risk rating"""
    try:
        # Get input data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Prepare features
        X = prepare_features(data)

        # Scale features
        X_scaled = models['multiclass_scaler'].transform(X)

        # Predict
        prediction = models['multiclass_classifier'].predict(X_scaled)[0]
        probabilities = models['multiclass_classifier'].predict_proba(X_scaled)[0]

        # Decode label
        risk_rating = models['label_encoder'].inverse_transform([prediction])[0]

        # Create probability dictionary
        prob_dict = {
            label: float(prob)
            for label, prob in zip(models['label_encoder'].classes_, probabilities)
        }

        return jsonify({
            'prediction_type': 'multiclass_classification',
            'risk_rating': risk_rating,
            'confidence': float(probabilities[prediction]),
            'probabilities': prob_dict,
            'model': 'XGBoost Classifier'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/all', methods=['POST'])
def predict_all():
    """Get all predictions for a fund"""
    try:
        # Get input data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Prepare features
        X = prepare_features(data)

        # Regression prediction
        X_reg_scaled = models['regression_scaler'].transform(X)
        return_prediction = models['regression_model'].predict(X_reg_scaled)[0]

        # Binary classification
        X_bin_scaled = models['binary_scaler'].transform(X)
        quality_prediction = models['binary_classifier'].predict(X_bin_scaled)[0]
        quality_proba = models['binary_classifier'].predict_proba(X_bin_scaled)[0]

        # Multi-class classification
        X_multi_scaled = models['multiclass_scaler'].transform(X)
        risk_prediction = models['multiclass_classifier'].predict(X_multi_scaled)[0]
        risk_proba = models['multiclass_classifier'].predict_proba(X_multi_scaled)[0]
        risk_rating = models['label_encoder'].inverse_transform([risk_prediction])[0]

        return jsonify({
            'prediction_type': 'comprehensive',
            'regression': {
                'predicted_1year_return': float(return_prediction),
                'predicted_1year_return_pct': f"{return_prediction * 100:.2f}%"
            },
            'classification': {
                'investment_quality': 'good' if quality_prediction == 1 else 'poor',
                'confidence': float(quality_proba[quality_prediction])
            },
            'risk_rating': {
                'rating': risk_rating,
                'confidence': float(risk_proba[risk_prediction])
            },
            'overall_recommendation': get_overall_recommendation(
                return_prediction, quality_prediction, risk_rating
            )
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_overall_recommendation(return_pred, quality_pred, risk_rating):
    """Generate overall investment recommendation"""
    score = 0

    # Return contribution (0-3 points)
    if return_pred > 0.15:
        score += 3
    elif return_pred > 0.08:
        score += 2
    elif return_pred > 0:
        score += 1

    # Quality contribution (0-2 points)
    if quality_pred == 1:
        score += 2

    # Risk consideration
    risk_factor = {
        'Low': 'conservative',
        'Average': 'moderate',
        'High': 'aggressive'
    }.get(risk_rating, 'unknown')

    # Generate recommendation
    if score >= 4:
        recommendation = "Strong Buy"
        explanation = "High expected return with good quality rating"
    elif score >= 3:
        recommendation = "Buy"
        explanation = "Positive outlook with acceptable quality"
    elif score >= 2:
        recommendation = "Hold"
        explanation = "Mixed signals, suitable for existing positions"
    else:
        recommendation = "Avoid"
        explanation = "Low expected return or poor quality rating"

    return {
        'recommendation': recommendation,
        'explanation': explanation,
        'risk_profile': risk_factor,
        'score': score
    }


@app.route('/features', methods=['GET'])
def get_features():
    """Return list of expected features"""
    return jsonify({
        'feature_count': len(models['feature_names']),
        'features': models['feature_names'],
        'note': 'All features are optional. Missing features will be imputed.'
    })


if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("\n" + "="*80)
        print("Starting Flask Prediction Service")
        print("="*80)
        print("\nEndpoints:")
        print("  GET  /                     - Health check")
        print("  POST /predict/regression   - Predict 1-year return")
        print("  POST /predict/classification - Classify investment quality")
        print("  POST /predict/risk         - Predict risk rating")
        print("  POST /predict/all          - Get all predictions")
        print("  GET  /features             - List expected features")
        print("\n" + "="*80)

        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load models. Please run train.py first.")
