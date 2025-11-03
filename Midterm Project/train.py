#!/usr/bin/env python3
"""
Training Script for Mutual Funds & ETFs Performance Prediction
ML ZoomCamp 2025 - Midterm Project

This script trains and saves the final models for:
1. Regression: 1-year return prediction
2. Binary Classification: Investment quality classification
3. Multi-class Classification: Risk rating prediction
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

# Models
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load and prepare the mutual funds and ETFs data"""
    print("Loading data...")

    # Load datasets
    df_mutual = pd.read_csv('MutualFunds.csv')
    df_etf = pd.read_csv('ETFs.csv')

    # Combine datasets
    df = pd.concat([df_mutual, df_etf], axis=0, ignore_index=True)

    print(f"  Total funds: {len(df):,}")
    print(f"  Total features: {df.shape[1]}")

    return df


def select_features(df):
    """Select relevant features for modeling"""
    print("\nSelecting features...")

    features_to_keep = [
        # Target variables
        'fund_return_1year', 'morningstar_overall_rating', 'morningstar_risk_rating',

        # Fund characteristics
        'total_net_assets', 'fund_category', 'inception_date',

        # Expense ratios
        'fund_prospectus_net_expense_ratio', 'fund_annual_report_net_expense_ratio',

        # Asset allocation
        'asset_stocks', 'asset_bonds', 'asset_cash', 'asset_others',

        # Sector exposure
        'fund_sector_technology', 'fund_sector_healthcare', 'fund_sector_financial_services',
        'fund_sector_consumer_cyclical', 'fund_sector_industrials', 'fund_sector_energy',

        # Performance metrics
        'year_to_date_return', 'fund_return_ytd', 'fund_return_3months',
        'fund_return_3years', 'fund_return_5years',

        # Risk metrics
        'fund_beta_3years', 'fund_alpha_3years', 'fund_sharpe_ratio_3years',
        'fund_stdev_3years', 'fund_r_squared_3years',

        # Valuation metrics
        'fund_price_earning_ratio', 'fund_price_book_ratio',

        # Ratings
        'morningstar_return_rating',

        # ESG
        'esg_score', 'environment_score', 'social_score', 'governance_score'
    ]

    # Keep only available columns
    features_to_keep = [col for col in features_to_keep if col in df.columns]
    df_clean = df[features_to_keep].copy()

    print(f"  Selected {len(features_to_keep)} features")

    return df_clean


def prepare_features(df):
    """Prepare features for modeling"""
    print("\nPreparing features...")

    # Create binary target for investment quality
    df['investment_quality'] = (df['morningstar_overall_rating'] >= 4).astype(int)

    # Separate targets and features
    target_cols = ['fund_return_1year', 'morningstar_overall_rating',
                   'morningstar_risk_rating', 'investment_quality']

    # Use only numerical features for simplicity
    feature_cols = [col for col in df.columns if col not in target_cols]
    numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numerical_cols].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    print(f"  Features shape: {X_imputed.shape}")
    print(f"  Missing values after imputation: {X_imputed.isnull().sum().sum()}")

    return df, X_imputed, imputer


def train_regression_model(df, X_imputed):
    """Train regression model for 1-year return prediction"""
    print("\n" + "="*80)
    print("TRAINING REGRESSION MODEL - 1-Year Return Prediction")
    print("="*80)

    # Prepare data
    y = df.loc[X_imputed.index, 'fund_return_1year']
    mask = ~y.isnull()
    X = X_imputed[mask]
    y = y[mask]

    print(f"Dataset size: {X.shape}")
    print(f"Target statistics: Mean={y.mean():.4f}, Std={y.std():.4f}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model
    print("\nTraining XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nTest Set Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

    return model, scaler, {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_binary_classifier(df, X_imputed):
    """Train binary classifier for investment quality"""
    print("\n" + "="*80)
    print("TRAINING BINARY CLASSIFIER - Investment Quality")
    print("="*80)

    # Prepare data
    y = df.loc[X_imputed.index, 'investment_quality']
    mask = ~y.isnull()
    X = X_imputed[mask]
    y = y[mask]

    print(f"Dataset size: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model
    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    return model, scaler, {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1, 'roc_auc': roc_auc
    }


def train_multiclass_classifier(df, X_imputed):
    """Train multi-class classifier for risk rating"""
    print("\n" + "="*80)
    print("TRAINING MULTI-CLASS CLASSIFIER - Risk Rating")
    print("="*80)

    # Prepare data
    y = df.loc[X_imputed.index, 'morningstar_risk_rating']
    mask = ~y.isnull()
    X = X_imputed[mask]
    y = y[mask]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Dataset size: {X.shape}")
    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {pd.Series(y_encoded).value_counts().sort_index().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model
    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (Macro): {precision:.4f}")
    print(f"  Recall (Macro): {recall:.4f}")
    print(f"  F1-Score (Macro): {f1:.4f}")

    return model, scaler, le, {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1
    }


def save_models(models_dict, output_dir='models'):
    """Save all models and supporting objects"""
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save each model
    for name, obj in models_dict.items():
        filepath = os.path.join(output_dir, f"{name}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  ✓ Saved: {filepath}")

    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'models': list(models_dict.keys()),
        'python_version': '3.8+',
        'framework': 'XGBoost'
    }

    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n✓ All models saved to '{output_dir}/' directory!")


def main():
    """Main training pipeline"""
    print("="*80)
    print("MUTUAL FUNDS & ETFs PERFORMANCE PREDICTION - TRAINING PIPELINE")
    print("ML ZoomCamp 2025 - Midterm Project")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and prepare data
    df = load_and_prepare_data()
    df = select_features(df)
    df, X_imputed, imputer = prepare_features(df)

    # Train models
    reg_model, reg_scaler, reg_metrics = train_regression_model(df, X_imputed)
    bin_model, bin_scaler, bin_metrics = train_binary_classifier(df, X_imputed)
    multi_model, multi_scaler, le, multi_metrics = train_multiclass_classifier(df, X_imputed)

    # Save models
    models_to_save = {
        'regression_model': reg_model,
        'regression_scaler': reg_scaler,
        'binary_classifier': bin_model,
        'binary_scaler': bin_scaler,
        'multiclass_classifier': multi_model,
        'multiclass_scaler': multi_scaler,
        'label_encoder': le,
        'imputer': imputer,
        'feature_names': X_imputed.columns.tolist()
    }

    save_models(models_to_save)

    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - PERFORMANCE SUMMARY")
    print("="*80)

    print("\n1. Regression (1-Year Return Prediction)")
    print(f"   RMSE: {reg_metrics['rmse']:.4f}")
    print(f"   MAE:  {reg_metrics['mae']:.4f}")
    print(f"   R²:   {reg_metrics['r2']:.4f}")

    print("\n2. Binary Classification (Investment Quality)")
    print(f"   Accuracy: {bin_metrics['accuracy']:.4f}")
    print(f"   F1-Score: {bin_metrics['f1']:.4f}")
    print(f"   ROC-AUC:  {bin_metrics['roc_auc']:.4f}")

    print("\n3. Multi-class Classification (Risk Rating)")
    print(f"   Accuracy: {multi_metrics['accuracy']:.4f}")
    print(f"   F1-Score (Macro): {multi_metrics['f1']:.4f}")

    print("\n" + "="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
