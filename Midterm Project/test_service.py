#!/usr/bin/env python3
"""
Test script for the Flask prediction service
ML ZoomCamp 2025 - Midterm Project
"""

import requests
import json


def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    response = requests.get('http://localhost:5000/')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_regression():
    """Test regression endpoint"""
    print("Testing regression prediction...")

    # Sample input data
    data = {
        'total_net_assets': 1000000000,
        'fund_prospectus_net_expense_ratio': 0.05,
        'fund_return_3years': 0.15,
        'fund_beta_3years': 1.05,
        'asset_stocks': 0.85,
        'asset_bonds': 0.10,
        'fund_sharpe_ratio_3years': 1.2
    }

    response = requests.post(
        'http://localhost:5000/predict/regression',
        json=data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_classification():
    """Test binary classification endpoint"""
    print("Testing binary classification...")

    data = {
        'total_net_assets': 500000000,
        'fund_prospectus_net_expense_ratio': 0.08,
        'fund_return_3years': 0.10,
        'fund_beta_3years': 0.95,
        'fund_sharpe_ratio_3years': 0.8
    }

    response = requests.post(
        'http://localhost:5000/predict/classification',
        json=data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_risk():
    """Test risk rating endpoint"""
    print("Testing risk rating prediction...")

    data = {
        'fund_beta_3years': 1.2,
        'fund_stdev_3years': 0.18,
        'fund_sharpe_ratio_3years': 0.9,
        'asset_stocks': 0.90,
        'asset_bonds': 0.05
    }

    response = requests.post(
        'http://localhost:5000/predict/risk',
        json=data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_all_predictions():
    """Test comprehensive prediction endpoint"""
    print("Testing all predictions endpoint...")

    data = {
        'total_net_assets': 2000000000,
        'fund_prospectus_net_expense_ratio': 0.03,
        'fund_return_3years': 0.20,
        'fund_return_5years': 0.18,
        'fund_beta_3years': 1.1,
        'fund_alpha_3years': 2.5,
        'fund_sharpe_ratio_3years': 1.5,
        'fund_stdev_3years': 0.15,
        'asset_stocks': 0.80,
        'asset_bonds': 0.15,
        'asset_cash': 0.05
    }

    response = requests.post(
        'http://localhost:5000/predict/all',
        json=data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_features():
    """Test features endpoint"""
    print("Testing features endpoint...")
    response = requests.get('http://localhost:5000/features')
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Feature Count: {data['feature_count']}")
    print(f"First 10 features: {data['features'][:10]}\n")
    return response.status_code == 200


def main():
    """Run all tests"""
    print("="*80)
    print("FLASK PREDICTION SERVICE - TEST SUITE")
    print("="*80)
    print("\nMake sure the service is running on http://localhost:5000\n")

    tests = [
        ("Health Check", test_health_check),
        ("Regression Prediction", test_regression),
        ("Binary Classification", test_classification),
        ("Risk Rating Prediction", test_risk),
        ("All Predictions", test_all_predictions),
        ("Features List", test_features)
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"ERROR in {name}: {e}\n")
            results.append((name, False))

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80)


if __name__ == '__main__':
    main()
