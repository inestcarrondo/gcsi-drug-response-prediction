#!/usr/bin/env python3
"""
Test script for the gCSI Drug Response Prediction API
"""
import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting /model/info endpoint...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def test_prediction():
    """Test prediction endpoint with sample data"""
    print("\nTesting /predict endpoint...")
    
    # Sample data for prediction
    test_data = {
        "samples": [
            {
                "norm_drugname": "VINBLASTINE",
                "norm_celllinename": "22RV1_PROSTATE",
                "primarytissue": "prostate",
                "trtduration": 72.0,
                "doublingtime": 48.0,
                "maxlog10concentration": -5.0,
                "gr_aoc": 0.1,
                "gr50": -6.5,
                "r_square_gr": 0.95,
                "h_gr": 2.0,
                "n_conc": 8.0,
                "log10_conc_step": 0.5,
                "smiles": "COc1ccc2[nH]c3c(c2c1)C1(C(=O)N(C)c2cc(OC)c(OC)cc21)C(CC3)N1CCc2cc(OC)c(OC)cc2C1"
            },
            {
                "norm_drugname": "PACLITAXEL",
                "norm_celllinename": "A549_LUNG",
                "primarytissue": "lung",
                "trtduration": 72.0,
                "doublingtime": 24.0,
                "maxlog10concentration": -6.0
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predictions: {result['predictions']}")
        print(f"Processing time: {result['processing_time_seconds']:.3f}s")
        print(f"Model info: {result['model_info']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=" * 50)
    print("gCSI Drug Response API Test Suite")
    print("=" * 50)
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        print("Server not responding. Make sure the API is running on localhost:8000")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Prediction", test_prediction)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results.append((test_name, "ERROR"))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    for test_name, result in results:
        print(f"{test_name:20} {result}")

if __name__ == "__main__":
    main()