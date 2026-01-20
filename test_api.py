"""
API Testing Script
Quick test of all API endpoints
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

print("="*80)
print("AADHAAR OBSERVATORY API TEST")
print("="*80)

# Test 1: Health Check
print("\n[1] Testing Health Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Summary Statistics
print("\n[2] Testing Summary Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/summary")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total Records: {data.get('total_records', 0):,}")
    print(f"States: {data.get('states', 0)}")
    print(f"Districts: {data.get('districts', 0)}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Get States
print("\n[3] Testing States Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/states")
    print(f"Status: {response.status_code}")
    states = response.json()
    if isinstance(states, list):
        print(f"Number of states: {len(states)}")
        if states:
            print(f"Sample: {states[0]}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Regional Analysis
print("\n[4] Testing Regional Analysis...")
try:
    payload = {
        "type": "state",
        "region": "Maharashtra"
    }
    response = requests.post(
        f"{BASE_URL}/api/regional-analysis",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Region: {data.get('region')}")
        print(f"Records: {data.get('record_count', 0):,}")
except Exception as e:
    print(f"Error: {e}")

# Test 5: Risk Assessment
print("\n[5] Testing Risk Assessment...")
try:
    response = requests.get(f"{BASE_URL}/api/risk-assessment")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✓ Risk assessment data available")
except Exception as e:
    print(f"Error: {e}")

# Test 6: ML Predictions (if models are loaded)
print("\n[6] Testing ML Predictions...")
try:
    payload = {
        "type": "vulnerability",
        "features": {
            "total_enrolments": 5000,
            "total_updates": 300,
            "total_biometric_updates": 150,
            "biometric_coverage": 0.85,
            "update_rate": 0.06,
            "biometric_update_rate": 0.03
        }
    }
    response = requests.post(
        f"{BASE_URL}/api/predictions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        pred = data.get('predictions', [{}])[0]
        print(f"Prediction: {pred.get('prediction')}")
        print(f"Confidence: {pred.get('confidence', 0)*100:.1f}%")
    elif response.status_code == 503:
        print("⚠ ML models not loaded (run: python train_models.py)")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
print("API TEST COMPLETED")
print("="*80)

print("\n📌 API is running at: http://localhost:5000")
print("\n🌐 Available Endpoints:")
print("   GET  /                     - API info")
print("   GET  /api/health           - Health check")
print("   GET  /api/summary          - Overall statistics")
print("   GET  /api/states           - List all states")
print("   GET  /api/districts        - List all districts")
print("   POST /api/regional-analysis - Regional analysis")
print("   GET  /api/risk-assessment  - Risk assessment")
print("   POST /api/predictions      - ML predictions")
print("   POST /api/batch-predictions - Batch predictions")
print("\n💡 Tip: Open dashboard.html in browser or use Postman for testing")
