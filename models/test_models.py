"""
Quick Test Script for ML Models
Run this after training models to verify everything works
"""

import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

from predict import ModelPredictor
import pandas as pd

print("="*80)
print("ML MODEL TEST SCRIPT")
print("="*80)

# Step 1: Initialize predictor
print("\n[1/4] Initializing predictor...")
predictor = ModelPredictor(model_dir='.')

# Step 2: Load models
print("\n[2/4] Loading trained models...")
if not predictor.load_models():
    print("\n❌ FAILED: Models not found")
    print("\nTo train models, run:")
    print("  python train_models.py")
    sys.exit(1)

print("\n✅ SUCCESS: All models loaded")

# Step 3: Load sample data
print("\n[3/4] Loading sample data...")
try:
    df = pd.read_csv('../data/processed/merged_with_features.csv')
    print(f"✅ Loaded {len(df):,} records")
    
    # Get feature columns
    feature_cols = [c for c in df.columns 
                   if c not in ['fragility_category', 'state', 'district', 'date', 'pincode']]
    print(f"✅ Found {len(feature_cols)} features")
    
except FileNotFoundError:
    print("❌ Sample data not found")
    print("Run: python run_analysis_fast.py")
    sys.exit(1)

# Step 4: Test predictions
print("\n[4/4] Testing predictions...")
print("-" * 80)

# Sample 5 random records
samples = df.sample(5)

for idx, (i, row) in enumerate(samples.iterrows(), 1):
    print(f"\n🔍 Sample {idx}:")
    print(f"   State: {row['state']}")
    if 'district' in row:
        print(f"   District: {row['district']}")
    
    # Prepare features
    features = row[feature_cols].to_frame().T
    
    try:
        # Vulnerability prediction
        vuln_pred, vuln_prob = predictor.predict_vulnerability(features)
        print(f"   📊 Vulnerability: {vuln_pred[0]}")
        print(f"   🎯 Confidence: {vuln_prob[0].max()*100:.1f}%")
        
        # Demand prediction
        demand_pred = predictor.predict_demand(features)
        print(f"   📈 Demand Forecast: {demand_pred[0]:,.0f}")
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)

print("\n✅ Models are working correctly!")
print("\nNext steps:")
print("  1. Start API server: python ../api_server.py")
print("  2. Test API endpoint:")
print("     curl -X POST http://localhost:5000/api/predictions \\")
print('       -H "Content-Type: application/json" \\')
print('       -d \'{"type": "vulnerability", "features": {...}}\'')
print("\nFor detailed usage, see: MODEL_USAGE_GUIDE.md")
