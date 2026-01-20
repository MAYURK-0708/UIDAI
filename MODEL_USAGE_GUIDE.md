# 🤖 ML Model Usage Guide

## Quick Start

### Step 1: Train Models
```bash
python train_models.py
```

This creates model files in `models/`:
- `vulnerability_model.pkl` - Predicts risk categories (Low/Moderate/High/Critical)
- `demand_model.pkl` - Forecasts future enrolment demand
- `scaler.pkl` - Feature scaling transformer
- `label_encoder.pkl` - Category label encoder

---

## Usage Methods

### Method 1: Python Script

```python
# Load the predictor
from models.predict import ModelPredictor
import pandas as pd

# Initialize
predictor = ModelPredictor()
predictor.load_models()

# Prepare your features
features = pd.DataFrame({
    'total_enrolments': [5000],
    'total_updates': [300],
    'biometric_coverage': [0.85],
    # ... add all required features
})

# Predict vulnerability
vulnerability, confidence = predictor.predict_vulnerability(features)
print(f"Risk Level: {vulnerability[0]}")
print(f"Confidence: {confidence[0].max()*100:.1f}%")

# Predict demand
demand = predictor.predict_demand(features)
print(f"Predicted Demand: {demand[0]:,.0f}")
```

### Method 2: API Endpoints

Start the API server:
```bash
python api_server.py
```

#### Predict Vulnerability
```bash
curl -X POST http://localhost:5000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "type": "vulnerability",
    "features": {
      "total_enrolments": 5000,
      "total_updates": 300,
      "biometric_coverage": 0.85,
      "update_rate": 0.06,
      "biometric_update_rate": 0.04
    }
  }'
```

Response:
```json
{
  "type": "vulnerability",
  "predictions": [{
    "prediction": "Moderate",
    "confidence": 0.87,
    "probabilities": {
      "Low": 0.05,
      "Moderate": 0.87,
      "High": 0.06,
      "Critical": 0.02
    }
  }],
  "timestamp": "2026-01-20T10:30:00"
}
```

#### Predict Demand
```bash
curl -X POST http://localhost:5000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "type": "demand",
    "features": {
      "total_enrolments": 5000,
      "population_density": 450,
      "growth_rate": 0.12
    }
  }'
```

#### Batch Predictions
```bash
curl -X POST http://localhost:5000/api/batch-predictions \
  -H "Content-Type: application/json" \
  -d '{
    "type": "vulnerability",
    "regions": ["Maharashtra", "Gujarat", "Tamil Nadu"]
  }'
```

### Method 3: Direct Model Loading

```python
import pickle
import pandas as pd

# Load models
with open('models/vulnerability_model.pkl', 'rb') as f:
    vuln_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Prepare features
features = pd.DataFrame({...})  # Your feature data

# Scale and predict
X_scaled = scaler.transform(features)
prediction = vuln_model.predict(X_scaled)
category = label_encoder.inverse_transform(prediction)

print(f"Predicted Category: {category[0]}")
```

---

## Required Features

Both models need these features (must match training):

```python
required_features = [
    'total_enrolments',
    'total_updates', 
    'total_biometric_updates',
    'biometric_coverage',
    'update_rate',
    'biometric_update_rate',
    # Add any other features from feature_engineering
]
```

To get exact feature list:
```python
import pandas as pd
df = pd.read_csv('data/processed/merged_with_features.csv')
features = [c for c in df.columns if c not in 
           ['fragility_category', 'state', 'district', 'date', 'pincode']]
print(features)
```

---

## Testing the Models

```bash
# Test with sample prediction
cd models
python predict.py
```

This will:
1. Load all models
2. Load sample data
3. Make test predictions
4. Show results

---

## Monitoring Model Performance

Track predictions over time:

```python
import pandas as pd
from datetime import datetime

# Log predictions
predictions_log = []

def log_prediction(features, prediction, confidence):
    predictions_log.append({
        'timestamp': datetime.now(),
        'features': features,
        'prediction': prediction,
        'confidence': confidence
    })

# Save logs
pd.DataFrame(predictions_log).to_csv('logs/predictions.csv')

# Analyze performance
logs = pd.read_csv('logs/predictions.csv')
print(f"Average Confidence: {logs['confidence'].mean():.2f}")
print(f"Predictions Distribution:\n{logs['prediction'].value_counts()}")
```

---

## Common Issues & Solutions

### Issue: "Model file not found"
**Solution:** Run `python train_models.py` first

### Issue: "Feature mismatch"
**Solution:** Ensure input features match training features exactly

### Issue: "Scaler error"
**Solution:** Features must be in same order as training

### Issue: "Low confidence scores"
**Solution:** Model may need retraining with more data

---

## API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/predictions` | POST | Single/batch predictions |
| `/api/batch-predictions` | POST | Bulk regional predictions |
| `/api/health` | GET | Check system status |

---

## Next Steps

1. ✅ Train models: `python train_models.py`
2. ✅ Test predictions: `python models/predict.py`
3. ✅ Start API: `python api_server.py`
4. ✅ Integrate into dashboard
5. ✅ Monitor performance
6. 🔄 Retrain periodically with new data

---

## Production Deployment

For production use:

```python
# Use gunicorn for API
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app

# Set up model versioning
models/
  ├── v1/
  │   ├── vulnerability_model.pkl
  │   └── demand_model.pkl
  ├── v2/
  │   ├── vulnerability_model.pkl
  │   └── demand_model.pkl
  └── current -> v2/

# Add model validation
from sklearn.metrics import accuracy_score

def validate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy > 0.80  # 80% threshold
```

---

**Built for UIDAI Data Hackathon 2026** 🇮🇳
