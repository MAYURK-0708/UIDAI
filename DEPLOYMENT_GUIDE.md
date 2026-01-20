# AADHAAR OBSERVATORY - SETUP & DEPLOYMENT GUIDE

## 🚀 Quick Start (Windows/Linux/Mac)

### Option 1: Local Python Setup

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Prepare Data
Place your UIDAI datasets in the `data/raw/` directory:
```
data/raw/
├── enrolment/api_data_aadhar_enrolment/*.csv
├── demographic/api_data_aadhar_demographic/*.csv
└── biometric/api_data_aadhar_biometric/*.csv
```

#### Step 3: Run Analysis Pipeline
```bash
python run_analysis.py
```

This generates processed data in `data/processed/`

#### Step 4: Train Advanced Models
```bash
python train_models.py
```

This trains ML models and saves them to `models/`

#### Step 5: Launch All Services
```bash
python start_all_services.py
```

Services will be available at:
- **Web Dashboard**: http://localhost:5000/dashboard.html
- **Streamlit Dashboard**: http://localhost:8501
- **API Server**: http://localhost:5000/api

---

### Option 2: Docker Deployment

#### Step 1: Build Docker Image
```bash
docker build -t aadhaar-observatory .
```

#### Step 2: Run with Docker Compose
```bash
docker-compose up
```

---

## 📡 API Documentation

### Health Check
```bash
GET /api/health

Response:
{
  "status": "healthy",
  "records": 10000,
  "last_updated": "2024-01-19T10:30:00"
}
```

### Summary Statistics
```bash
GET /api/summary

Response:
{
  "total_records": 10000,
  "states": 36,
  "districts": 725,
  "enrollment_total": 5000000,
  "biometric_coverage_avg": 72.5,
  "fragility_distribution": {
    "Low Fragility": 200,
    "Moderate Fragility": 300,
    "High Fragility": 150,
    "Critical Fragility": 75
  }
}
```

### List States
```bash
GET /api/states

Response: [
  {
    "state": "Uttar Pradesh",
    "districts": 75,
    "enrollments": 500000,
    "updates": 150000,
    "avg_biometric_coverage": 78.5
  },
  ...
]
```

### Get Districts
```bash
GET /api/districts?state=Uttar%20Pradesh&limit=20

Response: {
  "Lucknow": {
    "daily_enrollments": 5000,
    "daily_updates": 1500,
    "biometric_coverage": 82
  },
  ...
}
```

### Regional Analysis
```bash
POST /api/regional-analysis

Request:
{
  "type": "state",
  "region": "Uttar Pradesh"
}

Response:
{
  "region": "Uttar Pradesh",
  "record_count": 5000,
  "enrollment_stats": {
    "total": 500000,
    "mean": 100,
    "std": 25
  },
  "development_stages": {
    "Highly Developed": 1000,
    "Developing": 2000,
    "Emerging": 1500,
    "Initial Stage": 500
  }
}
```

### Clustering Analysis
```bash
POST /api/clustering

Request:
{
  "n_clusters": 4
}

Response:
{
  "n_clusters": 4,
  "profiles": {
    "Cluster_0": {
      "regions": 100,
      "avg_enrollment": 5000,
      "avg_update_rate": 1500,
      "risk_score": 45.2,
      "top_states": {"UP": 25, "MP": 20}
    },
    ...
  }
}
```

### Risk Assessment
```bash
GET /api/risk-assessment

Response:
{
  "by_state": {
    "Uttar Pradesh": {
      "risk_score": 45.2,
      "risk_category": "Moderate"
    },
    ...
  },
  "critical_regions": {
    "State1": 5,
    "State2": 3
  }
}
```

### Predictions
```bash
POST /api/predictions

Request:
{
  "type": "demand",
  "days_ahead": 90
}

Response:
{
  "type": "demand",
  "forecast_days": 90,
  "predictions": [5000, 5200, 5100, ...],
  "last_updated": "2024-01-19T10:30:00"
}
```

---

## 🎯 Project File Structure

```
UADAI/
├── api_server.py                 # Flask REST API
├── start_all_services.py         # Service launcher
├── train_models.py               # Model training
├── run_analysis.py               # Main analysis pipeline
├── dashboard.html                # Web dashboard
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose
├── requirements.txt              # Python dependencies
│
├── src/
│   ├── data_loader.py           # Data loading & cleaning
│   ├── feature_engineering.py   # Feature creation
│   ├── anomaly_detection.py     # Anomaly detection
│   ├── predictive_models.py     # ML models
│   ├── visualizations.py        # Plotting functions
│   └── advanced_analytics.py    # Advanced ML models
│
├── dashboard/
│   └── app.py                   # Streamlit app
│
├── notebooks/
│   └── aadhaar_observatory.ipynb # Main notebook
│
├── data/
│   ├── raw/                     # Original data
│   │   ├── enrolment/
│   │   ├── demographic/
│   │   └── biometric/
│   └── processed/               # Processed data
│
├── models/                      # Trained models (generated)
│   ├── vulnerability_model.pkl
│   └── demand_model.pkl
│
└── outputs/
    ├── figures/                 # Generated plots
    ├── models/                  # Forecast files
    └── reports/                 # Documentation
```

---

## ⚙️ Configuration

### Environment Variables
Create a `.env` file:

```env
FLASK_ENV=production
FLASK_DEBUG=False
API_PORT=5000
STREAMLIT_PORT=8501
DATA_DIR=./data/processed
MODEL_DIR=./models
LOG_LEVEL=INFO
```

### Database Integration (Optional)
For production, integrate with a database:

```python
# In api_server.py
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:password@localhost/aadhaar')
df = pd.read_sql('SELECT * FROM processed_data', engine)
```

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
netstat -tulpn | grep 5000

# Kill process
kill -9 <PID>

# Or run on different port
python api_server.py --port 5001
```

### Missing Data Files
```bash
# Ensure data structure:
data/raw/enrolment/api_data_aadhar_enrolment/*.csv
data/raw/demographic/api_data_aadhar_demographic/*.csv
data/raw/biometric/api_data_aadhar_biometric/*.csv

# Then run:
python run_analysis.py
```

### Module Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Streamlit Connection Error
```bash
# Ensure API server is running first
python api_server.py

# In another terminal:
streamlit run dashboard/app.py
```

---

## 📊 Data Workflow

```
Raw Data (CSVs)
      ↓
Data Loading (data_loader.py)
      ↓
Data Cleaning
      ↓
Feature Engineering (feature_engineering.py)
      ↓
Processed Data (saved)
      ↓
Analysis & Modeling
├── Anomaly Detection (anomaly_detection.py)
├── Predictive Models (predictive_models.py)
├── Advanced Analytics (advanced_analytics.py)
└── Visualizations (visualizations.py)
      ↓
Results & Insights
├── API Server
├── Streamlit Dashboard
└── Web Dashboard
```

---

## 🔐 Security Best Practices

1. **Sensitive Data**: Never commit API keys or credentials
2. **Use .env files**: Load credentials from environment
3. **HTTPS**: Deploy with SSL/TLS certificates
4. **Rate Limiting**: Implement API rate limits
5. **Authentication**: Add API key validation

Example:
```python
from functools import wraps
from flask import request, jsonify

API_KEY = os.getenv('API_KEY')

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if not key or key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/secure-endpoint', methods=['GET'])
@require_api_key
def secure_endpoint():
    return jsonify({'data': 'sensitive'})
```

---

## 📈 Performance Optimization

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(state):
    # Do work
    return result
```

### Batch Processing
```python
# Instead of processing row by row
for batch in pd.read_csv(file, chunksize=10000):
    process_batch(batch)
```

### Database Indexing
```sql
CREATE INDEX idx_state_date ON records(state, date);
CREATE INDEX idx_fragility ON records(fragility_category);
```

---

## 🚢 Production Deployment

### Using Gunicorn + Nginx

```bash
# Install gunicorn
pip install gunicorn

# Run Flask app with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name observatory.example.com;

    location /api {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
    }

    location /dashboard {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
    }
}
```

### Systemd Service
```ini
[Unit]
Description=Aadhaar Observatory API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/aadhaar-observatory
ExecStart=/usr/bin/gunicorn -w 4 api_server:app
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 📞 Support & Documentation

- **GitHub Issues**: Report bugs and request features
- **Documentation**: See README.md for detailed explanation
- **API Docs**: Available at `/api/docs` (when using Swagger)
- **Jupyter Notebook**: Full analysis walkthrough

---

**Built with ❤️ for better governance and inclusive digital identity**
