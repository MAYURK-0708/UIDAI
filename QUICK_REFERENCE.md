# 🚀 QUICK REFERENCE GUIDE - AADHAAR OBSERVATORY v2.0

## ⚡ 5-Minute Quick Start

### 1. Setup Environment
```bash
cd C:\Users\DebSarkar\Desktop\UADAI
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Prepare Data
```
Ensure data files in:
data/raw/
├── enrolment/api_data_aadhar_enrolment/*.csv
├── demographic/api_data_aadhar_demographic/*.csv
└── biometric/api_data_aadhar_biometric/*.csv
```

### 3. Run Analysis
```bash
python run_analysis.py
```

### 4. Launch Services
```bash
# All-in-one command:
python start_all_services.py

# Or individual services:
python api_server.py        # Terminal 1
streamlit run dashboard/app.py  # Terminal 2
```

### 5. Access Dashboards
- 🌐 **Web Dashboard**: http://localhost:5000 → Open `dashboard.html`
- 📊 **Streamlit**: http://localhost:8501
- 📡 **API**: http://localhost:5000/api/health

---

## 📋 File Guide

### Core Analysis
| File | Purpose |
|------|---------|
| `run_analysis.py` | Main analysis pipeline |
| `train_models.py` | ML model training |
| `notebooks/aadhaar_observatory.ipynb` | Jupyter notebook |

### API & Web
| File | Purpose |
|------|---------|
| `api_server.py` | REST API (8+ endpoints) |
| `dashboard.html` | Web dashboard |
| `dashboard/app.py` | Streamlit app |

### Modules
| File | Purpose |
|------|---------|
| `src/data_loader.py` | Data loading & cleaning |
| `src/feature_engineering.py` | Create 8 metrics |
| `src/anomaly_detection.py` | 5 detection methods |
| `src/predictive_models.py` | Forecasting models |
| `src/visualizations.py` | Plotting functions |
| `src/advanced_analytics.py` | ML clustering & PCA |

### Configuration & Deployment
| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker image |
| `docker-compose.yml` | Docker Compose |
| `.env.template` | Config template |
| `.gitignore` | Git ignore rules |

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | Full documentation |
| `DEPLOYMENT_GUIDE.md` | Deployment instructions |
| `UPGRADE_SUMMARY.md` | What's new in v2.0 |
| `QUICK_REFERENCE.md` | This file! |

---

## 🔥 Common Commands

### Data & Analysis
```bash
# Run full analysis pipeline
python run_analysis.py

# Train ML models
python train_models.py

# Run Jupyter notebook
jupyter notebook notebooks/aadhaar_observatory.ipynb
```

### Services
```bash
# Start all services
python start_all_services.py

# Start API only
python api_server.py

# Start Streamlit only
streamlit run dashboard/app.py
```

### Docker
```bash
# Build image
docker build -t aadhaar-observatory .

# Run with Compose
docker-compose up

# Stop containers
docker-compose down

# View logs
docker-compose logs -f
```

### Git
```bash
# Check status
git status

# View commits
git log --oneline

# Push changes
git push origin main

# Pull updates
git pull origin main
```

---

## 📡 API Endpoints Quick Reference

### Health & Info
```bash
GET /api/health              # System status
GET /api/                    # API info
```

### Data Access
```bash
GET /api/summary             # Overall statistics
GET /api/states              # List all states
GET /api/districts?state=X   # Districts in state
```

### Analysis
```bash
POST /api/regional-analysis  # Regional deep-dive
POST /api/clustering         # Clustering analysis
POST /api/risk-assessment    # Risk evaluation
POST /api/predictions        # Demand forecasting
```

### Example Requests
```bash
# Get summary
curl http://localhost:5000/api/summary

# Get states
curl http://localhost:5000/api/states

# Regional analysis
curl -X POST http://localhost:5000/api/regional-analysis \
  -H "Content-Type: application/json" \
  -d '{"type":"state","region":"Uttar Pradesh"}'

# Risk assessment
curl http://localhost:5000/api/risk-assessment
```

---

## 🎯 Feature Highlights

### 8 Advanced Metrics
1. **Growth Rate** - Change in enrollments/updates
2. **Rolling Averages** - 3/6/12-month smoothing
3. **Volatility Index** - Instability measurement
4. **Seasonal Index** - Pattern identification
5. **Update Ratio** - Updates vs enrollments
6. **Biometric Stress Index** - Bio system pressure
7. **Demographic Volatility** - Demographic instability
8. **Aadhaar Fragility Index** - Composite risk score

### 5 Anomaly Detection Methods
1. **Z-Score** - Statistical outliers
2. **STL Decomposition** - Trend/seasonal analysis
3. **Isolation Forest** - ML outlier detection
4. **LOF** - Density-based detection
5. **PELT** - Change-point detection

### ML Models (6 Algorithms)
- Random Forest (classification)
- XGBoost (classification + regression)
- LightGBM (classification)
- Gradient Boosting (regression)
- Prophet (time series)
- Voting Ensemble (combined)

### 2 Dashboards
- **Streamlit** - Python-native dashboard
- **Web Dashboard** - HTML/JS/Plotly

### 1 REST API
- 8+ endpoints
- JSON responses
- CORS-enabled
- Production-ready

---

## ⚙️ Configuration

### Environment Variables (`.env`)
```env
FLASK_ENV=development
FLASK_DEBUG=False
API_PORT=5000
STREAMLIT_PORT=8501
DATA_DIR=./data/processed
LOG_LEVEL=INFO
ANOMALY_THRESHOLD=3
N_CLUSTERS=4
```

### Run Configuration
Edit before running:
```python
# In run_analysis.py
anomaly_threshold = 3  # Z-score threshold
n_clusters = 4  # Number of clustering groups
forecast_periods = 90  # Days to forecast
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find and kill process
netstat -tulpn | grep 5000
kill -9 <PID>

# Or use different port
python api_server.py --port 5001
```

### Missing Data
```bash
# Check data directory
ls data/raw/enrolment/api_data_aadhar_enrolment/

# Ensure CSV files exist
# Then run analysis
python run_analysis.py
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or specific package
pip install flask --upgrade
```

### Streamlit Connection Issues
```bash
# Make sure API is running first
python api_server.py

# In new terminal
streamlit run dashboard/app.py --server.port 8501
```

---

## 📊 Output Files Generated

After running `python run_analysis.py`:

### Processed Data
```
data/processed/
├── enrolment_cleaned.csv
├── demographic_cleaned.csv
├── biometric_cleaned.csv
├── enrolment_with_anomalies.csv
└── merged_with_features.csv (main file)
```

### After `python train_models.py`:

### Trained Models
```
models/
├── vulnerability_model.pkl
└── demand_model.pkl
```

### Generated Reports
```
outputs/
├── figures/
│   ├── enrollment_trends.png
│   ├── biometric_coverage.png
│   └── ... (10+ visualizations)
├── models/
│   ├── forecast_*.csv
│   └── predictions_*.pkl
└── reports/
    ├── hackathon_report.md
    └── pitch_deck.md
```

---

## 📈 Project Statistics

| Metric | Value |
|--------|-------|
| Total Lines of Code | 5,000+ |
| Python Files | 10+ |
| API Endpoints | 8+ |
| ML Models | 6+ |
| Dashboards | 2 |
| Documentation Pages | 3 |
| Dependencies | 35+ |
| Test Coverage | Ready |

---

## 🔐 Security Checklist

- ✅ No credentials in code
- ✅ Environment variables for secrets
- ✅ CORS configuration
- ✅ Error handling
- ✅ Input validation
- ✅ Logging enabled
- ✅ Rate limiting ready
- ✅ Docker isolation

---

## 📱 Access URLs

| Service | URL | Port |
|---------|-----|------|
| Web Dashboard | http://localhost:5000 | 5000 |
| API Server | http://localhost:5000/api | 5000 |
| Streamlit | http://localhost:8501 | 8501 |

---

## 🎓 Learning Path

1. **Start**: Read `README.md`
2. **Understand**: Run `notebooks/aadhaar_observatory.ipynb`
3. **Execute**: Run `python run_analysis.py`
4. **Explore**: Visit dashboards and API
5. **Deploy**: Follow `DEPLOYMENT_GUIDE.md`
6. **Enhance**: Train models with `python train_models.py`

---

## 🚀 Next Steps

1. **Verify data** in `data/raw/`
2. **Run analysis** with `python run_analysis.py`
3. **Start services** with `python start_all_services.py`
4. **Open dashboard** at http://localhost:5000
5. **Query API** at http://localhost:5000/api

---

## 📞 Resources

- **GitHub**: https://github.com/Debjyoti-sarkar/UIDAI
- **Full README**: See `README.md`
- **Deployment**: See `DEPLOYMENT_GUIDE.md`
- **What's New**: See `UPGRADE_SUMMARY.md`
- **API Docs**: Hit `/api/` endpoint

---

## 💡 Tips & Tricks

### Fast Development
```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# Run tests
pytest -v
```

### Performance
```python
# Use caching in API
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

# Batch process large files
for chunk in pd.read_csv('large.csv', chunksize=10000):
    process(chunk)
```

### Debugging
```python
# Add logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print intermediate values
print(f"Debug: {variable}")

# Use debugger
import pdb; pdb.set_trace()
```

---

**Version**: 2.0
**Last Updated**: Jan 19, 2026
**Status**: ✅ Production Ready

🎉 Happy coding!
