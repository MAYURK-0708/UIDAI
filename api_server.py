"""
Flask REST API for Aadhaar Observatory
Provides API endpoints for data access, analysis, and predictions
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

from data_loader import AadhaarDataLoader
from advanced_analytics import AdvancedAnalytics, SegmentationEngine
from anomaly_detection import AnomalyDetector

# Try to load ML models
try:
    import pickle
    MODEL_DIR = current_dir / 'models'
    
    def load_ml_models():
        """Load trained ML models"""
        models = {}
        try:
            with open(MODEL_DIR / 'vulnerability_model.pkl', 'rb') as f:
                models['vulnerability'] = pickle.load(f)
            with open(MODEL_DIR / 'demand_model.pkl', 'rb') as f:
                models['demand'] = pickle.load(f)
            with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
                models['scaler'] = pickle.load(f)
            with open(MODEL_DIR / 'label_encoder.pkl', 'rb') as f:
                models['label_encoder'] = pickle.load(f)
            logger.info("✓ ML models loaded successfully")
            return models
        except FileNotFoundError:
            logger.warning("ML models not found. Run: python train_models.py")
            return None
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            return None
    
    ML_MODELS = load_ml_models()
except ImportError:
    ML_MODELS = None
    logger.warning("Pickle not available")

# Initialize Flask app
app = Flask(__name__, 
            template_folder=str(current_dir / 'templates'),
            static_folder=str(current_dir / 'static'))
CORS(app)

# Global data cache
_data_cache = {
    'merged_df': None,
    'analytics': None,
    'last_updated': None
}

def load_processed_data():
    """Load processed data from cache or disk"""
    global _data_cache
    
    try:
        data_dir = current_dir / 'data' / 'processed'
        merged_file = data_dir / 'merged_with_features.csv'
        
        if merged_file.exists():
            df = pd.read_csv(merged_file, parse_dates=['date'])
            _data_cache['merged_df'] = df
            _data_cache['analytics'] = AdvancedAnalytics(df)
            _data_cache['last_updated'] = datetime.now()
            logger.info(f"Loaded {len(df)} records from processed data")
            return df
        else:
            logger.warning(f"Processed data not found at {merged_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        return None

# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return {
        'project': "India's Aadhaar Stress, Vulnerability & Inclusion Observatory",
        'version': '2.0',
        'hackathon': 'UIDAI Data Hackathon 2026',
        'endpoints': {
            'health': '/api/health',
            'summary': '/api/summary',
            'states': '/api/states',
            'districts': '/api/districts',
            'regional_analysis': '/api/regional-analysis',
            'clustering': '/api/clustering',
            'predictions': '/api/predictions',
            'risk_assessment': '/api/risk-assessment'
        }
    }, 200

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        df = _data_cache['merged_df']
        if df is not None:
            return {
                'status': 'healthy',
                'records': len(df),
                'last_updated': _data_cache['last_updated'].isoformat() if _data_cache['last_updated'] else None,
                'timestamp': datetime.now().isoformat()
            }, 200
        else:
            return {'status': 'degraded', 'message': 'Data not loaded'}, 503
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}, 500

# ============================================================================
# Data Access Endpoints
# ============================================================================

@app.route('/api/summary', methods=['GET'])
def summary():
    """Get overall summary statistics"""
    df = _data_cache['merged_df']
    if df is None:
        return {'error': 'Data not loaded'}, 503
    
    try:
        summary_stats = {
            'total_records': len(df),
            'states': df['state'].nunique() if 'state' in df.columns else 0,
            'districts': df['district'].nunique() if 'district' in df.columns else 0,
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None
            },
            'enrollment_total': df['daily_enrollments'].sum() if 'daily_enrollments' in df.columns else 0,
            'update_total': df['daily_updates'].sum() if 'daily_updates' in df.columns else 0,
            'biometric_coverage_avg': df['biometric_coverage'].mean() if 'biometric_coverage' in df.columns else 0
        }
        
        if 'fragility_category' in df.columns:
            fragility_dist = df['fragility_category'].value_counts().to_dict()
            summary_stats['fragility_distribution'] = fragility_dist
        
        return jsonify(summary_stats), 200
    except Exception as e:
        logger.error(f"Error in summary: {e}")
        return {'error': str(e)}, 500

@app.route('/api/states', methods=['GET'])
def get_states():
    """Get list of all states with key metrics"""
    df = _data_cache['merged_df']
    if df is None or 'state' not in df.columns:
        return {'error': 'Data not available'}, 503
    
    try:
        states_data = []
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            states_data.append({
                'state': state,
                'districts': state_df['district'].nunique() if 'district' in state_df.columns else 0,
                'enrollments': state_df['daily_enrollments'].sum() if 'daily_enrollments' in state_df.columns else 0,
                'updates': state_df['daily_updates'].sum() if 'daily_updates' in state_df.columns else 0,
                'avg_biometric_coverage': state_df['biometric_coverage'].mean() if 'biometric_coverage' in state_df.columns else 0
            })
        
        return jsonify(sorted(states_data, key=lambda x: x['enrollments'], reverse=True)), 200
    except Exception as e:
        logger.error(f"Error fetching states: {e}")
        return {'error': str(e)}, 500

@app.route('/api/districts', methods=['GET'])
def get_districts():
    """Get district-level data with filtering"""
    df = _data_cache['merged_df']
    if df is None or 'district' not in df.columns:
        return {'error': 'Data not available'}, 503
    
    try:
        state_filter = request.args.get('state')
        limit = request.args.get('limit', 50, type=int)
        
        districts_df = df.copy()
        if state_filter:
            districts_df = districts_df[districts_df['state'] == state_filter]
        
        grouped = districts_df.groupby('district').agg({
            'daily_enrollments': 'sum' if 'daily_enrollments' in districts_df.columns else 'count',
            'daily_updates': 'sum' if 'daily_updates' in districts_df.columns else 'count',
            'biometric_coverage': 'mean' if 'biometric_coverage' in districts_df.columns else lambda x: 0
        }).sort_values('daily_enrollments', ascending=False).head(limit)
        
        return jsonify(grouped.to_dict('index')), 200
    except Exception as e:
        logger.error(f"Error fetching districts: {e}")
        return {'error': str(e)}, 500

# ============================================================================
# Advanced Analytics Endpoints
# ============================================================================

@app.route('/api/regional-analysis', methods=['POST'])
def regional_analysis():
    """Analyze specific region or state"""
    df = _data_cache['merged_df']
    if df is None:
        return {'error': 'Data not loaded'}, 503
    
    try:
        data = request.get_json()
        region_type = data.get('type', 'state')  # 'state' or 'district'
        region_name = data.get('region')
        
        if region_type == 'state':
            region_df = df[df['state'] == region_name]
        else:
            region_df = df[df['district'] == region_name]
        
        if len(region_df) == 0:
            return {'error': f'{region_type} not found'}, 404
        
        # Create analytics object
        analytics = AdvancedAnalytics(region_df)
        
        analysis = {
            'region': region_name,
            'type': region_type,
            'record_count': len(region_df),
            'enrollment_stats': {
                'total': region_df['daily_enrollments'].sum() if 'daily_enrollments' in region_df.columns else 0,
                'mean': region_df['daily_enrollments'].mean() if 'daily_enrollments' in region_df.columns else 0,
                'std': region_df['daily_enrollments'].std() if 'daily_enrollments' in region_df.columns else 0
            },
            'update_stats': {
                'total': region_df['daily_updates'].sum() if 'daily_updates' in region_df.columns else 0,
                'mean': region_df['daily_updates'].mean() if 'daily_updates' in region_df.columns else 0
            },
            'biometric_coverage': region_df['biometric_coverage'].mean() if 'biometric_coverage' in region_df.columns else 0
        }
        
        # Add segmentation
        segmented = SegmentationEngine.segment_by_development(region_df)
        analysis['development_stages'] = segmented['development_stage'].value_counts().to_dict() if 'development_stage' in segmented.columns else {}
        
        return jsonify(analysis), 200
    except Exception as e:
        logger.error(f"Error in regional analysis: {e}")
        return {'error': str(e)}, 500

@app.route('/api/clustering', methods=['POST'])
def clustering():
    """Perform clustering analysis"""
    analytics = _data_cache['analytics']
    if analytics is None:
        return {'error': 'Analytics engine not available'}, 503
    
    try:
        data = request.get_json() or {}
        n_clusters = data.get('n_clusters', 4)
        
        clusters = analytics.cluster_regions(n_clusters=n_clusters)
        profiles = analytics.get_regional_profiles()
        
        return jsonify({
            'n_clusters': n_clusters,
            'profiles': profiles
        }), 200
    except Exception as e:
        logger.error(f"Error in clustering: {e}")
        return {'error': str(e)}, 500

@app.route('/api/risk-assessment', methods=['GET'])
def risk_assessment():
    """Get risk assessment for all regions"""
    df = _data_cache['merged_df']
    if df is None:
        return {'error': 'Data not loaded'}, 503
    
    try:
        df_copy = df.copy()
        
        # Compute risk scores
        df_copy['risk_score'] = df_copy.apply(AdvancedAnalytics.compute_risk_score, axis=1)
        
        # Categorize risk
        def risk_category(score):
            if score >= 70:
                return 'Critical'
            elif score >= 50:
                return 'High'
            elif score >= 30:
                return 'Moderate'
            else:
                return 'Low'
        
        df_copy['risk_category'] = df_copy['risk_score'].apply(risk_category)
        
        # Aggregate by state
        risk_by_state = df_copy.groupby('state').agg({
            'risk_score': 'mean',
            'risk_category': lambda x: x.value_counts().index[0]
        }).sort_values('risk_score', ascending=False)
        
        return jsonify({
            'by_state': risk_by_state.to_dict('index'),
            'critical_regions': df_copy[df_copy['risk_category'] == 'Critical']['state'].value_counts().to_dict()
        }), 200
    except Exception as e:
        logger.error(f"Error in risk assessment: {e}")
        return {'error': str(e)}, 500

@app.route('/api/predictions', methods=['POST'])
def predictions():
    """Get ML model predictions for vulnerability and demand"""
    if ML_MODELS is None:
        return {'error': 'ML models not loaded. Run: python train_models.py'}, 503
    
    try:
        data = request.get_json() or {}
        prediction_type = data.get('type', 'vulnerability')
        features = data.get('features')  # Dictionary or list of feature values
        
        if features is None:
            return {'error': 'Missing features parameter'}, 400
        
        # Convert features to DataFrame
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        elif isinstance(features, list):
            features_df = pd.DataFrame(features)
        else:
            return {'error': 'Features must be dict or list'}, 400
        
        # Make predictions based on type
        if prediction_type == 'vulnerability':
            # Scale features
            X_scaled = ML_MODELS['scaler'].transform(features_df)
            
            # Predict
            predictions_encoded = ML_MODELS['vulnerability'].predict(X_scaled)
            probabilities = ML_MODELS['vulnerability'].predict_proba(X_scaled)
            
            # Decode predictions
            predictions = ML_MODELS['label_encoder'].inverse_transform(predictions_encoded)
            
            # Format response
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    'prediction': pred,
                    'confidence': float(probabilities[i].max()),
                    'probabilities': {
                        cls: float(prob) 
                        for cls, prob in zip(ML_MODELS['label_encoder'].classes_, probabilities[i])
                    }
                })
            
            return jsonify({
                'type': 'vulnerability',
                'predictions': results,
                'timestamp': datetime.now().isoformat()
            }), 200
            
        elif prediction_type == 'demand':
            # Scale features
            X_scaled = ML_MODELS['scaler'].transform(features_df)
            
            # Predict
            predictions = ML_MODELS['demand'].predict(X_scaled)
            
            return jsonify({
                'type': 'demand',
                'predictions': predictions.tolist(),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        else:
            return {'error': 'Unknown prediction type. Use: vulnerability or demand'}, 400
            
    except Exception as e:
        logger.error(f"Error in predictions: {e}")
        return {'error': str(e)}, 500

@app.route('/api/batch-predictions', methods=['POST'])
def batch_predictions():
    """Get batch predictions for multiple regions"""
    if ML_MODELS is None:
        return {'error': 'ML models not loaded'}, 503
    
    df = _data_cache['merged_df']
    if df is None:
        return {'error': 'Data not loaded'}, 503
    
    try:
        data = request.get_json() or {}
        regions = data.get('regions', [])  # List of state/district names
        prediction_type = data.get('type', 'vulnerability')
        
        if not regions:
            # Use all regions if none specified
            df_sample = df.sample(min(100, len(df)))
        else:
            df_sample = df[df['state'].isin(regions) | df['district'].isin(regions)]
        
        # Get feature columns
        feature_cols = [c for c in df_sample.columns 
                       if c not in ['fragility_category', 'state', 'district', 'date', 'pincode']]
        
        X = df_sample[feature_cols].fillna(0)
        X_scaled = ML_MODELS['scaler'].transform(X)
        
        if prediction_type == 'vulnerability':
            predictions = ML_MODELS['vulnerability'].predict(X_scaled)
            predictions_decoded = ML_MODELS['label_encoder'].inverse_transform(predictions)
            
            results = pd.DataFrame({
                'state': df_sample['state'].values,
                'district': df_sample['district'].values if 'district' in df_sample.columns else None,
                'prediction': predictions_decoded
            })
            
            return jsonify({
                'type': 'vulnerability',
                'count': len(results),
                'predictions': results.to_dict('records')
            }), 200
        
        return {'error': 'Only vulnerability predictions supported in batch mode'}, 400
        
    except Exception as e:
        logger.error(f"Error in batch predictions: {e}")
        return {'error': str(e)}, 500

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return {'error': 'Endpoint not found'}, 404

@app.errorhandler(500)
def server_error(error):
    return {'error': 'Internal server error'}, 500

# ============================================================================
# Initialization
# ============================================================================

if __name__ == '__main__':
    # Load data on startup
    print("Loading Aadhaar Observatory data...")
    load_processed_data()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
