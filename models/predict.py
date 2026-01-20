"""
Model Prediction Helper
Load trained models and make predictions
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path


class ModelPredictor:
    """Load and use trained models for predictions"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.vulnerability_model = None
        self.demand_model = None
        self.scaler = None
        self.label_encoder = None
        
    def load_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        
        try:
            # Load vulnerability model
            with open(self.model_dir / 'vulnerability_model.pkl', 'rb') as f:
                self.vulnerability_model = pickle.load(f)
            print("✓ Vulnerability model loaded")
            
            # Load demand model
            with open(self.model_dir / 'demand_model.pkl', 'rb') as f:
                self.demand_model = pickle.load(f)
            print("✓ Demand model loaded")
            
            # Load scaler
            with open(self.model_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("✓ Scaler loaded")
            
            # Load label encoder
            with open(self.model_dir / 'label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✓ Label encoder loaded")
            
            return True
        except FileNotFoundError as e:
            print(f"✗ Model file not found: {e}")
            print("Please run: python train_models.py")
            return False
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False
    
    def predict_vulnerability(self, features_df):
        """
        Predict vulnerability category for given features
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            predictions: Array of predicted categories
            probabilities: Array of prediction probabilities
        """
        if self.vulnerability_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(features_df)
        
        # Predict
        predictions_encoded = self.vulnerability_model.predict(X_scaled)
        probabilities = self.vulnerability_model.predict_proba(X_scaled)
        
        # Decode predictions
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions, probabilities
    
    def predict_demand(self, features_df):
        """
        Predict future demand for given features
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            predictions: Array of predicted demand values
        """
        if self.demand_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(features_df)
        
        # Predict
        predictions = self.demand_model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self, model_type='vulnerability'):
        """
        Get feature importance from the model
        
        Args:
            model_type: 'vulnerability' or 'demand'
            
        Returns:
            dict: Feature names and their importance scores
        """
        if model_type == 'vulnerability' and self.vulnerability_model:
            # For ensemble, get feature importance from first estimator
            if hasattr(self.vulnerability_model, 'feature_importances_'):
                return self.vulnerability_model.feature_importances_
        elif model_type == 'demand' and self.demand_model:
            if hasattr(self.demand_model, 'feature_importances_'):
                return self.demand_model.feature_importances_
        
        return None


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("MODEL PREDICTION EXAMPLE")
    print("="*80)
    
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Load models
    if predictor.load_models():
        print("\n✓ All models loaded successfully!")
        print("\nYou can now use:")
        print("  - predictor.predict_vulnerability(features_df)")
        print("  - predictor.predict_demand(features_df)")
        
        # Example: Create sample features
        print("\n" + "="*80)
        print("SAMPLE PREDICTION")
        print("="*80)
        
        # Load some real data for demonstration
        try:
            import sys
            sys.path.insert(0, '../src')
            from data_loader import AadhaarDataLoader
            
            df = pd.read_csv('../data/processed/merged_with_features.csv')
            
            # Get feature columns (same as used in training)
            feature_cols = [c for c in df.columns 
                          if c not in ['fragility_category', 'state', 'district', 'date', 'pincode']]
            
            # Sample one row
            sample_features = df[feature_cols].iloc[[0]]
            
            print(f"\nPredicting for sample data...")
            
            # Predict vulnerability
            vuln_pred, vuln_prob = predictor.predict_vulnerability(sample_features)
            print(f"\nVulnerability Prediction: {vuln_pred[0]}")
            print(f"Confidence: {vuln_prob[0].max()*100:.1f}%")
            
            # Predict demand
            demand_pred = predictor.predict_demand(sample_features)
            print(f"\nDemand Prediction: {demand_pred[0]:,.0f}")
            
        except Exception as e:
            print(f"\nSample prediction skipped: {e}")
    else:
        print("\n✗ Failed to load models")
        print("Run: python train_models.py")
