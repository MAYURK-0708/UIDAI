"""
Advanced Model Training & Evaluation
Trains ensemble models for vulnerability prediction and demand forecasting
"""

import pandas as pd
import numpy as np
import sys
import pickle
from pathlib import Path
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import AadhaarDataLoader
from feature_engineering import AadhaarFeatureEngineer


class ModelTrainer:
    """Train and manage ML models"""
    
    def __init__(self, data_dir='data/processed', model_dir='models'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.vulnerability_model = None
        self.demand_model = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load processed data"""
        print("Loading processed data...")
        try:
            df = pd.read_csv(self.data_dir / 'merged_with_features.csv')
            print(f"✓ Loaded {len(df)} records")
            return df
        except FileNotFoundError:
            print("✗ Processed data not found. Run: python run_analysis.py")
            return None
    
    def train_vulnerability_model(self, df):
        """Train model for vulnerability prediction"""
        print("\n" + "="*80)
        print("TRAINING VULNERABILITY PREDICTION MODEL")
        print("="*80)
        
        try:
            # Prepare features and target
            feature_cols = [c for c in df.columns 
                          if c not in ['fragility_category', 'state', 'district', 'date', 'pincode']]
            
            X = df[feature_cols].fillna(0)
            
            if 'fragility_category' not in df.columns:
                print("✗ Target column 'fragility_category' not found")
                return None
            
            y = df['fragility_category']
            
            # Encode target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            print(f"Features: {len(feature_cols)}")
            print(f"Samples: {len(X)}")
            print(f"Classes: {len(le.classes_)}")
            print(f"Class distribution:\n{pd.Series(y).value_counts()}\n")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble
            print("Training ensemble models...")
            
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            # Voting classifier
            voting_clf = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model),
                    ('lgb', lgb_model)
                ],
                voting='soft'
            )
            
            voting_clf.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = voting_clf.predict(X_test_scaled)
            
            print("\n📊 Model Performance:")
            print(f"Accuracy: {voting_clf.score(X_test_scaled, y_test):.4f}")
            print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
            
            # Cross-validation
            cv_scores = cross_val_score(voting_clf, X_train_scaled, y_train, cv=5)
            print(f"Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Feature importance (from RF)
            importances = rf_model.fit(X_train_scaled, y_train).feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Important Features:\n{importance_df.head(10).to_string()}\n")
            
            # Save model
            self.vulnerability_model = voting_clf
            self.model_encoder = le
            
            model_path = self.model_dir / 'vulnerability_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': voting_clf,
                    'encoder': le,
                    'scaler': self.scaler,
                    'feature_cols': feature_cols
                }, f)
            
            print(f"✓ Model saved to {model_path}")
            
            return voting_clf
        
        except Exception as e:
            print(f"✗ Error training vulnerability model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_demand_model(self, df):
        """Train model for demand forecasting"""
        print("\n" + "="*80)
        print("TRAINING DEMAND FORECASTING MODEL")
        print("="*80)
        
        try:
            # Aggregate time series data
            time_series = df.groupby('date').agg({
                'daily_enrollments': 'sum' if 'daily_enrollments' in df.columns else 'count'
            }).reset_index()
            
            if len(time_series) < 30:
                print("✗ Insufficient time series data for training")
                return None
            
            print(f"Time periods: {len(time_series)}")
            
            # Create lag features
            for lag in [1, 7, 30]:
                time_series[f'lag_{lag}'] = time_series['daily_enrollments'].shift(lag)
            
            time_series = time_series.dropna()
            
            X = time_series[[c for c in time_series.columns if c.startswith('lag')]]
            y = time_series['daily_enrollments']
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble
            print("Training gradient boosting models...")
            
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            # Train both
            gb_model.fit(X_train_scaled, y_train)
            xgb_model.fit(X_train_scaled, y_train)
            
            # Ensemble predictions
            y_pred = (gb_model.predict(X_test_scaled) + xgb_model.predict(X_test_scaled)) / 2
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n📊 Model Performance:")
            print(f"RMSE: {rmse:,.0f}")
            print(f"R² Score: {r2:.4f}")
            print(f"MAE: {np.mean(np.abs(y_test - y_pred)):,.0f}")
            
            # Save model
            self.demand_model = (gb_model, xgb_model)
            
            model_path = self.model_dir / 'demand_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'gb_model': gb_model,
                    'xgb_model': xgb_model,
                    'scaler': scaler,
                    'feature_cols': X.columns.tolist()
                }, f)
            
            print(f"✓ Model saved to {model_path}\n")
            
            return (gb_model, xgb_model)
        
        except Exception as e:
            print(f"✗ Error training demand model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_all(self):
        """Train all models"""
        print("\n" + "="*80)
        print("🤖 AADHAAR OBSERVATORY - MODEL TRAINING")
        print("="*80 + "\n")
        
        df = self.load_data()
        if df is None:
            return False
        
        # Train models
        vuln_model = self.train_vulnerability_model(df)
        demand_model = self.train_demand_model(df)
        
        # Summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        print("\nModels saved to:", self.model_dir)
        print("\nTo use models in predictions:")
        print("  1. Load pickle files from models/")
        print("  2. Use in API endpoints for real-time predictions")
        print("  3. Monitor model performance over time\n")
        
        return vuln_model is not None and demand_model is not None


if __name__ == '__main__':
    trainer = ModelTrainer()
    success = trainer.train_all()
    sys.exit(0 if success else 1)
