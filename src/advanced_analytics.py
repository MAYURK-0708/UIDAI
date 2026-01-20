"""
Advanced Analytics Module
Provides ML-based clustering, segmentation, and predictive insights
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class AdvancedAnalytics:
    """Advanced analytics engine for regional profiling and clustering"""
    
    def __init__(self, data_df):
        self.df = data_df.copy()
        self.scaler = StandardScaler()
        self.pca = None
        self.clusters = None
        
    @staticmethod
    def compute_risk_score(row):
        """Compute comprehensive risk score for each region"""
        components = []
        weights = []
        
        # Enrollment volatility (40% weight)
        if 'enrol_std' in row.index:
            enrol_risk = min(row['enrol_std'] / 1000, 1.0)
            components.append(enrol_risk)
            weights.append(0.40)
        
        # Population vulnerability (30% weight)
        if 'child_population_pct' in row.index:
            vuln_risk = row['child_population_pct'] / 100
            components.append(vuln_risk)
            weights.append(0.30)
        
        # Update lag (20% weight)
        if 'update_lag_days' in row.index:
            lag_risk = min(row['update_lag_days'] / 365, 1.0)
            components.append(lag_risk)
            weights.append(0.20)
        
        # Biometric coverage gap (10% weight)
        if 'biometric_coverage' in row.index:
            bio_risk = 1 - (row['biometric_coverage'] / 100)
            components.append(bio_risk)
            weights.append(0.10)
        
        if components:
            risk_score = np.average(components, weights=weights[:len(components)])
            return min(risk_score * 100, 100)
        return 0
    
    def cluster_regions(self, n_clusters=4, features=None):
        """Cluster regions based on key features"""
        if features is None:
            features = ['enrol_growth', 'update_velocity', 'bio_penetration', 
                       'age_diversity', 'geographic_spread']
        
        # Select available features
        available_features = [f for f in features if f in self.df.columns]
        if not available_features:
            return None
        
        X = self.df[available_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal clusters if n_clusters is 'auto'
        if n_clusters == 'auto':
            silhouette_scores = []
            K_range = range(2, min(10, len(self.df) // 5))
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                silhouette_scores.append((k, score))
            n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(X_scaled)
        self.df['cluster'] = self.clusters
        
        return self.clusters
    
    def perform_pca(self, n_components=2):
        """Perform PCA for dimensionality reduction"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        X = self.df[numeric_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=n_components)
        pca_result = self.pca.fit_transform(X_scaled)
        
        self.df[f'PC1'] = pca_result[:, 0]
        if n_components > 1:
            self.df[f'PC2'] = pca_result[:, 1]
        
        return pca_result, self.pca.explained_variance_ratio_
    
    def predict_vulnerability(self, target_col='fragility_category'):
        """Predict vulnerability based on features"""
        try:
            feature_cols = [c for c in self.df.columns 
                           if c not in [target_col, 'state', 'district', 'date', 'pincode']]
            X = self.df[feature_cols].fillna(0)
            
            if target_col in self.df.columns:
                y = self.df[target_col].astype('category').cat.codes
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return model, importance_df.head(10)
        except Exception as e:
            print(f"Error in vulnerability prediction: {e}")
        
        return None, None
    
    def predict_demand(self, dates_ahead=90):
        """Predict future enrolment demand"""
        try:
            if 'date' not in self.df.columns or 'daily_enrollments' not in self.df.columns:
                return None
            
            time_series = self.df.groupby('date')['daily_enrollments'].sum()
            
            if len(time_series) < 30:
                return None
            
            X = np.arange(len(time_series)).reshape(-1, 1)
            y = time_series.values
            
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Predict forward
            future_X = np.arange(len(time_series), len(time_series) + dates_ahead).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            return predictions, time_series
        except Exception as e:
            print(f"Error in demand prediction: {e}")
        
        return None, None
    
    def get_regional_profiles(self):
        """Generate detailed regional profiles"""
        if 'cluster' not in self.df.columns:
            self.cluster_regions()
        
        profiles = {}
        for cluster_id in self.df['cluster'].unique():
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            profiles[f'Cluster_{cluster_id}'] = {
                'regions': len(cluster_data),
                'avg_enrollment': cluster_data['daily_enrollments'].mean() if 'daily_enrollments' in cluster_data.columns else 0,
                'avg_update_rate': cluster_data['daily_updates'].mean() if 'daily_updates' in cluster_data.columns else 0,
                'risk_score': cluster_data['risk_score'].mean() if 'risk_score' in cluster_data.columns else 0,
                'top_states': cluster_data['state'].value_counts().head(3).to_dict() if 'state' in cluster_data.columns else {}
            }
        
        return profiles
    
    def anomaly_insights(self):
        """Identify key anomalies and insights"""
        insights = []
        
        if 'anomaly_score' in self.df.columns:
            anomalies = self.df[self.df['anomaly_score'] > self.df['anomaly_score'].quantile(0.95)]
            if len(anomalies) > 0:
                insights.append({
                    'type': 'High Anomaly Regions',
                    'count': len(anomalies),
                    'regions': anomalies['state'].value_counts().head(3).to_dict()
                })
        
        if 'risk_score' in self.df.columns:
            high_risk = self.df[self.df['risk_score'] > 70]
            if len(high_risk) > 0:
                insights.append({
                    'type': 'High Risk Regions',
                    'count': len(high_risk),
                    'regions': high_risk['state'].value_counts().head(3).to_dict()
                })
        
        return insights


class SegmentationEngine:
    """Geographic and demographic segmentation"""
    
    @staticmethod
    def segment_by_population(df, age_cols=['age_0_5', 'age_5_17', 'age_18_greater']):
        """Segment by age group concentration"""
        available_cols = [c for c in age_cols if c in df.columns]
        
        if not available_cols:
            return df
        
        df_copy = df.copy()
        total = df_copy[available_cols].sum(axis=1)
        
        for col in available_cols:
            df_copy[f'{col}_pct'] = (df_copy[col] / total * 100).fillna(0)
        
        # Create population segments
        child_pct = df_copy.get('age_0_5_pct', 0) + df_copy.get('age_5_17_pct', 0)
        
        def segment(val):
            if val >= 40:
                return 'Youth-Heavy'
            elif val >= 25:
                return 'Balanced'
            else:
                return 'Aging'
        
        df_copy['population_segment'] = child_pct.apply(segment)
        return df_copy
    
    @staticmethod
    def segment_by_development(df, metrics=['bio_penetration', 'update_velocity']):
        """Segment by development indicators"""
        if 'bio_penetration' not in df.columns:
            return df
        
        df_copy = df.copy()
        
        bio_pct = df_copy['bio_penetration'].fillna(0)
        
        def segment(val):
            if val >= 80:
                return 'Highly Developed'
            elif val >= 60:
                return 'Developing'
            elif val >= 40:
                return 'Emerging'
            else:
                return 'Initial Stage'
        
        df_copy['development_stage'] = bio_pct.apply(segment)
        return df_copy
