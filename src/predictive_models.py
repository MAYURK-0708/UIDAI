"""
Predictive Modeling Module
Implements forecasting, classification, and clustering models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, silhouette_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    """Forecast Aadhaar demand using multiple methods"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
    
    def prepare_time_series(self, df, value_col, group_cols=['state']):
        """Prepare time series data for forecasting"""
        # Aggregate by date and group
        ts_data = df.groupby(['date'] + group_cols)[value_col].sum().reset_index()
        ts_data = ts_data.sort_values('date')
        return ts_data
    
    def arima_forecast(self, df, value_col, periods=6, order=(1, 1, 1), group_col='state'):
        """
        ARIMA Forecasting
        
        Args:
            df: DataFrame with time series data
            value_col: Column to forecast
            periods: Number of periods to forecast
            order: ARIMA order (p, d, q)
            group_col: Column to group by
        
        Returns:
            Dictionary of forecasts by group
        """
        print(f"Running ARIMA forecasting for {periods} periods...")
        forecasts = {}
        
        for name, group in df.groupby(group_col):
            try:
                # Prepare data
                ts = group.set_index('date')[value_col]
                ts = ts.asfreq('MS')  # Monthly start frequency
                ts = ts.fillna(method='ffill')
                
                # Fit ARIMA
                model = ARIMA(ts, order=order)
                fitted_model = model.fit()
                
                # Forecast
                forecast = fitted_model.forecast(steps=periods)
                
                # Create forecast dataframe
                last_date = ts.index[-1]
                future_dates = pd.date_range(start=last_date, periods=periods+1, freq='MS')[1:]
                
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    group_col: name,
                    f'{value_col}_forecast': forecast.values,
                    'method': 'ARIMA'
                })
                
                forecasts[name] = forecast_df
                
                # Calculate metrics on training data
                fitted_values = fitted_model.fittedvalues
                actual = ts[fitted_values.index]
                rmse = np.sqrt(mean_squared_error(actual, fitted_values))
                mae = mean_absolute_error(actual, fitted_values)
                
                print(f"  {name}: RMSE={rmse:.2f}, MAE={mae:.2f}")
                
            except Exception as e:
                print(f"  ARIMA failed for {name}: {str(e)}")
                continue
        
        self.forecasts['arima'] = pd.concat(forecasts.values(), ignore_index=True)
        return self.forecasts['arima']
    
    def prophet_forecast(self, df, value_col, periods=6, group_col='state'):
        """
        Prophet Forecasting (Facebook's forecasting tool)
        
        Handles seasonality and holidays automatically
        
        Args:
            df: DataFrame with time series data
            value_col: Column to forecast
            periods: Number of periods to forecast
            group_col: Column to group by
        
        Returns:
            Dictionary of forecasts by group
        """
        print(f"Running Prophet forecasting for {periods} periods...")
        forecasts = {}
        
        for name, group in df.groupby(group_col):
            try:
                # Prepare data for Prophet (requires 'ds' and 'y' columns)
                prophet_df = group[['date', value_col]].copy()
                prophet_df.columns = ['ds', 'y']
                prophet_df = prophet_df.dropna()
                
                if len(prophet_df) < 10:  # Need minimum data
                    continue
                
                # Fit Prophet
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                model.fit(prophet_df)
                
                # Create future dataframe
                future = model.make_future_dataframe(periods=periods, freq='MS')
                forecast = model.predict(future)
                
                # Extract forecast
                forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
                forecast_df[group_col] = name
                forecast_df.columns = ['date', f'{value_col}_forecast', f'{value_col}_lower', f'{value_col}_upper', group_col]
                forecast_df['method'] = 'Prophet'
                
                forecasts[name] = forecast_df
                
                # Calculate metrics
                actual = prophet_df['y'].values
                predicted = forecast['yhat'][:len(actual)].values
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mae = mean_absolute_error(actual, predicted)
                
                print(f"  {name}: RMSE={rmse:.2f}, MAE={mae:.2f}")
                
            except Exception as e:
                print(f"  Prophet failed for {name}: {str(e)}")
                continue
        
        if forecasts:
            self.forecasts['prophet'] = pd.concat(forecasts.values(), ignore_index=True)
            return self.forecasts['prophet']
        else:
            return pd.DataFrame()
    
    def simple_moving_average_forecast(self, df, value_col, periods=6, window=3, group_col='state'):
        """
        Simple Moving Average Forecast (baseline)
        
        Args:
            df: DataFrame with time series data
            value_col: Column to forecast
            periods: Number of periods to forecast
            window: Moving average window
            group_col: Column to group by
        
        Returns:
            Forecast dataframe
        """
        print(f"Running Moving Average forecasting for {periods} periods...")
        forecasts = []
        
        for name, group in df.groupby(group_col):
            # Calculate moving average
            ts = group.set_index('date')[value_col]
            ma = ts.rolling(window=window).mean().iloc[-1]
            
            # Create forecast (constant)
            last_date = ts.index[-1]
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq='MS')[1:]
            
            forecast_df = pd.DataFrame({
                'date': future_dates,
                group_col: name,
                f'{value_col}_forecast': ma,
                'method': 'Moving Average'
            })
            
            forecasts.append(forecast_df)
        
        self.forecasts['ma'] = pd.concat(forecasts, ignore_index=True)
        return self.forecasts['ma']


class RiskClassifier:
    """Classify regions by risk level"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def create_risk_labels(self, df, risk_col='aadhaar_fragility_index'):
        """
        Create risk labels based on fragility index
        
        Categories:
        - Low: 0-25
        - Medium: 25-50
        - High: 50-75
        - Critical: 75-100
        """
        df = df.copy()
        
        if risk_col in df.columns:
            df['risk_label'] = pd.cut(
                df[risk_col],
                bins=[0, 25, 50, 75, 100],
                labels=['Low', 'Medium', 'High', 'Critical'],
                include_lowest=True
            )
        else:
            # Use composite score
            df['risk_label'] = 'Medium'  # Default
        
        return df
    
    def train_risk_classifier(self, df, feature_cols, target_col='risk_label'):
        """
        Train Random Forest classifier for risk prediction
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature columns
            target_col: Target column
        
        Returns:
            Trained model and metrics
        """
        print("Training Risk Classifier...")
        
        # Prepare data
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df[target_col].fillna('Medium')
        
        # Remove rows with missing labels
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)
        
        # Predictions
        y_pred = self.model.predict(X_scaled)
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        return self.model, feature_importance
    
    def predict_risk(self, df, feature_cols):
        """Predict risk for new data"""
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities


class RegionClusterer:
    """Cluster regions by behavior patterns"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
    
    def kmeans_clustering(self, df, feature_cols, n_clusters=4):
        """
        KMeans Clustering
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns
            n_clusters: Number of clusters
        
        Returns:
            DataFrame with cluster labels
        """
        print(f"Running KMeans clustering with {n_clusters} clusters...")
        
        # Prepare data
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_scaled, df['cluster_kmeans'])
        print(f"  Silhouette Score: {silhouette:.3f}")
        
        # Store model
        self.models['kmeans'] = kmeans
        
        # Cluster summary
        cluster_summary = df.groupby('cluster_kmeans').agg({
            'state': 'count',
            **{col: 'mean' for col in feature_cols[:5]}  # Top 5 features
        })
        print("\nCluster Summary:")
        print(cluster_summary)
        
        return df
    
    def hierarchical_clustering(self, df, feature_cols, n_clusters=4):
        """
        Hierarchical Clustering
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns
            n_clusters: Number of clusters
        
        Returns:
            DataFrame with cluster labels
        """
        print(f"Running Hierarchical clustering with {n_clusters} clusters...")
        
        # Prepare data
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        df['cluster_hierarchical'] = hierarchical.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_scaled, df['cluster_hierarchical'])
        print(f"  Silhouette Score: {silhouette:.3f}")
        
        # Store model
        self.models['hierarchical'] = hierarchical
        
        return df
    
    def dbscan_clustering(self, df, feature_cols, eps=0.5, min_samples=5):
        """
        DBSCAN Clustering (density-based)
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature columns
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
        
        Returns:
            DataFrame with cluster labels
        """
        print(f"Running DBSCAN clustering...")
        
        # Prepare data
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['cluster_dbscan'] = dbscan.fit_predict(X_scaled)
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(df['cluster_dbscan'])) - (1 if -1 in df['cluster_dbscan'].values else 0)
        n_noise = list(df['cluster_dbscan']).count(-1)
        
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        
        # Store model
        self.models['dbscan'] = dbscan
        
        return df


if __name__ == "__main__":
    print("Predictive Modeling Module Ready")
    print("\nAvailable Models:")
    print("1. ARIMA Forecasting")
    print("2. Prophet Forecasting")
    print("3. Moving Average Forecasting")
    print("4. Random Forest Risk Classification")
    print("5. KMeans Clustering")
    print("6. Hierarchical Clustering")
    print("7. DBSCAN Clustering")
