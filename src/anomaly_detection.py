"""
Anomaly Detection Module
Implements multiple anomaly detection techniques for Aadhaar data
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL
import ruptures as rpt
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Detect anomalies in Aadhaar time series data"""
    
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.anomalies = {}
    
    def rolling_zscore_detection(self, df, value_col, window=12, group_cols=['state', 'district']):
        """
        Detect anomalies using rolling Z-score
        
        Method: Calculate rolling mean and std, then identify values beyond threshold std deviations
        
        Formula: Z = (X - μ_rolling) / σ_rolling
        Anomaly if |Z| > threshold (default: 3)
        
        Args:
            df: DataFrame with time series data
            value_col: Column to detect anomalies in
            window: Rolling window size
            group_cols: Columns to group by
        
        Returns:
            DataFrame with anomaly flags and z-scores
        """
        df = df.sort_values(['date'] + group_cols).copy()
        
        # Calculate rolling statistics
        df['rolling_mean'] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df['rolling_std'] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        # Calculate Z-score
        df['zscore'] = (df[value_col] - df['rolling_mean']) / df['rolling_std'].replace(0, np.nan)
        
        # Flag anomalies
        df['anomaly_zscore'] = (np.abs(df['zscore']) > self.threshold).astype(int)
        df['anomaly_severity_zscore'] = np.abs(df['zscore'])
        
        # Store results
        self.anomalies['zscore'] = df[df['anomaly_zscore'] == 1].copy()
        
        print(f"Z-score Detection: Found {df['anomaly_zscore'].sum()} anomalies")
        
        return df
    
    def stl_decomposition_detection(self, df, value_col, seasonal_period=12, group_cols=['state', 'district']):
        """
        Detect anomalies using STL (Seasonal-Trend decomposition using Loess)
        
        Method: Decompose time series into trend, seasonal, and residual components
        Anomalies are identified in the residual component
        
        Args:
            df: DataFrame with time series data
            value_col: Column to decompose
            seasonal_period: Period of seasonality (12 for monthly data)
            group_cols: Columns to group by
        
        Returns:
            DataFrame with decomposition and anomaly flags
        """
        df = df.sort_values(['date'] + group_cols).copy()
        
        df['trend'] = np.nan
        df['seasonal'] = np.nan
        df['residual'] = np.nan
        df['anomaly_stl'] = 0
        
        for name, group in df.groupby(group_cols):
            if len(group) >= 2 * seasonal_period:  # Need enough data for STL
                try:
                    # Perform STL decomposition
                    stl = STL(group[value_col].fillna(0), seasonal=seasonal_period, robust=True)
                    result = stl.fit()
                    
                    # Store components
                    df.loc[group.index, 'trend'] = result.trend
                    df.loc[group.index, 'seasonal'] = result.seasonal
                    df.loc[group.index, 'residual'] = result.resid
                    
                    # Detect anomalies in residuals
                    residual_mean = result.resid.mean()
                    residual_std = result.resid.std()
                    
                    if residual_std > 0:
                        residual_zscore = np.abs((result.resid - residual_mean) / residual_std)
                        df.loc[group.index, 'anomaly_stl'] = (residual_zscore > self.threshold).astype(int)
                except Exception as e:
                    # Silently skip regions with insufficient data
                    continue
        
        # Store results
        self.anomalies['stl'] = df[df['anomaly_stl'] == 1].copy()
        
        print(f"STL Detection: Found {df['anomaly_stl'].sum()} anomalies")
        
        return df
    
    def isolation_forest_detection(self, df, feature_cols, contamination=0.05):
        """
        Detect anomalies using Isolation Forest
        
        Method: Unsupervised learning algorithm that isolates anomalies
        Works well for high-dimensional data
        
        Args:
            df: DataFrame with features
            feature_cols: List of columns to use as features
            contamination: Expected proportion of anomalies (0.05 = 5%)
        
        Returns:
            DataFrame with anomaly flags
        """
        print(f"Running Isolation Forest on {len(df)} records...")
        df = df.copy()
        
        # Prepare features
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict (-1 for anomalies, 1 for normal)
        predictions = iso_forest.fit_predict(X)
        df['anomaly_iforest'] = (predictions == -1).astype(int)
        
        # Get anomaly scores (lower = more anomalous)
        df['anomaly_score_iforest'] = -iso_forest.score_samples(X)
        
        # Store results
        self.anomalies['iforest'] = df[df['anomaly_iforest'] == 1].copy()
        
        print(f"Isolation Forest Detection: Found {df['anomaly_iforest'].sum()} anomalies")
        
        return df
    
    def lof_detection(self, df, feature_cols, n_neighbors=20, contamination=0.05):
        """
        Detect anomalies using Local Outlier Factor (LOF)
        
        Method: Density-based anomaly detection
        Identifies points with substantially lower density than neighbors
        
        Args:
            df: DataFrame with features
            feature_cols: List of columns to use as features
            n_neighbors: Number of neighbors to consider
            contamination: Expected proportion of anomalies
        
        Returns:
            DataFrame with anomaly flags
        """
        print(f"Running LOF on {len(df)} records...")
        df = df.copy()
        
        # Prepare features
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Fit LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        
        # Predict (-1 for anomalies, 1 for normal)
        predictions = lof.fit_predict(X)
        df['anomaly_lof'] = (predictions == -1).astype(int)
        
        # Get negative outlier factor (lower = more anomalous)
        df['anomaly_score_lof'] = -lof.negative_outlier_factor_
        
        # Store results
        self.anomalies['lof'] = df[df['anomaly_lof'] == 1].copy()
        
        print(f"LOF Detection: Found {df['anomaly_lof'].sum()} anomalies")
        
        return df
    
    def changepoint_detection(self, df, value_col, group_cols=['state', 'district'], min_size=6):
        """
        Detect change points using PELT algorithm
        
        Method: Identifies structural breaks in time series
        Useful for detecting policy changes, system updates, etc.
        
        Args:
            df: DataFrame with time series data
            value_col: Column to detect change points in
            group_cols: Columns to group by
            min_size: Minimum segment size
        
        Returns:
            DataFrame with change point flags
        """
        df = df.sort_values(['date'] + group_cols).copy()
        df['changepoint'] = 0
        
        groups = list(df.groupby(group_cols))
        total = len(groups)
        print(f"Processing {total} groups for changepoint detection...")
        
        for i, (name, group) in enumerate(groups):
            if (i + 1) % 50 == 0:  # Progress every 50 groups
                print(f"  Progress: {i+1}/{total} groups processed")
            if len(group) >= 2 * min_size:
                try:
                    # Prepare signal
                    signal = group[value_col].fillna(0).values
                    
                    # Detect change points using PELT
                    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
                    change_points = algo.predict(pen=10)
                    
                    # Mark change points (excluding the last point which is always included)
                    for cp in change_points[:-1]:
                        if cp < len(group):
                            df.loc[group.index[cp], 'changepoint'] = 1
                except Exception as e:
                    print(f"Change point detection failed for {name}: {str(e)}")
                    continue
        
        # Store results
        self.anomalies['changepoint'] = df[df['changepoint'] == 1].copy()
        
        print(f"Change Point Detection: Found {df['changepoint'].sum()} change points")
        
        return df
    
    def detect_all_anomalies(self, df, value_col, feature_cols=None):
        """
        Run all anomaly detection methods
        
        Args:
            df: DataFrame with time series data
            value_col: Primary column for time series methods
            feature_cols: Columns for multivariate methods
        
        Returns:
            DataFrame with all anomaly flags
        """
        print("Running all anomaly detection methods...")
        
        # Time series methods
        df = self.rolling_zscore_detection(df, value_col)
        df = self.stl_decomposition_detection(df, value_col)
        df = self.changepoint_detection(df, value_col)
        
        # Multivariate methods
        if feature_cols is None:
            # Use numeric columns
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove anomaly columns
            feature_cols = [col for col in feature_cols if not col.startswith('anomaly')]
            feature_cols = [col for col in feature_cols if col in df.columns][:10]  # Limit to 10 features
        
        if len(feature_cols) > 0:
            df = self.isolation_forest_detection(df, feature_cols)
            df = self.lof_detection(df, feature_cols)
        
        # Create composite anomaly flag
        anomaly_cols = [col for col in df.columns if col.startswith('anomaly_') and not col.endswith('_score')]
        df['anomaly_composite'] = df[anomaly_cols].sum(axis=1)
        df['is_anomaly'] = (df['anomaly_composite'] >= 2).astype(int)  # Flag if 2+ methods agree
        
        print(f"\nComposite Anomaly Detection: Found {df['is_anomaly'].sum()} anomalies (2+ methods agree)")
        
        return df
    
    def get_anomaly_summary(self):
        """Get summary of detected anomalies"""
        summary = {}
        for method, anomalies in self.anomalies.items():
            summary[method] = {
                'count': len(anomalies),
                'states': anomalies['state'].nunique() if 'state' in anomalies.columns else 0,
                'districts': anomalies['district'].nunique() if 'district' in anomalies.columns else 0
            }
        return pd.DataFrame(summary).T


if __name__ == "__main__":
    print("Anomaly Detection Module Ready")
    print("\nAvailable Methods:")
    print("1. Rolling Z-score Detection")
    print("2. STL Decomposition")
    print("3. Isolation Forest")
    print("4. Local Outlier Factor (LOF)")
    print("5. Change Point Detection (PELT)")
