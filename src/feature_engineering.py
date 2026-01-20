"""
Feature Engineering Module
Creates advanced metrics and indices for Aadhaar analysis
"""

import pandas as pd
import numpy as np
from scipy import stats


class AadhaarFeatureEngineer:
    """Create advanced features and indices for Aadhaar data"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_growth_rate(df, value_col, group_cols=['state', 'district'], periods=1):
        """
        Calculate growth rate (MoM, YoY)
        
        Formula: Growth Rate = ((Value_t - Value_t-n) / Value_t-n) * 100
        
        Args:
            df: DataFrame with time series data
            value_col: Column to calculate growth for
            group_cols: Columns to group by
            periods: Number of periods to look back (1=MoM, 12=YoY)
        
        Returns:
            DataFrame with growth rate column
        """
        df = df.sort_values(['date'] + group_cols)
        
        df[f'{value_col}_growth_rate_{periods}m'] = df.groupby(group_cols)[value_col].pct_change(periods) * 100
        
        return df
    
    @staticmethod
    def calculate_rolling_average(df, value_col, windows=[3, 6, 12], group_cols=['state', 'district']):
        """
        Calculate rolling averages
        
        Formula: Rolling Avg = Mean(Value_t-n to Value_t)
        
        Args:
            df: DataFrame with time series data
            value_col: Column to calculate rolling average for
            windows: List of window sizes (in months)
            group_cols: Columns to group by
        
        Returns:
            DataFrame with rolling average columns
        """
        df = df.sort_values(['date'] + group_cols)
        
        for window in windows:
            df[f'{value_col}_ma_{window}m'] = df.groupby(group_cols)[value_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        return df
    
    @staticmethod
    def calculate_volatility_index(df, value_col, window=6, group_cols=['state', 'district']):
        """
        Calculate Volatility Index
        
        Formula: Volatility = StdDev(Growth_Rate) over rolling window
        
        Higher volatility indicates unstable demand patterns
        
        Args:
            df: DataFrame with time series data
            value_col: Column to calculate volatility for
            window: Rolling window size
            group_cols: Columns to group by
        
        Returns:
            DataFrame with volatility index column
        """
        df = df.sort_values(['date'] + group_cols)
        
        # First calculate growth rate if not exists
        if f'{value_col}_growth_rate_1m' not in df.columns:
            df = AadhaarFeatureEngineer.calculate_growth_rate(df, value_col, group_cols, periods=1)
        
        # Calculate rolling standard deviation of growth rate
        df[f'{value_col}_volatility_{window}m'] = df.groupby(group_cols)[f'{value_col}_growth_rate_1m'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        return df
    
    @staticmethod
    def calculate_seasonal_index(df, value_col, group_cols=['state', 'district']):
        """
        Calculate Seasonal Index
        
        Formula: Seasonal Index = (Actual Value / 12-month Moving Average) * 100
        
        Values > 100 indicate above-average activity
        Values < 100 indicate below-average activity
        
        Args:
            df: DataFrame with time series data
            value_col: Column to calculate seasonal index for
            group_cols: Columns to group by
        
        Returns:
            DataFrame with seasonal index column
        """
        df = df.sort_values(['date'] + group_cols)
        
        # Calculate 12-month moving average
        if f'{value_col}_ma_12m' not in df.columns:
            df = AadhaarFeatureEngineer.calculate_rolling_average(df, value_col, windows=[12], group_cols=group_cols)
        
        # Calculate seasonal index
        df[f'{value_col}_seasonal_index'] = (df[value_col] / df[f'{value_col}_ma_12m']) * 100
        df[f'{value_col}_seasonal_index'] = df[f'{value_col}_seasonal_index'].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    @staticmethod
    def calculate_update_to_enrolment_ratio(enrolment_df, demographic_df, biometric_df):
        """
        Calculate Update-to-Enrolment Ratio
        
        Formula: Update Ratio = (Total Updates / Total Enrolments) * 100
        
        High ratios indicate regions with high update activity relative to enrolments
        
        Args:
            enrolment_df: Enrolment data
            demographic_df: Demographic update data
            biometric_df: Biometric update data
        
        Returns:
            Merged DataFrame with update ratio
        """
        # Aggregate by date, state, district
        group_cols = ['date', 'state', 'district']
        
        # Sum enrolments
        enrol_agg = enrolment_df.groupby(group_cols).agg({
            'total_enrolments': 'sum'
        }).reset_index()
        
        # Sum demographic updates
        demo_cols = [col for col in demographic_df.columns if col.startswith('demo_age')]
        if demo_cols:
            demographic_df['total_demo_updates'] = demographic_df[demo_cols].sum(axis=1)
        demo_agg = demographic_df.groupby(group_cols).agg({
            'total_demo_updates': 'sum'
        }).reset_index()
        
        # Sum biometric updates
        bio_cols = [col for col in biometric_df.columns if col.startswith('bio_age')]
        if bio_cols:
            biometric_df['total_bio_updates'] = biometric_df[bio_cols].sum(axis=1)
        bio_agg = biometric_df.groupby(group_cols).agg({
            'total_bio_updates': 'sum'
        }).reset_index()
        
        # Merge all
        merged = enrol_agg.merge(demo_agg, on=group_cols, how='outer')
        merged = merged.merge(bio_agg, on=group_cols, how='outer')
        merged = merged.fillna(0)
        
        # Calculate ratio
        merged['total_updates'] = merged['total_demo_updates'] + merged['total_bio_updates']
        merged['update_to_enrolment_ratio'] = (merged['total_updates'] / merged['total_enrolments'].replace(0, np.nan)) * 100
        merged['update_to_enrolment_ratio'] = merged['update_to_enrolment_ratio'].replace([np.inf, -np.inf], np.nan)
        
        return merged
    
    @staticmethod
    def calculate_biometric_stress_index(biometric_df, window=6):
        """
        Calculate Biometric Stress Index (BSI)
        
        Formula: BSI = w1 * (Bio_Update_Rate) + w2 * (Bio_Volatility) + w3 * (Age_Concentration)
        
        Where:
        - Bio_Update_Rate: Normalized biometric update frequency
        - Bio_Volatility: Volatility of biometric updates
        - Age_Concentration: Concentration in specific age groups (higher = more stress)
        
        Weights: w1=0.5, w2=0.3, w3=0.2
        
        Higher BSI indicates regions under biometric system stress
        
        Args:
            biometric_df: Biometric update data
            window: Rolling window for volatility calculation
        
        Returns:
            DataFrame with BSI column
        """
        df = biometric_df.copy()
        
        # Ensure total_bio_updates exists
        bio_cols = [col for col in df.columns if col.startswith('bio_age')]
        if 'total_bio_updates' not in df.columns and bio_cols:
            df['total_bio_updates'] = df[bio_cols].sum(axis=1)
        
        # Calculate update rate (normalized by max)
        df['bio_update_rate'] = df['total_bio_updates'] / df['total_bio_updates'].max()
        
        # Calculate volatility
        df = AadhaarFeatureEngineer.calculate_volatility_index(
            df, 'total_bio_updates', window=window, group_cols=['state', 'district']
        )
        df['bio_volatility_norm'] = df['total_bio_updates_volatility_6m'] / df['total_bio_updates_volatility_6m'].max()
        df['bio_volatility_norm'] = df['bio_volatility_norm'].fillna(0)
        
        # Calculate age concentration (Herfindahl index)
        if bio_cols:
            total = df[bio_cols].sum(axis=1).replace(0, 1)
            age_shares = df[bio_cols].div(total, axis=0)
            df['age_concentration'] = (age_shares ** 2).sum(axis=1)
        else:
            df['age_concentration'] = 0
        
        # Calculate BSI
        w1, w2, w3 = 0.5, 0.3, 0.2
        df['biometric_stress_index'] = (
            w1 * df['bio_update_rate'] +
            w2 * df['bio_volatility_norm'] +
            w3 * df['age_concentration']
        ) * 100
        
        return df
    
    @staticmethod
    def calculate_demographic_volatility_index(demographic_df, window=6):
        """
        Calculate Demographic Volatility Index (DVI)
        
        Formula: DVI = w1 * (Demo_Update_Frequency) + w2 * (Demo_Volatility)
        
        Where:
        - Demo_Update_Frequency: Normalized demographic update rate
        - Demo_Volatility: Volatility of demographic updates
        
        Weights: w1=0.6, w2=0.4
        
        Higher DVI indicates unstable demographic patterns
        
        Args:
            demographic_df: Demographic update data
            window: Rolling window for volatility calculation
        
        Returns:
            DataFrame with DVI column
        """
        df = demographic_df.copy()
        
        # Ensure total_demo_updates exists
        demo_cols = [col for col in df.columns if col.startswith('demo_age')]
        if 'total_demo_updates' not in df.columns and demo_cols:
            df['total_demo_updates'] = df[demo_cols].sum(axis=1)
        
        # Calculate update frequency (normalized)
        df['demo_update_freq'] = df['total_demo_updates'] / df['total_demo_updates'].max()
        
        # Calculate volatility
        df = AadhaarFeatureEngineer.calculate_volatility_index(
            df, 'total_demo_updates', window=window, group_cols=['state', 'district']
        )
        df['demo_volatility_norm'] = df['total_demo_updates_volatility_6m'] / df['total_demo_updates_volatility_6m'].max()
        df['demo_volatility_norm'] = df['demo_volatility_norm'].fillna(0)
        
        # Calculate DVI
        w1, w2 = 0.6, 0.4
        df['demographic_volatility_index'] = (
            w1 * df['demo_update_freq'] +
            w2 * df['demo_volatility_norm']
        ) * 100
        
        return df
    
    @staticmethod
    def calculate_aadhaar_fragility_index(merged_df):
        """
        Calculate Aadhaar Fragility Index (AFI) - Composite Metric
        
        Formula: AFI = w1*BSI + w2*DVI + w3*(Update_Ratio) + w4*(Growth_Volatility)
        
        Where:
        - BSI: Biometric Stress Index
        - DVI: Demographic Volatility Index
        - Update_Ratio: Update-to-Enrolment Ratio (normalized)
        - Growth_Volatility: Volatility of enrolment growth (normalized)
        
        Weights: w1=0.3, w2=0.25, w3=0.25, w4=0.2
        
        AFI Score Interpretation:
        - 0-25: Low Fragility (Stable)
        - 25-50: Moderate Fragility (Monitor)
        - 50-75: High Fragility (Intervention Needed)
        - 75-100: Critical Fragility (Urgent Action Required)
        
        Args:
            merged_df: DataFrame with all metrics
        
        Returns:
            DataFrame with AFI column
        """
        df = merged_df.copy()
        
        # Normalize components to 0-100 scale
        if 'biometric_stress_index' in df.columns:
            bsi_norm = df['biometric_stress_index']
        else:
            bsi_norm = 0
        
        if 'demographic_volatility_index' in df.columns:
            dvi_norm = df['demographic_volatility_index']
        else:
            dvi_norm = 0
        
        if 'update_to_enrolment_ratio' in df.columns:
            update_ratio_norm = (df['update_to_enrolment_ratio'] / df['update_to_enrolment_ratio'].max()) * 100
            update_ratio_norm = update_ratio_norm.fillna(0)
        else:
            update_ratio_norm = 0
        
        if 'total_enrolments_volatility_6m' in df.columns:
            growth_vol_norm = (df['total_enrolments_volatility_6m'] / df['total_enrolments_volatility_6m'].max()) * 100
            growth_vol_norm = growth_vol_norm.fillna(0)
        else:
            growth_vol_norm = 0
        
        # Calculate AFI
        w1, w2, w3, w4 = 0.3, 0.25, 0.25, 0.2
        df['aadhaar_fragility_index'] = (
            w1 * bsi_norm +
            w2 * dvi_norm +
            w3 * update_ratio_norm +
            w4 * growth_vol_norm
        )
        
        # Categorize fragility
        df['fragility_category'] = pd.cut(
            df['aadhaar_fragility_index'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Moderate', 'High', 'Critical'],
            include_lowest=True
        )
        
        return df
    
    @staticmethod
    def create_all_features(enrolment_df, demographic_df, biometric_df):
        """
        Create all features in one go
        
        Returns:
            Tuple of (enrolment_featured, demographic_featured, biometric_featured, merged_featured)
        """
        print("Creating all features...")
        
        # Enrolment features
        print("  Processing enrolment features...")
        enrol = enrolment_df.copy()
        enrol = AadhaarFeatureEngineer.calculate_growth_rate(enrol, 'total_enrolments', periods=1)
        enrol = AadhaarFeatureEngineer.calculate_rolling_average(enrol, 'total_enrolments', windows=[3, 6, 12])
        enrol = AadhaarFeatureEngineer.calculate_volatility_index(enrol, 'total_enrolments', window=6)
        enrol = AadhaarFeatureEngineer.calculate_seasonal_index(enrol, 'total_enrolments')
        
        # Demographic features
        print("  Processing demographic features...")
        demo = demographic_df.copy()
        demo_cols = [col for col in demo.columns if col.startswith('demo_age')]
        if demo_cols:
            demo['total_demo_updates'] = demo[demo_cols].sum(axis=1)
            demo = AadhaarFeatureEngineer.calculate_demographic_volatility_index(demo, window=6)
        
        # Biometric features
        print("  Processing biometric features...")
        bio = biometric_df.copy()
        bio_cols = [col for col in bio.columns if col.startswith('bio_age')]
        if bio_cols:
            bio['total_bio_updates'] = bio[bio_cols].sum(axis=1)
            bio = AadhaarFeatureEngineer.calculate_biometric_stress_index(bio, window=6)
        
        # Merge and create composite metrics
        print("  Creating composite metrics...")
        merged = AadhaarFeatureEngineer.calculate_update_to_enrolment_ratio(enrol, demo, bio)
        
        # Add BSI and DVI to merged
        if 'biometric_stress_index' in bio.columns:
            bio_agg = bio.groupby(['date', 'state', 'district'])['biometric_stress_index'].mean().reset_index()
            merged = merged.merge(bio_agg, on=['date', 'state', 'district'], how='left')
        
        if 'demographic_volatility_index' in demo.columns:
            demo_agg = demo.groupby(['date', 'state', 'district'])['demographic_volatility_index'].mean().reset_index()
            merged = merged.merge(demo_agg, on=['date', 'state', 'district'], how='left')
        
        # Calculate AFI
        merged = AadhaarFeatureEngineer.calculate_aadhaar_fragility_index(merged)
        
        print("  Feature engineering complete!")
        
        return enrol, demo, bio, merged


if __name__ == "__main__":
    print("Feature Engineering Module Ready")
    print("\nAvailable Features:")
    print("1. Growth Rate (MoM, YoY)")
    print("2. Rolling Averages (3m, 6m, 12m)")
    print("3. Volatility Index")
    print("4. Seasonal Index")
    print("5. Update-to-Enrolment Ratio")
    print("6. Biometric Stress Index (BSI)")
    print("7. Demographic Volatility Index (DVI)")
    print("8. Aadhaar Fragility Index (AFI)")
