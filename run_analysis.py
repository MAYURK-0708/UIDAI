"""
Run the complete Aadhaar Observatory analysis
This script executes the full pipeline without requiring Jupyter
"""

import sys
sys.path.append('src')

print("="*80)
print("AADHAAR OBSERVATORY - COMPLETE ANALYSIS")
print("="*80)

# Step 1: Load and clean data
print("\n[1/6] Loading and cleaning datasets...")
try:
    from data_loader import AadhaarDataLoader
    
    loader = AadhaarDataLoader(data_dir='.')
    enrolment_df, demographic_df, biometric_df = loader.load_all_data()
    loader.save_cleaned_data(output_dir='data/processed')
    
    print(f"✓ Enrolment data: {len(enrolment_df):,} records")
    print(f"✓ Demographic data: {len(demographic_df):,} records")
    print(f"✓ Biometric data: {len(biometric_df):,} records")
    print("✓ Cleaned data saved to data/processed/")
except Exception as e:
    print(f"✗ Error in data loading: {str(e)}")
    sys.exit(1)

# Step 2: Feature engineering
print("\n[2/6] Creating advanced features...")
try:
    from feature_engineering import AadhaarFeatureEngineer
    
    enrol_featured, demo_featured, bio_featured, merged_featured = \
        AadhaarFeatureEngineer.create_all_features(enrolment_df, demographic_df, biometric_df)
    
    print(f"✓ Created features for {len(merged_featured):,} records")
    
    # Save featured data
    merged_featured.to_csv('data/processed/merged_with_features.csv', index=False)
    print("✓ Featured data saved")
    
    # Show AFI distribution if available
    if 'aadhaar_fragility_index' in merged_featured.columns:
        fragility_counts = merged_featured['fragility_category'].value_counts()
        print("\nFragility Distribution:")
        for category, count in fragility_counts.items():
            print(f"  {category}: {count:,} ({count/len(merged_featured)*100:.1f}%)")
except Exception as e:
    print(f"✗ Error in feature engineering: {str(e)}")
    import traceback
    traceback.print_exc()

# Step 3: Anomaly detection
print("\n[3/6] Detecting anomalies...")
try:
    from anomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector(threshold=3)
    
    # Run anomaly detection on enrolment data
    feature_cols = ['total_enrolments']
    if 'total_enrolments_growth_rate_1m' in enrol_featured.columns:
        feature_cols.append('total_enrolments_growth_rate_1m')
    if 'total_enrolments_volatility_6m' in enrol_featured.columns:
        feature_cols.append('total_enrolments_volatility_6m')
    
    enrol_anomalies = detector.detect_all_anomalies(
        enrol_featured,
        value_col='total_enrolments',
        feature_cols=feature_cols
    )
    
    # Save anomaly data
    enrol_anomalies.to_csv('data/processed/enrolment_with_anomalies.csv', index=False)
    print("✓ Anomaly detection complete")
    
    # Show summary
    summary = detector.get_anomaly_summary()
    print("\nAnomaly Summary:")
    print(summary)
except Exception as e:
    print(f"✗ Error in anomaly detection: {str(e)}")
    import traceback
    traceback.print_exc()

# Step 4: Predictive modeling
print("\n[4/6] Building predictive models...")
try:
    from predictive_models import DemandForecaster
    
    forecaster = DemandForecaster()
    
    # Prepare time series data (top 5 states)
    top_states = enrol_featured.groupby('state')['total_enrolments'].sum().nlargest(5).index
    ts_data = enrol_featured[enrol_featured['state'].isin(top_states)]
    ts_data = forecaster.prepare_time_series(ts_data, 'total_enrolments', group_cols=['state'])
    
    print(f"✓ Prepared time series data: {len(ts_data):,} records")
    
    # Try ARIMA forecasting
    try:
        arima_forecast = forecaster.arima_forecast(
            ts_data,
            value_col='total_enrolments',
            periods=6,
            order=(1, 1, 1),
            group_col='state'
        )
        
        if len(arima_forecast) > 0:
            arima_forecast.to_csv('outputs/models/arima_forecast.csv', index=False)
            print(f"✓ ARIMA forecast saved: {len(arima_forecast)} records")
    except Exception as e:
        print(f"⚠ ARIMA forecasting skipped: {str(e)}")
    
    # Try Prophet forecasting (if available)
    try:
        prophet_forecast = forecaster.prophet_forecast(
            ts_data,
            value_col='total_enrolments',
            periods=6,
            group_col='state'
        )
        
        if len(prophet_forecast) > 0:
            prophet_forecast.to_csv('outputs/models/prophet_forecast.csv', index=False)
            print(f"✓ Prophet forecast saved: {len(prophet_forecast)} records")
    except Exception as e:
        print(f"⚠ Prophet forecasting skipped: {str(e)}")
    
except Exception as e:
    print(f"⚠ Predictive modeling partially completed: {str(e)}")

# Step 5: Generate visualizations
print("\n[5/6] Generating visualizations...")
try:
    from visualizations import AadhaarVisualizer
    import matplotlib.pyplot as plt
    
    viz = AadhaarVisualizer(output_dir='outputs/figures')
    
    # Distribution plot
    if 'aadhaar_fragility_index' in merged_featured.columns:
        viz.plot_distribution(merged_featured, 'aadhaar_fragility_index',
                            title='Aadhaar Fragility Index Distribution',
                            save_name='afi_distribution')
        plt.close('all')
        print("✓ AFI distribution plot saved")
    
    # Risk ranking
    if 'aadhaar_fragility_index' in merged_featured.columns:
        latest_data = merged_featured.sort_values('date').groupby(['state', 'district']).tail(1)
        viz.plot_risk_ranking(latest_data, risk_col='aadhaar_fragility_index',
                            top_n=20, save_name='risk_ranking')
        print("✓ Risk ranking plot saved")
    
    print("✓ Visualizations generated")
except Exception as e:
    print(f"⚠ Visualization generation partially completed: {str(e)}")

# Step 6: Summary statistics
print("\n[6/6] Generating summary statistics...")
try:
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    
    print(f"\nTotal Enrolments Analyzed: {enrolment_df['total_enrolments'].sum():,}")
    print(f"States Covered: {enrolment_df['state'].nunique()}")
    print(f"Districts Covered: {enrolment_df['district'].nunique()}")
    print(f"Date Range: {enrolment_df['date'].min()} to {enrolment_df['date'].max()}")
    
    if 'is_anomaly' in enrol_anomalies.columns:
        anomaly_count = enrol_anomalies['is_anomaly'].sum()
        print(f"\nAnomalies Detected: {anomaly_count:,}")
        print(f"Anomaly Rate: {anomaly_count/len(enrol_anomalies)*100:.2f}%")
    
    if 'aadhaar_fragility_index' in merged_featured.columns:
        critical_regions = merged_featured[merged_featured['fragility_category'] == 'Critical']
        high_regions = merged_featured[merged_featured['fragility_category'] == 'High']
        print(f"\nCritical Fragility Regions: {len(critical_regions):,}")
        print(f"High Fragility Regions: {len(high_regions):,}")
    
    print("\n" + "="*80)
    print("All outputs saved to:")
    print("  • Processed data: data/processed/")
    print("  • Visualizations: outputs/figures/")
    print("  • Models: outputs/models/")
    print("\nNext step: Launch dashboard with 'streamlit run dashboard/app.py'")
    print("="*80)
    
except Exception as e:
    print(f"⚠ Summary generation error: {str(e)}")

print("\n✓ Analysis pipeline complete!")
