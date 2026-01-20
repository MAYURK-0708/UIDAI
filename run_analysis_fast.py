"""
Run the complete Aadhaar Observatory analysis (FAST VERSION)
This script executes the pipeline with optimized settings
"""

import sys
sys.path.append('src')

print("="*80)
print("AADHAAR OBSERVATORY - FAST ANALYSIS")
print("="*80)

# Step 1: Load and clean data
print("\n[1/5] Loading and cleaning datasets...")
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
print("\n[2/5] Creating advanced features...")
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

# Step 3: Quick anomaly detection (Z-score only)
print("\n[3/5] Quick anomaly detection (Z-score method only)...")
try:
    from anomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector(threshold=3)
    
    # Run only Z-score detection (fastest method)
    feature_cols = ['total_enrolments']
    if 'total_enrolments_growth_rate_1m' in enrol_featured.columns:
        feature_cols.append('total_enrolments_growth_rate_1m')
    
    enrol_anomalies = detector.rolling_zscore_detection(
        enrol_featured,
        value_col='total_enrolments'
    )
    
    # Save anomaly data
    enrol_anomalies.to_csv('data/processed/enrolment_with_anomalies.csv', index=False)
    print("✓ Quick anomaly detection complete")
    
    anomaly_count = enrol_anomalies['anomaly_zscore'].sum()
    print(f"  Anomalies found: {anomaly_count:,} ({anomaly_count/len(enrol_anomalies)*100:.2f}%)")
    
except Exception as e:
    print(f"✗ Error in anomaly detection: {str(e)}")
    import traceback
    traceback.print_exc()

# Step 4: Generate visualizations
print("\n[4/5] Generating visualizations...")
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

# Step 5: Summary statistics
print("\n[5/5] Generating summary statistics...")
try:
    print("\n" + "="*80)
    print("FAST ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    
    print(f"\nTotal Enrolments Analyzed: {enrolment_df['total_enrolments'].sum():,}")
    print(f"States Covered: {enrolment_df['state'].nunique()}")
    print(f"Districts Covered: {enrolment_df['district'].nunique()}")
    print(f"Date Range: {enrolment_df['date'].min()} to {enrolment_df['date'].max()}")
    
    if 'anomaly_zscore' in enrol_anomalies.columns:
        anomaly_count = enrol_anomalies['anomaly_zscore'].sum()
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
    print("\nNext step: Launch dashboard with 'streamlit run dashboard/app.py'")
    print("="*80)
    
except Exception as e:
    print(f"⚠ Summary generation error: {str(e)}")

print("\n✓ Fast analysis pipeline complete!")
