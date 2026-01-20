"""
Data Loading and Cleaning Module
Handles loading, cleaning, and merging of Aadhaar datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AadhaarDataLoader:
    """Load and clean Aadhaar datasets"""
    
    def __init__(self, data_dir='../data/raw'):
        self.data_dir = Path(data_dir)
        self.enrolment_df = None
        self.demographic_df = None
        self.biometric_df = None
        self.merged_df = None
        
    def load_enrolment_data(self):
        """Load Aadhaar enrolment dataset from multiple CSV files"""
        print("Loading Enrolment Data...")
        enrolment_path = self.data_dir / 'api_data_aadhar_enrolment'
        
        # Get all CSV files
        csv_files = sorted(enrolment_path.glob('*.csv'))
        
        # Load and concatenate
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  Loaded {file.name}: {len(df)} rows")
        
        self.enrolment_df = pd.concat(dfs, ignore_index=True)
        print(f"Total Enrolment Records: {len(self.enrolment_df)}")
        print(f"Columns: {list(self.enrolment_df.columns)}\n")
        
        return self.enrolment_df
    
    def load_demographic_data(self):
        """Load Aadhaar demographic update dataset from multiple CSV files"""
        print("Loading Demographic Update Data...")
        demographic_path = self.data_dir / 'api_data_aadhar_demographic'
        
        # Get all CSV files
        csv_files = sorted(demographic_path.glob('*.csv'))
        
        # Load and concatenate
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  Loaded {file.name}: {len(df)} rows")
        
        self.demographic_df = pd.concat(dfs, ignore_index=True)
        print(f"Total Demographic Update Records: {len(self.demographic_df)}")
        print(f"Columns: {list(self.demographic_df.columns)}\n")
        
        return self.demographic_df
    
    def load_biometric_data(self):
        """Load Aadhaar biometric update dataset from multiple CSV files"""
        print("Loading Biometric Update Data...")
        biometric_path = self.data_dir / 'api_data_aadhar_biometric'
        
        # Get all CSV files
        csv_files = sorted(biometric_path.glob('*.csv'))
        
        # Load and concatenate
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  Loaded {file.name}: {len(df)} rows")
        
        self.biometric_df = pd.concat(dfs, ignore_index=True)
        print(f"Total Biometric Update Records: {len(self.biometric_df)}")
        print(f"Columns: {list(self.biometric_df.columns)}\n")
        
        return self.biometric_df
    
    def clean_enrolment_data(self):
        """Clean and standardize enrolment data"""
        print("Cleaning Enrolment Data...")
        df = self.enrolment_df.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Add derived columns
        df['total_enrolments'] = df.get('age_0_5', 0) + df.get('age_5_17', 0) + df.get('age_18_greater', 0)
        
        self.enrolment_df = df
        print(f"Cleaned Enrolment Data: {len(df)} rows\n")
        
        return df
    
    def clean_demographic_data(self):
        """Clean and standardize demographic update data"""
        print("Cleaning Demographic Update Data...")
        df = self.demographic_df.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        self.demographic_df = df
        print(f"Cleaned Demographic Data: {len(df)} rows\n")
        
        return df
    
    def clean_biometric_data(self):
        """Clean and standardize biometric update data"""
        print("Cleaning Biometric Update Data...")
        df = self.biometric_df.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Add total biometric updates
        bio_cols = [col for col in df.columns if col.startswith('bio_age')]
        if bio_cols:
            df['total_biometric_updates'] = df[bio_cols].sum(axis=1)
        
        self.biometric_df = df
        print(f"Cleaned Biometric Data: {len(df)} rows\n")
        
        return df
    
    def load_all_data(self):
        """Load and clean all datasets"""
        self.load_enrolment_data()
        self.load_demographic_data()
        self.load_biometric_data()
        
        self.clean_enrolment_data()
        self.clean_demographic_data()
        self.clean_biometric_data()
        
        return self.enrolment_df, self.demographic_df, self.biometric_df
    
    def save_cleaned_data(self, output_dir='../data/processed'):
        """Save cleaned datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.enrolment_df is not None:
            self.enrolment_df.to_csv(output_path / 'enrolment_cleaned.csv', index=False)
            print(f"Saved: {output_path / 'enrolment_cleaned.csv'}")
        
        if self.demographic_df is not None:
            self.demographic_df.to_csv(output_path / 'demographic_cleaned.csv', index=False)
            print(f"Saved: {output_path / 'demographic_cleaned.csv'}")
        
        if self.biometric_df is not None:
            self.biometric_df.to_csv(output_path / 'biometric_cleaned.csv', index=False)
            print(f"Saved: {output_path / 'biometric_cleaned.csv'}")


def quick_data_summary(df, name="Dataset"):
    """Generate quick summary statistics"""
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nFirst Few Rows:\n{df.head()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the data loader
    loader = AadhaarDataLoader(data_dir='../data/raw')
    enrolment, demographic, biometric = loader.load_all_data()
    
    # Show summaries
    quick_data_summary(enrolment, "Enrolment Data")
    quick_data_summary(demographic, "Demographic Update Data")
    quick_data_summary(biometric, "Biometric Update Data")
    
    # Save cleaned data
    loader.save_cleaned_data()
